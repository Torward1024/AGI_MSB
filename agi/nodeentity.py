"""
AGI Neural Node (Neocortical Column Analog)
Implements a computational unit with a mini-neural network, local memory,
confidence assessment, and learning capabilities.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from common.base.baseentity import BaseEntity
from common.utils.logging_setup import logger


@dataclass
class MemoryPattern:
    """Individual memory pattern with metadata"""
    vector: np.ndarray  # Pattern embedding
    label: str  # Semantic label/tag
    confidence: float  # Storage confidence
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0


class NodeMemory:
    """Local memory system using scikit-learn for similarity search"""
    
    def __init__(self, 
                 max_capacity: int = 1000,
                 search_algorithm: str = 'auto',
                 metric: str = 'cosine'):
        """
        Initialize node memory.
        
        Args:
            max_capacity: Maximum number of patterns to store
            search_algorithm: 'brute', 'kd_tree', 'ball_tree', or 'auto'
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        self.max_capacity = max_capacity
        self.search_algorithm = search_algorithm
        self.metric = metric
        self.patterns: List[MemoryPattern] = []
        self.vectors: np.ndarray = np.empty((0, 0))
        self.index: Optional[NearestNeighbors] = None
        
        # Statistics
        self.storage_count = 0
        self.retrieval_count = 0
        self.last_accessed = datetime.now()
    
    def add_pattern(self, 
                    vector: np.ndarray,
                    label: str,
                    confidence: float = 1.0,
                    metadata: Optional[Dict] = None) -> str:
        """Add a pattern to memory with automatic indexing"""
        
        # Check capacity
        if len(self.patterns) >= self.max_capacity:
            self._evict_least_used()
        
        # Create pattern
        pattern = MemoryPattern(
            vector=vector.copy(),
            label=label,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.patterns.append(pattern)
        
        # Update index
        if self.vectors.shape[0] == 0:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        
        # Rebuild index if needed (can be optimized for batch additions)
        self._rebuild_index()
        
        pattern_id = f"mem_{self.storage_count}_{int(datetime.now().timestamp())}"
        self.storage_count += 1
        
        logger.debug(f"Stored pattern {pattern_id} in node memory")
        return pattern_id
    
    def query_similar(self, 
                      query_vector: np.ndarray,
                      k: int = 5,
                      threshold: float = 0.7) -> List[Tuple[MemoryPattern, float]]:
        """Find k most similar patterns above similarity threshold"""
        
        if len(self.patterns) == 0 or self.index is None:
            return []
        
        # Ensure query is 2D
        query_vector = query_vector.reshape(1, -1)
        
        # Find nearest neighbors
        try:
            distances, indices = self.index.kneighbors(query_vector, n_neighbors=min(k, len(self.patterns)))
        except ValueError as e:
            logger.error(f"Error querying memory: {e}")
            return []
        
        # Convert distances to similarities (cosine similarity for cosine distance)
        if self.metric == 'cosine':
            similarities = 1 - distances.flatten()
        else:
            # For euclidean: similarity = 1 / (1 + distance)
            similarities = 1 / (1 + distances.flatten())
        
        # Filter by threshold and collect results
        results = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= threshold:
                pattern = self.patterns[idx]
                pattern.access_count += 1
                results.append((pattern, float(sim)))
        
        self.retrieval_count += 1
        self.last_accessed = datetime.now()
        
        if results:
            logger.debug(f"Memory query returned {len(results)} matches")
        
        return results
    
    def _rebuild_index(self):
        """Rebuild the NearestNeighbors index"""
        if self.vectors.shape[0] == 0:
            self.index = None
            return
        
        try:
            self.index = NearestNeighbors(
                n_neighbors=min(10, self.vectors.shape[0]),
                algorithm=self.search_algorithm,
                metric=self.metric
            )
            self.index.fit(self.vectors)
        except Exception as e:
            logger.error(f"Failed to rebuild memory index: {e}")
            self.index = None
    
    def _evict_least_used(self):
        """Evict least frequently accessed pattern (LRU strategy)"""
        if not self.patterns:
            return
        
        # Find pattern with minimum access count
        min_idx = min(range(len(self.patterns)), 
                     key=lambda i: self.patterns[i].access_count)
        
        evicted = self.patterns.pop(min_idx)
        self.vectors = np.delete(self.vectors, min_idx, axis=0)
        
        logger.debug(f"Evicted pattern with label '{evicted.label}' from memory")
        self._rebuild_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "pattern_count": len(self.patterns),
            "storage_count": self.storage_count,
            "retrieval_count": self.retrieval_count,
            "last_accessed": self.last_accessed,
            "vector_dim": self.vectors.shape[1] if self.vectors.shape[0] > 0 else 0,
            "memory_usage_mb": self.vectors.nbytes / (1024 * 1024)
        }
    
    def clear(self):
        """Clear all memory patterns"""
        self.patterns.clear()
        self.vectors = np.empty((0, 0))
        self.index = None
        logger.debug("Node memory cleared")


class ConfidenceAssessor(nn.Module):
    """Small neural network for confidence assessment"""
    
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output confidence score between 0 and 1"""
        return self.network(x)


class MiniNeuralNetwork(nn.Module):
    """Configurable mini-neural network for node processing"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dims: List[int] = None,
                 network_type: str = "mlp"):
        """
        Initialize mini-neural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            network_type: 'mlp' or 'transformer'
        """
        super().__init__()
        
        self.network_type = network_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [1024, 768, 512] if network_type == "mlp" else [768]
        
        if network_type == "mlp":
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
            
        elif network_type == "transformer":
            # Lightweight transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            )
            self.network = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.output_projection = nn.Linear(input_dim, output_dim)
        
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        if self.network_type == "mlp":
            return self.network(x)
        else:
            # Transformer expects [batch, seq_len, features]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            transformer_out = self.network(x)
            return self.output_projection(transformer_out).squeeze(1)


class NodeEntity(BaseEntity):
    """
    AGI Neural Node - fundamental computational unit.
    
    Each node contains:
    1. A mini-neural network for pattern transformation
    2. Local memory system for knowledge storage
    3. Confidence assessor for input relevance scoring
    4. Learning mechanisms for self-improvement
    """
    
    # === Объявляем все публичные атрибуты для BaseEntity ===
    specialization: str
    confidence_threshold: float
    input_slots: Dict[str, str]
    output_slots: Dict[str, str]
    activation_count: int
    last_activated: datetime
    energy_level: float
    input_dim: int
    output_dim: int
    network_type: str
    memory_capacity: int
    
    def __init__(self,
                 name: str,
                 specialization: str = "general",
                 input_dim: int = 768,
                 output_dim: int = 768,
                 confidence_threshold: float = 0.3,
                 network_type: str = "mlp",
                 memory_capacity: int = 1000,
                 isactive: bool = True,
                 use_cache: bool = False,  # Добавлен параметр use_cache
                 **kwargs):
        """
        Initialize a NodeEntity.
        
        Args:
            name: Unique name for the node
            specialization: Domain specialization
            input_dim: Input vector dimension
            output_dim: Output vector dimension
            confidence_threshold: Minimum confidence for activation
            network_type: Type of neural network ('mlp' or 'transformer')
            memory_capacity: Maximum patterns in memory
            isactive: Whether node is active
            use_cache: Enable caching for to_dict()
            **kwargs: Additional BaseEntity parameters
        """
        
        # Сначала вызываем super().__init__ с критическими полями
        super().__init__(
            name=name,
            isactive=isactive,
            use_cache=use_cache,
            **kwargs
        )
        
        # Теперь устанавливаем остальные атрибуты
        self.specialization = specialization
        self.confidence_threshold = confidence_threshold
        self.input_slots = {}
        self.output_slots = {}
        self.activation_count = 0
        self.last_activated = datetime.now()
        self.energy_level = 1.0  # Start with full energy
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.memory_capacity = memory_capacity
        
        # Initialize neural components (private attributes, not in _fields)
        # Mini-neural network
        self._network = MiniNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            network_type=network_type
        )
        
        # Local memory system
        self._memory = NodeMemory(max_capacity=memory_capacity)
        
        # Confidence assessor
        self._confidence_assessor = ConfidenceAssessor(input_dim=input_dim)
        
        # Optimizer for local learning
        self._optimizer = optim.Adam(
            list(self._network.parameters()) + 
            list(self._confidence_assessor.parameters()),
            lr=0.001,
            weight_decay=1e-5
        )
        
        # Learning state (private attributes)
        self._learning_rate = 0.001
        self._reward_history: List[float] = []
        self._error_history: List[float] = []
        self._success_count = 0
        self._total_activations = 0
        
        logger.info(f"Created NodeEntity '{name}' with specialization '{specialization}'")
    
    def forward(self, 
                input_vector: Union[np.ndarray, torch.Tensor],
                use_memory: bool = True) -> Dict[str, Any]:
        """
        Main forward pass through the node.
        
        Args:
            input_vector: Input vector (numpy or torch)
            use_memory: Whether to query memory before processing
            
        Returns:
            Dictionary containing output, confidence, and metadata
        """
        
        if not self.isactive:
            logger.warning(f"Node '{self.name}' is inactive")
            return self._build_inactive_response()
        
        # Convert to torch if needed
        if isinstance(input_vector, np.ndarray):
            input_tensor = torch.FloatTensor(input_vector)
        else:
            input_tensor = input_vector
        
        # Ensure correct shape
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Calculate confidence - set confidence assessor to eval mode
        self._confidence_assessor.eval()
        with torch.no_grad():
            confidence = self._confidence_assessor(input_tensor).item()
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Node '{self.name}' confidence {confidence:.3f} < threshold {self.confidence_threshold}")
            return self._build_low_confidence_response(confidence)
        
        # Query memory for similar patterns
        memory_results = []
        if use_memory and self._memory.patterns:
            input_np = input_tensor.detach().numpy()
            memory_results = self._memory.query_similar(
                input_np,
                k=3,
                threshold=0.6
            )
        
        # Process through neural network - set to eval mode
        self._network.eval()
        with torch.no_grad():
            output_tensor = self._network(input_tensor)
        
        # Convert outputs
        output_np = output_tensor.detach().numpy()
        
        # Update activation statistics
        self.activation_count += 1
        self.last_activated = datetime.now()
        self._total_activations += 1
        
        # Build response
        response = {
            "output": output_np,
            "confidence": confidence,
            "node_name": self.name,
            "specialization": self.specialization,
            "memory_matches": len(memory_results),
            "activation_count": self.activation_count,
            "energy_level": self.energy_level,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add top memory match if available
        if memory_results:
            best_pattern, similarity = memory_results[0]
            response["memory_match"] = {
                "label": best_pattern.label,
                "similarity": float(similarity),
                "confidence": best_pattern.confidence
            }
        
        logger.debug(f"Node '{self.name}' processed input with confidence {confidence:.3f}")
        return response
    
    def learn_from_experience(self,
                              input_vector: Union[np.ndarray, torch.Tensor],
                              target_output: Union[np.ndarray, torch.Tensor],
                              reward: float = 1.0) -> Dict[str, Any]:
        """
        Learn from a single experience using local backpropagation.
        
        Args:
            input_vector: Input vector
            target_output: Target output vector
            reward: Reinforcement learning reward
            
        Returns:
            Dictionary with learning statistics
        """
        
        if not self.isactive:
            return {"status": "inactive", "node": self.name}
        
        # Convert to torch tensors
        if isinstance(input_vector, np.ndarray):
            input_tensor = torch.FloatTensor(input_vector)
        else:
            input_tensor = input_vector
        
        if isinstance(target_output, np.ndarray):
            target_tensor = torch.FloatTensor(target_output)
        else:
            target_tensor = target_output
        
        # Ensure correct shapes
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)
        
        # Forward pass - set network to train mode
        self._network.train()
        output_tensor = self._network(input_tensor)
        
        # Calculate loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(output_tensor, target_tensor)
        
        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        # Update learning state
        loss_value = loss.item()
        self._error_history.append(loss_value)
        self._reward_history.append(reward)
        
        if reward > 0.7:
            self._success_count += 1
        
        # Store in memory if successful
        if reward > 0.5 and loss_value < 0.1:
            self.add_to_memory(
                vector=input_tensor.detach().numpy(),
                label=f"learned_{len(self._memory.patterns)}",
                confidence=reward
            )
        
        # Update confidence assessor
        self._update_confidence_assessor(input_tensor, reward)
        
        # Consume energy
        self.energy_level = max(0.0, self.energy_level - 0.01)
        
        logger.debug(f"Node '{self.name}' learned, loss: {loss_value:.4f}, reward: {reward:.3f}")
        
        return {
            "node": self.name,
            "loss": loss_value,
            "reward": reward,
            "success_count": self._success_count,
            "energy_level": self.energy_level,
            "memory_size": len(self._memory.patterns)
        }
    
    def add_to_memory(self,
                      vector: np.ndarray,
                      label: str,
                      confidence: float = 1.0,
                      metadata: Optional[Dict] = None) -> str:
        """
        Add a pattern to the node's local memory.
        
        Args:
            vector: Pattern vector
            label: Semantic label
            confidence: Storage confidence
            metadata: Optional metadata
            
        Returns:
            Pattern ID
        """
        return self._memory.add_pattern(vector, label, confidence, metadata)
    
    def query_memory(self,
                     query_vector: np.ndarray,
                     k: int = 5,
                     threshold: float = 0.7) -> List[Tuple[MemoryPattern, float]]:
        """
        Query the node's memory for similar patterns.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors
            threshold: Similarity threshold
            
        Returns:
            List of (pattern, similarity) tuples
        """
        return self._memory.query_similar(query_vector, k, threshold)
    
    def _update_confidence_assessor(self, 
                                   input_tensor: torch.Tensor,
                                   reward: float):
        """Update confidence assessor based on reward"""
        
        # Simple update: if reward is high, confidence should be high
        target_confidence = torch.tensor([[reward]], dtype=torch.float32)
        
        # Forward pass - set to train mode
        self._confidence_assessor.train()
        predicted_confidence = self._confidence_assessor(input_tensor)
        
        # Calculate loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_confidence, target_confidence)
        
        # Backward pass (only for confidence assessor)
        optimizer = optim.Adam(self._confidence_assessor.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Set back to eval mode for future predictions
        self._confidence_assessor.eval()
    
    def _build_inactive_response(self) -> Dict[str, Any]:
        """Build response for inactive node"""
        return {
            "output": None,
            "confidence": 0.0,
            "node_name": self.name,
            "status": "inactive",
            "error": f"Node '{self.name}' is inactive"
        }
    
    def _build_low_confidence_response(self, confidence: float) -> Dict[str, Any]:
        """Build response for low confidence"""
        return {
            "output": None,
            "confidence": confidence,
            "node_name": self.name,
            "status": "low_confidence",
            "error": f"Confidence {confidence:.3f} < threshold {self.confidence_threshold}"
        }
    
    def connect_input(self, 
                      slot_name: str, 
                      source_node_name: str) -> None:
        """Connect an input slot to a source node"""
        self.input_slots[slot_name] = source_node_name
        logger.debug(f"Node '{self.name}' input slot '{slot_name}' connected to '{source_node_name}'")
    
    def connect_output(self,
                       slot_name: str,
                       target_node_name: str) -> None:
        """Connect an output slot to a target node"""
        self.output_slots[slot_name] = target_node_name
        logger.debug(f"Node '{self.name}' output slot '{slot_name}' connected to '{target_node_name}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive node statistics"""
        memory_stats = self._memory.get_stats()
        
        success_rate = 0.0
        if self._total_activations > 0:
            success_rate = self._success_count / self._total_activations
        
        return {
            "name": self.name,
            "specialization": self.specialization,
            "isactive": self.isactive,
            "activation_count": self.activation_count,
            "last_activated": self.last_activated,
            "energy_level": self.energy_level,
            "confidence_threshold": self.confidence_threshold,
            "success_rate": success_rate,
            "average_error": (np.mean(self._error_history) if self._error_history else 0.0),
            "average_reward": (np.mean(self._reward_history) if self._reward_history else 0.0),
            "memory": memory_stats,
            "input_slots": len(self.input_slots),
            "output_slots": len(self.output_slots),
            "network_type": self.network_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "memory_capacity": self.memory_capacity
        }
    
    def recharge_energy(self, amount: float = 0.1) -> float:
        """Recharge node's energy level"""
        self.energy_level = min(1.0, self.energy_level + amount)
        logger.debug(f"Node '{self.name}' energy recharged to {self.energy_level:.3f}")
        return self.energy_level
    
    def clear_memory(self) -> None:
        """Clear all patterns from memory"""
        self._memory.clear()
        logger.info(f"Node '{self.name}' memory cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        data = super().to_dict()
        
        # Convert last_activated to string if it's a datetime
        if "last_activated" in data and isinstance(data["last_activated"], datetime):
            data["last_activated"] = data["last_activated"].isoformat()
        
        # Add node-specific data
        data.update({
            "network_state": {
                "network": {k: v.cpu() for k, v in self._network.state_dict().items()},
                "confidence_assessor": {k: v.cpu() for k, v in self._confidence_assessor.state_dict().items()}
            },
            "memory_patterns": [
                {
                    "vector": pattern.vector.tolist(),
                    "label": pattern.label,
                    "confidence": pattern.confidence,
                    "timestamp": pattern.timestamp.isoformat(),
                    "metadata": pattern.metadata,
                    "access_count": pattern.access_count
                }
                for pattern in self._memory.patterns
            ],
            "learning_state": {
                "reward_history": self._reward_history,
                "error_history": self._error_history,
                "success_count": self._success_count,
                "total_activations": self._total_activations
            }
        })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeEntity':
        """Create node from dictionary (deserialization)"""
        
        # Extract all fields
        name = data.get("name", "unnamed_node")
        specialization = data.get("specialization", "general")
        input_dim = data.get("input_dim", 768)
        output_dim = data.get("output_dim", 768)
        confidence_threshold = data.get("confidence_threshold", 0.3)
        network_type = data.get("network_type", "mlp")
        memory_capacity = data.get("memory_capacity", 1000)
        isactive = data.get("isactive", True)
        use_cache = data.get("_use_cache", False)
        
        input_slots = data.get("input_slots", {})
        output_slots = data.get("output_slots", {})
        activation_count = data.get("activation_count", 0)
        
        # Handle datetime - поддержка строки или datetime объекта
        last_activated = data.get("last_activated")
        if last_activated is None:
            last_activated = datetime.now()
        elif isinstance(last_activated, str):
            last_activated = datetime.fromisoformat(last_activated)
        # Если last_activated уже является datetime, оставляем как есть
        
        energy_level = data.get("energy_level", 1.0)
        
        # Create node instance
        node = cls(
            name=name,
            specialization=specialization,
            input_dim=input_dim,
            output_dim=output_dim,
            confidence_threshold=confidence_threshold,
            network_type=network_type,
            memory_capacity=memory_capacity,
            isactive=isactive,
            use_cache=use_cache
        )
        
        # Set additional fields
        node.input_slots = input_slots
        node.output_slots = output_slots
        node.activation_count = activation_count
        node.last_activated = last_activated
        node.energy_level = energy_level
        
        # Extract and restore network state
        network_state = data.get("network_state", {})
        if network_state:
            if "network" in network_state:
                node._network.load_state_dict(network_state["network"])
            if "confidence_assessor" in network_state:
                node._confidence_assessor.load_state_dict(network_state["confidence_assessor"])
        
        # Extract and restore memory patterns
        memory_patterns = data.get("memory_patterns", [])
        node._memory.clear()
        for pattern_data in memory_patterns:
            node._memory.add_pattern(
                vector=np.array(pattern_data["vector"]),
                label=pattern_data["label"],
                confidence=pattern_data["confidence"],
                metadata=pattern_data.get("metadata", {})
            )
        
        # Extract and restore learning state
        learning_state = data.get("learning_state", {})
        if learning_state:
            node._reward_history = learning_state.get("reward_history", [])
            node._error_history = learning_state.get("error_history", [])
            node._success_count = learning_state.get("success_count", 0)
            node._total_activations = learning_state.get("total_activations", 0)
        
        # Set networks to eval mode by default
        node._network.eval()
        node._confidence_assessor.eval()
        
        logger.info(f"Restored NodeEntity '{node.name}' from dict")
        return node
    
    def __repr__(self) -> str:
        """String representation of NodeEntity"""
        stats = self.get_stats()
        return (
            f"NodeEntity(name='{self.name}', "
            f"specialization='{self.specialization}', "
            f"active={self.isactive}, "
            f"activations={self.activation_count}, "
            f"energy={self.energy_level:.2f}, "
            f"memory={stats['memory']['pattern_count']} patterns)"
        )