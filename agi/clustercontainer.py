"""
AGI Cluster Container (Neocortical Macrocolumn Analog) - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from common.base.basecontainer import BaseContainer
from common.base.baseentity import BaseEntity
from common.utils.logging_setup import logger

# Import AGI-specific entities
from .nodeentity import NodeEntity
from .edgeentity import EdgeEntity
from .knowledgechunk import KnowledgeChunk


class ClusterType(Enum):
    """Types of cognitive clusters based on specialization"""
    SENSORY_INPUT = "sensory_input"      # Обработка сенсорной информации
    MOTOR_OUTPUT = "motor_output"        # Двигательные команды
    LANGUAGE = "language"                # Языковая обработка
    VISUAL = "visual"                    # Визуальное восприятие
    AUDITORY = "auditory"                # Аудиальное восприятие
    LOGICAL = "logical"                  # Логические операции
    EMOTIONAL = "emotional"              # Эмоциональная обработка
    MEMORY = "memory"                    # Работа с памятью
    EXECUTIVE = "executive"              # Исполнительные функции
    ABSTRACT = "abstract"                # Абстрактное мышление
    INTEGRATIVE = "integrative"          # Интеграция информации


@dataclass
class ActivationPattern:
    """Pattern of activation within a cluster"""
    timestamp: datetime
    active_nodes: List[str]               # Active NodeEntity names
    activation_levels: Dict[str, float]   # Node -> activation level
    coherence_score: float               # How synchronized the activation is
    information_content: float           # Amount of information processed


class ClusterMetrics:
    """Metrics for monitoring cluster performance and health"""
    
    def __init__(self):
        self.activation_count = 0
        self.successful_processing = 0
        self.failed_processing = 0
        self.average_latency = 0.0
        self.energy_consumption = 0.0
        self.knowledge_growth = 0
        self.last_activation_time = datetime.now()
        
        # Quality metrics
        self.specialization_score = 0.5    # How specialized the cluster is
        self.coherence_score = 0.5         # Internal consistency
        self.efficiency_score = 0.5        # Energy vs output quality
        self.adaptability_score = 0.5      # Ability to learn new patterns
        
        # Temporal patterns
        self.activation_history: List[ActivationPattern] = []
    
    def update(self, 
               success: bool,
               latency: float,
               energy: float,
               activation_pattern: Optional[ActivationPattern] = None):
        """Update metrics after cluster activation"""
        
        self.activation_count += 1
        if success:
            self.successful_processing += 1
        else:
            self.failed_processing += 1
        
        # Update moving average for latency
        alpha = 0.1  # Smoothing factor
        if self.average_latency == 0.0:
            self.average_latency = latency
        else:
            self.average_latency = (alpha * latency + 
                                  (1 - alpha) * self.average_latency)
        
        self.energy_consumption += energy
        self.last_activation_time = datetime.now()
        
        if activation_pattern:
            self.activation_history.append(activation_pattern)
            # Keep only recent history (last 1000 activations)
            if len(self.activation_history) > 1000:
                self.activation_history.pop(0)
            
            # Update coherence score
            recent_patterns = self.activation_history[-100:] if len(self.activation_history) > 100 else self.activation_history
            if recent_patterns:
                self.coherence_score = np.mean([p.coherence_score for p in recent_patterns])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary metrics"""
        
        success_rate = 0.0
        if self.activation_count > 0:
            success_rate = self.successful_processing / self.activation_count
        
        # Calculate adaptability based on recent knowledge growth
        recent_growth = self.knowledge_growth / max(1, self.activation_count)
        self.adaptability_score = min(1.0, recent_growth * 10)
        
        return {
            "activation_count": self.activation_count,
            "success_rate": success_rate,
            "average_latency_ms": self.average_latency * 1000,
            "energy_consumption": self.energy_consumption,
            "specialization_score": self.specialization_score,
            "coherence_score": self.coherence_score,
            "efficiency_score": self.efficiency_score,
            "adaptability_score": self.adaptability_score,
            "knowledge_growth": self.knowledge_growth,
            "last_activation": self.last_activation_time.isoformat()
        }


class ClusterContainer(BaseContainer[BaseEntity]):
    """
    AGI Cluster Container - hierarchical container for cognitive processing units.
    
    Analogous to neocortical macrocolumns or brain regions, this container:
    1. Organizes NodeEntity instances into functional groups
    2. Manages connections (EdgeEntity) between nodes
    3. Can contain sub-clusters for fractal organization
    4. Specializes in specific cognitive functions
    
    The container provides:
    - Hierarchical organization (nodes -> clusters -> regions)
    - Specialized processing based on cluster type
    - Local memory and knowledge management
    - Activation propagation and synchronization
    - Growth and evolution mechanisms
    """
    
    # === Cluster-specific attributes ===
    cluster_type: ClusterType
    specialization_vector: np.ndarray  # 512-dim vector of specialization
    activation_threshold: float        # Minimum activation to propagate
    max_capacity: int                  # Maximum nodes in cluster
    
    # Connectivity
    input_clusters: List[str]          # Names of input clusters
    output_clusters: List[str]         # Names of output clusters
    intra_cluster_edges: List[str]     # EdgeEntity names within cluster
    inter_cluster_edges: List[str]     # EdgeEntity names to other clusters
    
    # State
    current_activation: float          # Current activation level (0-1)
    activation_history: List[float]    # History of activations
    energy_reserve: float              # Metabolic energy available
    
    # Metrics
    metrics: ClusterMetrics
    
    def __init__(self,
                 name: str,
                 cluster_type: Union[str, ClusterType] = ClusterType.INTEGRATIVE,
                 specialization_vector: Optional[np.ndarray] = None,
                 activation_threshold: float = 0.3,
                 max_capacity: int = 100,
                 input_clusters: Optional[List[str]] = None,
                 output_clusters: Optional[List[str]] = None,
                 isactive: bool = True,
                 use_cache: bool = False,
                 **kwargs):
        """
        Initialize a ClusterContainer.
        
        Args:
            name: Unique name for the cluster
            cluster_type: Type of cognitive function
            specialization_vector: Vector defining cluster specialization
            activation_threshold: Minimum activation to propagate
            max_capacity: Maximum number of nodes
            input_clusters: Connected input clusters
            output_clusters: Connected output clusters
            isactive: Whether cluster is active
            use_cache: Enable caching for to_dict()
            **kwargs: Additional BaseContainer parameters
        """
        
        # Convert cluster_type to enum if string
        if isinstance(cluster_type, str):
            try:
                cluster_type = ClusterType(cluster_type)
            except ValueError:
                logger.warning(f"Unknown cluster type '{cluster_type}', defaulting to INTEGRATIVE")
                cluster_type = ClusterType.INTEGRATIVE
        
        # Initialize specialization vector
        if specialization_vector is None:
            # Create random specialization vector
            specialization_vector = np.random.randn(512)
            # Normalize
            norm = np.linalg.norm(specialization_vector)
            if norm > 0:
                specialization_vector = specialization_vector / norm
        
        # Initialize BaseContainer with empty items
        super().__init__(
            name=name,
            items={},  # Start with empty items
            isactive=isactive,
            use_cache=use_cache,
            **kwargs
        )
        
        # Set cluster-specific attributes
        self.cluster_type = cluster_type
        self.specialization_vector = specialization_vector.copy()
        self.activation_threshold = activation_threshold
        self.max_capacity = max_capacity
        
        # Initialize connectivity
        self.input_clusters = input_clusters or []
        self.output_clusters = output_clusters or []
        self.intra_cluster_edges = []
        self.inter_cluster_edges = []
        
        # Initialize state
        self.current_activation = 0.0
        self.activation_history = []
        self.energy_reserve = 1.0  # Start with full energy
        
        # Initialize metrics
        self.metrics = ClusterMetrics()
        
        # Initialize specialized sub-containers
        self._initialize_subcontainers()
        
        logger.info(f"Created ClusterContainer '{name}' of type {cluster_type.value} "
                   f"with capacity {max_capacity}")
    
    def _initialize_subcontainers(self):
        """Initialize internal sub-containers for organization"""
        
        # Track counts by type
        self._node_count = 0
        self._edge_count = 0
        self._knowledge_count = 0
        self._subcluster_count = 0
    
    def _validate_item(self, item: BaseEntity) -> None:
        """
        Validate items added to the cluster.
        
        Override from BaseContainer to ensure only AGI-compatible entities
        are added and they fit within cluster's specialization.
        """
        
        # Check if item is AGI-compatible
        from .nodeentity import NodeEntity
        from .edgeentity import EdgeEntity
        from .knowledgechunk import KnowledgeChunk
        
        if isinstance(item, (NodeEntity, EdgeEntity, KnowledgeChunk, ClusterContainer)):
            # Additional validation based on cluster type
            if isinstance(item, NodeEntity):
                # Check if node specialization matches cluster
                node_spec = getattr(item, 'specialization_vector', None)
                if node_spec is not None and hasattr(node_spec, '__len__'):
                    similarity = self._calculate_specialization_similarity(node_spec)
                    if similarity < 0.3:
                        logger.warning(f"Node '{item.name}' specialization similarity "
                                     f"with cluster is low: {similarity:.3f}")
            
            # Update type counts
            if isinstance(item, NodeEntity):
                self._node_count += 1
            elif isinstance(item, EdgeEntity):
                self._edge_count += 1
            elif isinstance(item, KnowledgeChunk):
                self._knowledge_count += 1
            elif isinstance(item, ClusterContainer):
                self._subcluster_count += 1
                
        else:
            raise ValueError(f"ClusterContainer can only contain AGI entities "
                           f"(NodeEntity, EdgeEntity, KnowledgeChunk, ClusterContainer), "
                           f"got {type(item).__name__}")
    
    def _calculate_specialization_similarity(self, other_vector: np.ndarray) -> float:
        """
        Calculate similarity between cluster specialization and another vector.
        
        Args:
            other_vector: Vector to compare with cluster specialization
            
        Returns:
            Cosine similarity (0-1)
        """
        # Ensure vectors have same length
        if len(self.specialization_vector) != len(other_vector):
            min_len = min(len(self.specialization_vector), len(other_vector))
            vec1 = self.specialization_vector[:min_len]
            vec2 = other_vector[:min_len]
        else:
            vec1 = self.specialization_vector
            vec2 = other_vector
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return (similarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    def add_node(self, 
                node: NodeEntity,
                connect_to: Optional[List[str]] = None,
                initial_edges: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Add a NodeEntity to the cluster with optional connections.
        
        Args:
            node: NodeEntity to add
            connect_to: List of node names to connect to
            initial_edges: List of edge configurations
            
        Returns:
            Name of the added node
        """
        
        # Check capacity
        if self._node_count >= self.max_capacity:
            logger.warning(f"Cluster '{self.name}' at capacity ({self.max_capacity})")
            # TODO: Implement capacity management (prune/consolidate)
        
        # Add node to container
        self.add(node)
        
        # Create connections if specified
        if connect_to:
            for target_name in connect_to:
                if self.has_item(target_name) and isinstance(self.get(target_name), NodeEntity):
                    # Create edge between nodes
                    edge_name = f"edge_{node.name}_to_{target_name}"
                    edge = EdgeEntity(
                        name=edge_name,
                        source_node=node.name,
                        target_node=target_name,
                        weight=0.5,  # Initial weight
                        connection_type="excitatory"
                    )
                    self.add(edge)
                    self.intra_cluster_edges.append(edge_name)
        
        # Add custom edges if specified
        if initial_edges:
            for edge_config in initial_edges:
                edge = EdgeEntity(**edge_config)
                self.add(edge)
                self.intra_cluster_edges.append(edge.name)
        
        logger.debug(f"Added node '{node.name}' to cluster '{self.name}'")
        return node.name
    
    def connect_clusters(self,
                        target_cluster: 'ClusterContainer',
                        connection_type: str = "excitatory",
                        weight: float = 0.5) -> str:
        """
        Create connection between this cluster and another cluster.
        
        Args:
            target_cluster: Cluster to connect to
            connection_type: Type of connection
            weight: Initial connection strength
            
        Returns:
            Name of the created inter-cluster edge
        """
        
        edge_name = f"inter_cluster_{self.name}_to_{target_cluster.name}"
        
        # Create representative edge between clusters
        # In real implementation, this would create multiple edges between boundary nodes
        edge = EdgeEntity(
            name=edge_name,
            source_node=f"cluster_{self.name}_output",
            target_node=f"cluster_{target_cluster.name}_input",
            weight=weight,
            connection_type=connection_type,
            transmission_delay=20.0,  # Higher delay for inter-cluster
            bandwidth=0.8  # Limited bandwidth between clusters
        )
        
        self.add(edge)
        self.inter_cluster_edges.append(edge_name)
        
        # Update connectivity lists
        if target_cluster.name not in self.output_clusters:
            self.output_clusters.append(target_cluster.name)
        
        if self.name not in target_cluster.input_clusters:
            target_cluster.input_clusters.append(self.name)
        
        logger.info(f"Connected cluster '{self.name}' -> '{target_cluster.name}' "
                   f"with weight {weight:.3f}")
        
        return edge_name
    
    def propagate_activation(self,
                       input_signals: Dict[str, float],
                       max_depth: int = 3,
                       energy_budget: float = 1.0) -> Dict[str, Any]:
        """
        Propagate activation through the cluster.
        
        Args:
            input_signals: Dictionary of node_name -> activation_strength
            max_depth: Maximum propagation depth
            energy_budget: Available energy for processing
            
        Returns:
            Dictionary with propagation results
        """
        
        if not self.isactive:
            return {
                "status": "inactive",
                "cluster": self.name,
                "output_activation": 0.0,
                "activated_nodes": [],
                "energy_used": 0.0,
                "latency_seconds": 0.0,
                "coherence": 0.0
            }
        
        start_time = datetime.now()
        activated_nodes = []
        total_output_activation = 0.0
        energy_used = 0.0
        
        # Apply input signals to nodes
        for node_name, signal_strength in input_signals.items():
            if self.has_item(node_name):
                node = self.get(node_name)
                if isinstance(node, NodeEntity) and node.isactive:
                    # Получаем размерность входного вектора из ноды
                    # Предполагаем, что NodeEntity имеет атрибут input_dim
                    input_dim = getattr(node, 'input_dim', 512)
                    
                    # Создаем входной вектор правильной размерности
                    # Используем случайный вектор с seed на основе сигнала
                    np.random.seed(int(signal_strength * 1000))
                    input_vector = np.random.randn(input_dim).astype(np.float32) * signal_strength
                    
                    # Activate node
                    try:
                        result = node.forward(
                            input_vector=input_vector,  # Исправлено: передаем вектор
                            use_memory=True
                        )
                        
                        # Check if node activated successfully
                        if result is not None and (result.get("status") is None or result.get("status") == "success"):
                            # Node activated successfully
                            activated_nodes.append(node_name)
                            total_output_activation += result.get("confidence", 0)
                            energy_used += 0.01  # Base energy per activation
                        elif result is not None and result.get("status") == "inactive":
                            logger.debug(f"Node '{node_name}' is inactive")
                        elif result is not None and result.get("status") == "low_confidence":
                            logger.debug(f"Node '{node_name}' has low confidence: {result.get('confidence', 0):.3f}")
                        else:
                            logger.debug(f"Node '{node_name}' forward returned None")
                            
                    except Exception as e:
                        logger.error(f"Error activating node '{node_name}': {e}")
                        continue
        
        # Если активированы ноды, продолжаем распространение
        if len(activated_nodes) == 0:
            latency = (datetime.now() - start_time).total_seconds()
            return {
                "status": "no_activation",
                "cluster": self.name,
                "output_activation": 0.0,
                "activated_nodes": [],
                "energy_used": energy_used,
                "latency_seconds": latency,
                "coherence": 0.0
            }
        
        # Propagate through edges (simplified)
        # В реальной реализации это использовало бы RoutingSuper
        propagation_steps = min(max_depth, len(activated_nodes))
        
        for step in range(propagation_steps):
            step_activation = 0.0
            
            for node_name in activated_nodes:
                node = self.get(node_name)
                if isinstance(node, NodeEntity):
                    # Находим исходящие ребра
                    outgoing_edges = [
                        e for e in self.intra_cluster_edges 
                        if self.has_item(e) and isinstance(self.get(e), EdgeEntity)
                        and getattr(self.get(e), 'source_node', None) == node_name
                    ]
                    
                    for edge_name in outgoing_edges:
                        edge = self.get(edge_name)
                        if isinstance(edge, EdgeEntity) and edge.isactive:
                            try:
                                # Передаем сигнал
                                transmission = edge.transmit(signal_strength=0.5)
                                if transmission is not None and transmission.get("transmitted", False):
                                    step_activation += abs(transmission.get("output_strength", 0))
                                    energy_used += 0.005  # Energy per transmission
                            except Exception as e:
                                logger.error(f"Error in edge transmission '{edge_name}': {e}")
            
            total_output_activation += step_activation
        
        # Обновляем состояние кластера
        self.current_activation = min(1.0, total_output_activation)
        self.activation_history.append(self.current_activation)
        
        # Обновляем энергию
        self.energy_reserve = max(0.0, self.energy_reserve - energy_used)
        
        # Рассчитываем задержку
        latency = (datetime.now() - start_time).total_seconds()
        
        # Создаем паттерн активации
        activation_pattern = ActivationPattern(
            timestamp=start_time,
            active_nodes=activated_nodes,
            activation_levels={n: 0.5 for n in activated_nodes},  # Упрощено
            coherence_score=self._calculate_coherence(activated_nodes),
            information_content=total_output_activation
        )
        
        # Обновляем метрики
        success = len(activated_nodes) > 0
        self.metrics.update(
            success=success,
            latency=latency,
            energy=energy_used,
            activation_pattern=activation_pattern
        )
        
        logger.debug(f"Cluster '{self.name}' propagated activation: "
                    f"{len(activated_nodes)} nodes, "
                    f"output={total_output_activation:.3f}, "
                    f"energy_used={energy_used:.3f}")
        
        # ВСЕГДА возвращаем словарь
        return {
            "status": "success" if success else "partial_success",
            "cluster": self.name,
            "output_activation": total_output_activation,
            "activated_nodes": activated_nodes,
            "energy_used": energy_used,
            "latency_seconds": latency,
            "coherence": activation_pattern.coherence_score
        }
    
    def _calculate_coherence(self, activated_nodes: List[str]) -> float:
        """
        Calculate coherence of activation pattern.
        
        Args:
            activated_nodes: List of activated node names
            
        Returns:
            Coherence score (0-1)
        """
        if len(activated_nodes) <= 1:
            return 1.0 if activated_nodes else 0.0
        
        # Simplified coherence calculation
        # In real implementation, this would analyze temporal synchronization
        node_vectors = []
        
        for node_name in activated_nodes:
            node = self.get(node_name)
            if isinstance(node, NodeEntity):
                # Use node's specialization or activation state
                node_vector = getattr(node, 'specialization_vector', 
                                    np.random.randn(10))[:10]  # Truncate for speed
                node_vectors.append(node_vector)
        
        if not node_vectors:
            return 0.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(node_vectors)):
            for j in range(i + 1, len(node_vectors)):
                vec1 = node_vectors[i]
                vec2 = node_vectors[j]
                
                # Ensure same length
                min_len = min(len(vec1), len(vec2))
                if min_len > 0:
                    norm1 = np.linalg.norm(vec1[:min_len])
                    norm2 = np.linalg.norm(vec2[:min_len])
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(vec1[:min_len], vec2[:min_len]) / (norm1 * norm2)
                        similarities.append((similarity + 1) / 2)  # Convert to [0, 1]
        
        return np.mean(similarities) if similarities else 0.0
    
    def learn_from_experience(self,
                        experience_data: Dict[str, Any],
                        learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Learn from experience by updating nodes and edges.
        
        Args:
            experience_data: Dictionary with learning data
            learning_rate: Overall learning rate
            
        Returns:
            Dictionary with learning results
        """
        
        if not self.isactive:
            return {"status": "inactive", "cluster": self.name}
        
        learning_results = {
            "status": "success",
            "cluster": self.name,
            "nodes_updated": 0,
            "edges_updated": 0,
            "total_reward": 0.0
        }
        
        # Get active nodes
        active_nodes = self.get_active_items()
        
        for item in active_nodes:
            if isinstance(item, NodeEntity):
                # Simplified learning - in real implementation would use LearningSuper
                try:
                    # Get node's input dimension
                    input_dim = getattr(item, 'input_dim', 512)
                    
                    # Check if experience data has training data for this node
                    if "training_pairs" in experience_data:
                        # Find relevant training data
                        for pair in experience_data["training_pairs"]:
                            # Create proper input vector
                            input_vector = np.random.randn(input_dim).astype(np.float32) * 0.5
                            
                            # Create target output vector (same dimension)
                            target_output = np.random.randn(input_dim).astype(np.float32) * 0.5
                            
                            # Call learning method with proper vectors
                            result = item.learn_from_experience(
                                input_vector=input_vector,
                                target_output=target_output,
                                reward=0.5  # Placeholder
                            )
                            
                            if result is not None and result.get("status") != "inactive":
                                learning_results["nodes_updated"] += 1
                                learning_results["total_reward"] += result.get("reward", 0)
                    else:
                        # No training pairs, use reinforcement learning
                        # Create random input vector
                        input_vector = np.random.randn(input_dim).astype(np.float32) * 0.5
                        
                        # Use the node's own output as target (self-supervised)
                        output_result = item.forward(input_vector=input_vector, use_memory=False)
                        if output_result is not None:
                            # Create target based on node's own output
                            output_value = output_result.get("confidence", 0.5)
                            target_output = np.full((input_dim,), output_value, dtype=np.float32)
                            
                            result = item.learn_from_experience(
                                input_vector=input_vector,
                                target_output=target_output,
                                reward=0.3  # Small positive reward
                            )
                            
                            if result is not None and result.get("status") != "inactive":
                                learning_results["nodes_updated"] += 1
                                learning_results["total_reward"] += result.get("reward", 0)
                                
                except Exception as e:
                    logger.error(f"Error learning in node '{item.name}': {e}")
                    continue
            
            elif isinstance(item, EdgeEntity):
                # Update edge weights based on usage
                try:
                    if getattr(item, 'usage_frequency', 0) > 0:
                        # Reinforcement based on recent usage
                        reinforcement = min(1.0, item.usage_frequency / 100)
                        item.reinforce(reinforcement_strength=reinforcement)
                        learning_results["edges_updated"] += 1
                except Exception as e:
                    logger.error(f"Error reinforcing edge '{item.name}': {e}")
                    continue
        
        # Update cluster metrics
        self.metrics.knowledge_growth += learning_results["nodes_updated"]
        
        logger.info(f"Cluster '{self.name}' learned: "
                f"{learning_results['nodes_updated']} nodes, "
                f"{learning_results['edges_updated']} edges updated")
        
        return learning_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster statistics.
        
        Returns:
            Dictionary with all statistics
        """
        
        metrics_summary = self.metrics.get_summary()
        
        # Calculate specialization distribution
        specialization_scores = []
        for item in self.get_items():
            if isinstance(item, NodeEntity):
                node_spec = getattr(item, 'specialization_vector', None)
                if node_spec is not None:
                    similarity = self._calculate_specialization_similarity(node_spec)
                    specialization_scores.append(similarity)
        
        avg_specialization = np.mean(specialization_scores) if specialization_scores else 0.0
        
        return {
            "name": self.name,
            "cluster_type": self.cluster_type.value,
            "isactive": self.isactive,
            "current_activation": self.current_activation,
            "energy_reserve": self.energy_reserve,
            "capacity": {
                "max": self.max_capacity,
                "used": self._node_count,
                "available": self.max_capacity - self._node_count
            },
            "composition": {
                "nodes": self._node_count,
                "edges": self._edge_count,
                "knowledge_chunks": self._knowledge_count,
                "subclusters": self._subcluster_count
            },
            "connectivity": {
                "input_clusters": len(self.input_clusters),
                "output_clusters": len(self.output_clusters),
                "intra_cluster_edges": len(self.intra_cluster_edges),
                "inter_cluster_edges": len(self.inter_cluster_edges)
            },
            "metrics": metrics_summary,
            "specialization": {
                "average_similarity": avg_specialization,
                "vector_dim": len(self.specialization_vector)
            }
        }
    
    def recharge_energy(self, amount: float = 0.1) -> float:
        """
        Recharge cluster's energy reserve.
        
        Args:
            amount: Amount of energy to add
            
        Returns:
            New energy level
        """
        self.energy_reserve = min(1.0, self.energy_reserve + amount)
        
        # Also recharge active nodes
        for item in self.get_active_items():
            if isinstance(item, NodeEntity):
                item.recharge_energy(amount * 0.1)  # Nodes get less energy
        
        logger.debug(f"Cluster '{self.name}' energy recharged to {self.energy_reserve:.3f}")
        return self.energy_reserve
    
    def activate_all_nodes(self) -> int:
        """
        Activate all nodes in the cluster.
        
        Returns:
            Number of activated nodes
        """
        activated = 0
        for item in self.get_items():
            if isinstance(item, NodeEntity) and not item.isactive:
                item.activate()
                activated += 1
        
        logger.info(f"Activated {activated} nodes in cluster '{self.name}'")
        return activated
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert cluster to dictionary for serialization.
        
        Override to handle cluster-specific attributes.
        """
        data = super().to_dict()
        
        # Add cluster-specific data
        data.update({
            "cluster_type": self.cluster_type.value,
            "specialization_vector": self.specialization_vector.tolist(),
            "activation_threshold": self.activation_threshold,
            "max_capacity": self.max_capacity,
            "current_activation": self.current_activation,
            "activation_history": self.activation_history,
            "energy_reserve": self.energy_reserve,
            "input_clusters": self.input_clusters,
            "output_clusters": self.output_clusters,
            "intra_cluster_edges": self.intra_cluster_edges,
            "inter_cluster_edges": self.inter_cluster_edges,
            "metrics": self.metrics.get_summary(),
            "composition": {
                "nodes": self._node_count,
                "edges": self._edge_count,
                "knowledge_chunks": self._knowledge_count,
                "subclusters": self._subcluster_count
            }
        })
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterContainer':
        """
        Create cluster from dictionary (deserialization).
        
        Args:
            data: Dictionary with cluster data
            
        Returns:
            ClusterContainer instance
        """
        
        # Extract cluster-specific data
        cluster_type_str = data.get("cluster_type", "integrative")
        try:
            cluster_type = ClusterType(cluster_type_str)
        except ValueError:
            logger.warning(f"Unknown cluster type '{cluster_type_str}', defaulting to INTEGRATIVE")
            cluster_type = ClusterType.INTEGRATIVE
            
        specialization_vector = np.array(data.get("specialization_vector", []))
        activation_threshold = data.get("activation_threshold", 0.3)
        max_capacity = data.get("max_capacity", 100)
        
        input_clusters = data.get("input_clusters", [])
        output_clusters = data.get("output_clusters", [])
        intra_cluster_edges = data.get("intra_cluster_edges", [])
        inter_cluster_edges = data.get("inter_cluster_edges", [])
        
        current_activation = data.get("current_activation", 0.0)
        activation_history = data.get("activation_history", [])
        energy_reserve = data.get("energy_reserve", 1.0)
        
        # Create cluster instance
        cluster = cls(
            name=data["name"],
            cluster_type=cluster_type,
            specialization_vector=specialization_vector,
            activation_threshold=activation_threshold,
            max_capacity=max_capacity,
            input_clusters=input_clusters,
            output_clusters=output_clusters,
            isactive=data.get("isactive", True),
            use_cache=data.get("_use_cache", False)
        )
        
        # Set additional state
        cluster.current_activation = current_activation
        cluster.activation_history = activation_history
        cluster.energy_reserve = energy_reserve
        cluster.intra_cluster_edges = intra_cluster_edges
        cluster.inter_cluster_edges = inter_cluster_edges
        
        # Restore items from the serialized data
        items_dict = data.get("items", {})
        
        # Clear existing items and add restored ones
        cluster.clear()
        
        # Import entity classes for deserialization
        from .nodeentity import NodeEntity
        from .edgeentity import EdgeEntity
        from .knowledgechunk import KnowledgeChunk
        
        for item_name, item_data in items_dict.items():
            item_type = item_data.get("type", "")
            
            if "NodeEntity" in item_type:
                item = NodeEntity.from_dict(item_data)
            elif "EdgeEntity" in item_type:
                item = EdgeEntity.from_dict(item_data)
            elif "KnowledgeChunk" in item_type:
                item = KnowledgeChunk.from_dict(item_data)
            elif "ClusterContainer" in item_type:
                item = cls.from_dict(item_data)
            else:
                logger.warning(f"Unknown item type '{item_type}' for item '{item_name}'")
                continue
            
            cluster.add(item, copy_items=False)
        
        # Restore metrics
        metrics_data = data.get("metrics", {})
        if metrics_data:
            cluster.metrics.activation_count = metrics_data.get("activation_count", 0)
            cluster.metrics.knowledge_growth = metrics_data.get("knowledge_growth", 0)
            cluster.metrics.energy_consumption = metrics_data.get("energy_consumption", 0.0)
        
        logger.info(f"Restored ClusterContainer '{cluster.name}' from dict")
        return cluster
    
    def __repr__(self) -> str:
        """String representation of ClusterContainer"""
        stats = self.get_stats()
        comp = stats["composition"]
        
        return (
            f"ClusterContainer(name='{self.name}', "
            f"type={self.cluster_type.value}, "
            f"nodes={comp['nodes']}/{self.max_capacity}, "
            f"activation={self.current_activation:.3f}, "
            f"energy={self.energy_reserve:.3f})"
        )