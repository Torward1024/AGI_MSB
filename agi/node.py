# agi/node.py
from typing import Dict, Any, Optional
from common.base.baseentity import BaseEntity
from common.utils.logging_setup import logger
import torch.nn as nn
import faiss
import torch

class NodeEntity(BaseEntity):
    """Represents a mini-neural network column in the fractal AGI graph.

    Extends BaseEntity to manage a node with a configurable neural network, vector memory,
    confidence assessor, and hierarchy level. Supports processing, learning, confidence assessment,
    and knowledge sharing.

    Attributes:
        nn_model (nn.Module): The mini-neural network for input processing.
        memory (faiss.Index): Vector database for storing patterns and knowledge.
        confidence_nn (nn.Module): Small NN to assess confidence in handling inputs.
        level (int): Hierarchical level (0 for micro, higher for meso/macro).

    Notes:
        - nn_model is built dynamically from a config dictionary.
        - Memory uses FAISS for efficient similarity search.
        - Confidence NN is a simple MLP outputting a score between 0 and 1.
    """
    nn_model: nn.Module
    memory: faiss.Index
    confidence_nn: nn.Module
    level: int

    def __init__(self, *, name: str = None, nn_config: Dict[str, Any], memory_dim: int = 128,
                 level: int = 0, isactive: bool = True):
        """Initialize the NodeEntity with neural components and level.

        Args:
            name (str, optional): Unique identifier for the node.
            nn_config (Dict[str, Any]): Configuration for building the mini-NN (e.g., {'layers': [128, 64], 'activation': 'relu'}).
            memory_dim (int): Dimensionality for the FAISS vector index.
            level (int): Hierarchical level in the fractal graph.
            isactive (bool): Activation status.

        Raises:
            ValueError: If nn_config is invalid or memory_dim <= 0.
        """
        super().__init__(name=name, isactive=isactive)
        if memory_dim <= 0:
            raise ValueError("Memory dimension must be positive")
        self.nn_model = self._build_nn(nn_config)
        self.memory = faiss.IndexFlatL2(memory_dim)
        self.confidence_nn = nn.Sequential(
            nn.Linear(memory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.level = level
        logger.debug(f"Initialized NodeEntity '{name}' at level {level}")

    def _build_nn(self, config: Dict[str, Any]) -> nn.Module:
        """Dynamically build the mini-neural network from config.

        Args:
            config (Dict[str, Any]): NN configuration (layers, activation, etc.).

        Returns:
            nn.Module: Constructed neural network.

        Raises:
            ValueError: If config is missing required keys or invalid.
        """
        if 'layers' not in config or not isinstance(config['layers'], list):
            raise ValueError("nn_config must include 'layers' as a list of integers")
        activation = config.get('activation', 'relu')
        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError(f"Invalid activation: {activation}")
        
        layers = []
        for i in range(len(config['layers']) - 1):
            layers.append(nn.Linear(config['layers'][i], config['layers'][i+1]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
        
        model = nn.Sequential(*layers)
        logger.debug(f"Built NN model with layers: {config['layers']}")
        return model

    def process_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Process an input tensor through the mini-NN, refining with memory query.

        Args:
            input_tensor (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Processed output.
        """
        with torch.no_grad():
            output = self.nn_model(input_tensor)
        # Query memory for similar patterns (placeholder)
        return output

    def learn(self, input_tensor: torch.Tensor, target: torch.Tensor, reward: float = 0.0) -> float:
        """Perform self-learning via backpropagation with optional RL reward adjustment.

        Args:
            input_tensor (torch.Tensor): Input data.
            target (torch.Tensor): Target output.
            reward (float): RL reward to adjust loss.

        Returns:
            float: Computed loss after update.
        """
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        output = self.process_input(input_tensor)
        loss = loss_fn(output, target) - reward
        loss.backward()
        optimizer.step()
        logger.debug(f"Node '{self.name}' learned with loss {loss.item()}")
        return loss.item()

    def assess_confidence(self, input_tensor: torch.Tensor) -> float:
        """Assess confidence in handling the input.

        Args:
            input_tensor (torch.Tensor): Input data.

        Returns:
            float: Confidence score (0-1).
        """
        with torch.no_grad():
            score = self.confidence_nn(input_tensor).item()
        logger.debug(f"Confidence score for input: {score}")
        return score

    def share_knowledge(self, other_node: 'NodeEntity', subset_ratio: float = 0.5) -> None:
        """Share a subset of knowledge (memory embeddings) with another node.

        Args:
            other_node (NodeEntity): Target node to share with.
            subset_ratio (float): Fraction of memory to share (0-1).

        Raises:
            ValueError: If subset_ratio is out of range.
        """
        if not 0 < subset_ratio <= 1:
            raise ValueError("Subset ratio must be between 0 and 1")
        # Placeholder: Compress and transfer embeddings
        logger.info(f"Shared {subset_ratio*100}% knowledge from '{self.name}' to '{other_node.name}'")

    def clear(self) -> None:
        """Clear neural models and memory to release resources."""
        self.nn_model = None
        self.confidence_nn = None
        self.memory = None
        super().clear()
        logger.debug(f"Cleared NodeEntity '{self.name}'")

    def __repr__(self) -> str:
        """Return a string representation of the NodeEntity."""
        return f"NodeEntity(name={self.name!r}, level={self.level}, isactive={self.isactive})"