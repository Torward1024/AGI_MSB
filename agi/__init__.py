"""
AGI (Artificial General Intelligence) Module
Contains core components for the fractal graph-based AGI architecture.
"""

from agi.nodeentity import (
    NodeEntity,
    NodeMemory,
    MemoryPattern,
    ConfidenceAssessor,
    MiniNeuralNetwork
)

from agi.edgeentity import (
    EdgeEntity,
    ConnectionType,
    LearningRule
)

__all__ = [
    # Node components
    'NodeEntity',
    'NodeMemory',
    'MemoryPattern',
    'ConfidenceAssessor',
    'MiniNeuralNetwork',
    
    # Edge components
    'EdgeEntity',
    'ConnectionType',
    'LearningRule'
]