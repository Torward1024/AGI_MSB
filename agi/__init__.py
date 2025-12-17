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

from agi.knowledgechunk import (
    KnowledgeChunk,
    CompressionMethod,
    KnowledgeStatus,
    SemanticMetadata
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
    'LearningRule',
    
    # Knowledge components
    'KnowledgeChunk',
    'CompressionMethod',
    'KnowledgeStatus',
    'SemanticMetadata'
]