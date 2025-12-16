# agi/cluster.py
from typing import Dict, Any, Optional
from common.base.basecontainer import BaseContainer
from agi.node import NodeEntity
from common.utils.logging_setup import logger
from dgl import DGLGraph
import dgl
import torch

class ClusterContainer(BaseContainer[NodeEntity]):
    """Manages a cluster of NodeEntities as a sub-graph in the fractal AGI.

    Extends BaseContainer to include a DGLGraph for efficient graph operations,
    supporting addition of nodes/edges, propagation, and recursive sub-clusters.

    Attributes:
        subgraph (DGLGraph): The DGL graph representing node connections.

    Notes:
        - Supports fractal recursion: Can contain sub-Clusters as nodes.
        - Uses DGL for message passing and parallelism.
    """
    subgraph: DGLGraph

    def __init__(self, *, name: str = None, items: Optional[Dict[str, NodeEntity]] = None,
                 isactive: bool = True, use_cache: bool = False):
        """Initialize the ClusterContainer with an empty DGL graph.

        Args:
            name (str, optional): Cluster identifier.
            items (Dict[str, NodeEntity], optional): Initial nodes.
            isactive (bool): Activation status.
            use_cache (bool): Enable caching for to_dict.
        """
        super().__init__(name=name, items=items, isactive=isactive, use_cache=use_cache)
        self.subgraph = DGLGraph()
        logger.debug(f"Initialized ClusterContainer '{name}'")

    def add(self, item: NodeEntity) -> None:
        """Add a NodeEntity and update the DGL graph.

        Args:
            item (NodeEntity): Node to add.

        Raises:
            ValueError: If item name already exists.
        """
        super().add(item)
        node_id = len(self.subgraph.nodes())
        self.subgraph.add_nodes(1, {'name': torch.tensor([item.name]), 'level': torch.tensor([item.level])})
        logger.debug(f"Added node '{item.name}' to cluster '{self.name}'")

    def add_edge(self, from_name: str, to_name: str, edge_type: str = 'neutral', weight: float = 1.0) -> None:
        """Add an edge between two nodes in the cluster.

        Args:
            from_name (str): Source node name.
            to_name (str): Target node name.
            edge_type (str): Edge type (excitatory, inhibitory, neutral).
            weight (float): Edge weight.

        Raises:
            ValueError: If nodes do not exist.
        """
        if from_name not in self._items or to_name not in self._items:
            raise ValueError("Both nodes must exist in the cluster")
        # Assuming node IDs are sequential
        from_id = list(self._items.keys()).index(from_name)
        to_id = list(self._items.keys()).index(to_name)
        self.subgraph.add_edges(from_id, to_id, {'type': torch.tensor([edge_type]), 'weight': torch.tensor([weight])})
        logger.debug(f"Added edge from '{from_name}' to '{to_name}' in cluster '{self.name}'")

    def propagate(self, input_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Propagate input through the sub-graph using message passing.

        Args:
            input_vector (torch.Tensor): Input embedding.

        Returns:
            Dict[str, torch.Tensor]: Outputs from each node.

        Notes:
            - Uses DGL message passing; placeholder for full implementation.
        """
        # Placeholder: Simulate propagation
        outputs = {}
        for name, node in self._items.items():
            outputs[name] = node.process_input(input_vector)
        logger.debug(f"Propagated input through cluster '{self.name}'")
        return outputs

    def clear(self) -> None:
        """Clear the sub-graph and nodes."""
        self.subgraph = None
        super().clear()
        logger.debug(f"Cleared ClusterContainer '{self.name}'")

    def __repr__(self) -> str:
        """Return a string representation of the ClusterContainer."""
        return f"ClusterContainer(name={self.name!r}, node_count={len(self._items)}, isactive={self.isactive})"