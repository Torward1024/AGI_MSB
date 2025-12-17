# agi/cluster.py
from typing import Dict, Any, Optional
from common.base.basecontainer import BaseContainer
from agi.node import NodeEntity
from common.utils.logging_setup import logger
from agi.graph import SimpleGraph
import torch

class ClusterContainer(BaseContainer[NodeEntity]):
    """Manages a cluster of NodeEntities as a sub-graph in the fractal AGI.

    Extends BaseContainer to include a SimpleGraph for graph operations,
    supporting addition of nodes/edges, propagation, and recursive sub-clusters.

    Attributes:
        subgraph (SimpleGraph): The custom graph representing node connections.
    """
    subgraph: SimpleGraph

    def __init__(self, *, name: str = None, items: Optional[Dict[str, NodeEntity]] = None,
                 isactive: bool = True, use_cache: bool = False):
        """Initialize the ClusterContainer with an empty SimpleGraph.

        Args:
            name (str, optional): Cluster identifier.
            items (Dict[str, NodeEntity], optional): Initial nodes.
            isactive (bool): Activation status.
            use_cache (bool): Enable caching for to_dict.
        """
        super().__init__(name=name, items=items or {}, isactive=isactive, use_cache=use_cache)
        self.subgraph = SimpleGraph()
        if items:
            for name, node in items.items():
                self.add(node)
        logger.debug(f"Initialized ClusterContainer '{name}'")

    def add(self, item: NodeEntity) -> None:
        """Add a NodeEntity and update the graph.

        Args:
            item (NodeEntity): Node to add.

        Raises:
            ValueError: If item name already exists.
        """
        super().add(item)
        self.subgraph.add_node(item.name, item.level)
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
        self.subgraph.add_edges(from_name, to_name, {'type': edge_type, 'weight': weight})
        logger.debug(f"Added edge from '{from_name}' to '{to_name}' in cluster '{self.name}'")

    def propagate(self, input_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Propagate input through the sub-graph using BFS message passing.

        Args:
            input_vector (torch.Tensor): Input embedding.

        Returns:
            Dict[str, torch.Tensor]: Outputs from each node.
        """
        def process_fn(node_name: str, aggregated: torch.Tensor) -> torch.Tensor:
            node = self._items[node_name]
            return node.process_input(aggregated)

        # Start from all root nodes (no incoming edges)
        indegrees = {n: 0 for n in self.subgraph.nodes}
        for edges in self.subgraph.edges.values():
            for e in edges:
                indegrees[e['dst']] += 1
        start_nodes = [n for n, deg in indegrees.items() if deg == 0]

        outputs = self.subgraph.propagate(start_nodes, input_vector, process_fn)
        logger.debug(f"Propagated input through cluster '{self.name}'")
        return outputs

    def clear(self) -> None:
        """Clear the sub-graph and nodes."""
        self.subgraph.clear()
        super().clear()
        logger.debug(f"Cleared ClusterContainer '{self.name}'")

    def __repr__(self) -> str:
        """Return a string representation of the ClusterContainer."""
        return f"ClusterContainer(name={self.name!r}, node_count={len(self._items)}, isactive={self.isactive})"