# agi/graph.py
from typing import Dict, Any, List, Optional
from common.utils.logging_setup import logger
import torch
from collections import defaultdict

class SimpleGraph:
    """Custom lightweight graph implementation for AGI clusters.

    Uses adjacency lists for edges, dict for node data. Supports add_node, add_edge,
    propagation via BFS traversal (simulating message passing), and basic queries.
    Designed for CPU-based operations; parallelism via Ray can be added externally.
    """
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}  # node_name -> {'data': ..., 'features': torch.Tensor, ...}
        self.edges: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)  # src -> [{'dst': dst, 'weight': w, 'type': t}, ...]
        self.num_nodes: int = 0
        logger.debug("SimpleGraph initialized")

    def add_nodes(self, num: int, data: Optional[Dict[str, Any]] = None) -> None:
        """Add nodes with optional data (dummy for compatibility)."""
        self.num_nodes += num
        # In practice, nodes are added via add_node with name

    def add_node(self, name: str, level: int, features: Optional[torch.Tensor] = None) -> None:
        """Add a named node with level and optional features."""
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = {'level': level, 'features': features or torch.zeros(128)}
        self.num_nodes += 1
        logger.debug(f"Added node '{name}' at level {level}")

    def add_edges(self, src: str, dst: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add an edge with optional data (type, weight)."""
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f"Nodes {src} or {dst} not found")
        edge_data = data or {'type': 'neutral', 'weight': 1.0}
        self.edges[src].append({'dst': dst, **edge_data})
        logger.debug(f"Added edge {src} -> {dst}")

    def num_nodes(self) -> int:
        return self.num_nodes

    def propagate(self, start_nodes: List[str], input_vector: torch.Tensor, node_process_fn) -> Dict[str, torch.Tensor]:
        """Simulate message passing via BFS traversal.

        Args:
            start_nodes: Nodes to start propagation from.
            input_vector: Initial input.
            node_process_fn: Function(node_name, incoming_msgs) -> output.

        Returns:
            Outputs from all visited nodes.
        """
        from collections import deque
        visited = set()
        queue = deque(start_nodes)
        outputs = {}
        incoming = defaultdict(list)  # node -> list of (src_output, weight)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Collect incoming messages
            msgs = incoming[current]
            aggregated = sum(msg * w for msg, w in msgs) / len(msgs) if msgs else input_vector

            # Process
            outputs[current] = node_process_fn(current, aggregated)

            # Propagate to neighbors
            for edge in self.edges[current]:
                dst = edge['dst']
                w = edge['weight']
                incoming[dst].append((outputs[current], w))
                queue.append(dst)

        logger.debug(f"Propagated through {len(visited)} nodes")
        return outputs

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.num_nodes = 0
        logger.debug("SimpleGraph cleared")