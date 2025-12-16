# agi/project.py
from typing import Dict, Any
from common.super.project import Project
from agi.cluster import ClusterContainer
from common.utils.logging_setup import logger

class GraphProject(Project):
    """Top-level project manager for the fractal AGI graph.

    Extends Project to manage ClusterContainers as items, providing methods
    for creating clusters, global operations like growth and sharing.

    Attributes:
        _item_type: Set to ClusterContainer.
    """
    _item_type = ClusterContainer

    def __init__(self, name: str = "AGI_Project", items: Optional[Dict[str, ClusterContainer]] = None):
        """Initialize the GraphProject.

        Args:
            name (str): Project name.
            items (Dict[str, ClusterContainer], optional): Initial clusters.
        """
        super().__init__(name=name, items=items)
        logger.debug(f"Initialized GraphProject '{name}'")

    def create_item(self, cluster_name: str, isactive: bool = True) -> None:
        """Create and add a new ClusterContainer.

        Args:
            cluster_name (str): Name for the new cluster.
            isactive (bool): Activation status.
        """
        cluster = ClusterContainer(name=cluster_name, isactive=isactive)
        self.add_item(cluster)
        logger.info(f"Created cluster '{cluster_name}' in project '{self.name}'")

    # Additional methods for global AGI operations can be added here, e.g., global_growth, global_sharing

    def clear(self) -> None:
        """Clear all clusters and resources."""
        super().clear()
        logger.debug(f"Cleared GraphProject '{self.name}'")

    def __repr__(self) -> str:
        """Return a string representation of the GraphProject."""
        return f"GraphProject(name={self.name!r}, cluster_count={len(self._items)})"