# agi/super/status.py
from common.super.super import Super
from typing import Dict, Any
from agi.project import GraphProject
from agi.cluster import ClusterContainer
from agi.node import NodeEntity
from common.utils.logging_setup import logger

class StatusSuper(Super):
    """Super class for retrieving AGI system status.

    Provides detailed structure, node counts, network sizes, etc.
    """
    OPERATION = "status"

    def __init__(self, manipulator):
        super().__init__(manipulator=manipulator)

    def _status_graphproject(self, project: GraphProject, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Get full AGI status."""
        status = {
            "project_name": project.name,
            "cluster_count": len(project._items),
            "total_nodes": 0,
            "total_params": 0,
            "structure": {}
        }
        for cluster_name, cluster in project._items.items():
            cluster_status = self._get_cluster_status(cluster)
            status["structure"][cluster_name] = cluster_status
            status["total_nodes"] += cluster_status["node_count"]
            status["total_params"] += cluster_status["total_params"]
        logger.info(f"StatusSuper retrieved AGI status: {status}")
        return self._build_response(project, True, "_status_graphproject", status)

    def _get_cluster_status(self, cluster: ClusterContainer) -> Dict[str, Any]:
        status = {
            "name": cluster.name,
            "node_count": len(cluster._items),
            "total_params": 0,
            "nodes": {}
        }
        for node_name, node in cluster._items.items():
            node_params = sum(p.numel() for p in node.nn_model.parameters())
            status["nodes"][node_name] = {
                "level": node.level,
                "memory_size": len(node.memory.vectors),
                "params": node_params
            }
            status["total_params"] += node_params
        return status

    def _status(self, obj: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return self._default_result(obj)