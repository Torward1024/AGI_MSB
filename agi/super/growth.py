# agi/super/growth.py
from common.super.super import Super
from typing import Dict, Any
from agi.node import NodeEntity
<<<<<<< HEAD
=======
from agi.cluster import ClusterContainer
>>>>>>> 2036bf4596f2ed47ea30843494bba4825968da31
from common.utils.logging_setup import logger
import torch

class GrowthSuper(Super):
    """Super class handling dynamic growth operations in the fractal AGI graph.

    Registered as 'growth' operation in AGIManipulator.
    Supports creating new nodes or clusters when confidence gaps are detected.
    """
    OPERATION = "growth"

    def __init__(self, manipulator):
        super().__init__(manipulator=manipulator)

    def _growth_nodeentity(self, node: NodeEntity, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for node-specific growth – not used directly."""
        return self._default_result(node)

    def _growth_clustercontainer(self, cluster, attributes: Dict[str, Any]) -> Dict[str, Any]:
<<<<<<< HEAD
        """Create a new NodeEntity inside a ClusterContainer."""
        from agi.cluster import ClusterContainer
        
=======
        """Create a new NodeEntity inside a ClusterContainer."""       
>>>>>>> 2036bf4596f2ed47ea30843494bba4825968da31
        name = attributes.get("name")
        if not name:
            raise ValueError("Growth requires 'name' for new node")
        
        nn_config = attributes.get("nn_config", {"layers": [128, 64, 128], "activation": "relu"})
        memory_dim = attributes.get("memory_dim", 128)
        level = attributes.get("level", cluster.subgraph.ndata.get('level', torch.tensor([0])).max().item() + 1)
        
        new_node = NodeEntity(name=name, nn_config=nn_config, memory_dim=memory_dim, level=level)
        cluster.add(new_node)
        
        logger.info(f"GrowthSuper created new node '{name}' in cluster '{cluster.name}' at level {level}")
        return self._build_response(cluster, True, "_growth_clustercontainer", {"new_node": name})

    def _growth_graphproject(self, project, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new ClusterContainer inside the GraphProject."""
        cluster_name = attributes.get("cluster_name")
        if not cluster_name:
            raise ValueError("Growth at project level requires 'cluster_name'")
        
        project.create_item(cluster_name=cluster_name)
        logger.info(f"GrowthSuper created new cluster '{cluster_name}' in project '{project.name}'")
        return self._build_response(project, True, "_growth_graphproject", {"new_cluster": cluster_name})

    def _growth(self, obj: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Default growth handler – delegates to type-specific methods."""
        return self._default_result(obj)