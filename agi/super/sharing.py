# agi/super/sharing.py
from common.super.super import Super
from typing import Dict, Any
from common.utils.logging_setup import logger

class SharingSuper(Super):
    """Super class handling knowledge diffusion between nodes."""
    OPERATION = "sharing"

    def __init__(self, manipulator):
        super().__init__(manipulator=manipulator)

    def _sharing_nodeentity(self, node, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Share knowledge from one node to another."""
        target_name = attributes.get("target")
        subset_ratio = attributes.get("subset_ratio", 0.5)
        
        cluster = self._manipulator.get_managing_object().get_item(attributes.get("cluster"))
        if not cluster or target_name not in cluster:
            raise ValueError(f"Target node '{target_name}' not found")
        
        target_node = cluster[target_name]
        node.share_knowledge(target_node, subset_ratio)
        
        logger.info(f"SharingSuper transferred knowledge from '{node.name}' to '{target_name}'")
        return self._build_response(node, True, "_sharing_nodeentity", {"target": target_name, "ratio": subset_ratio})

    def _sharing(self, obj: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return self._default_result(obj)