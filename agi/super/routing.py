# agi/super/routing.py
from common.super.super import Super
from typing import Dict, Any, List
from common.utils.logging_setup import logger
import torch

class RoutingSuper(Super):
    """Super class handling query routing decisions using the Orchestrator NN."""
    OPERATION = "routing"

    def __init__(self, manipulator):
        super().__init__(manipulator=manipulator)

    def _routing_graphproject(self, project, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Route a query through the full graph using Orchestrator NN."""
        query_vec = attributes.get("query_vector")
        if query_vec is None:
            raise ValueError("Routing requires 'query_vector'")
        
        # Dummy graph state – in real impl aggregate node states
        graph_state = torch.zeros(128)
        
        manipulator = self._manipulator
        preds = manipulator.orchestrator_nn(torch.cat([query_vec, graph_state]))
        
        route = preds[0].sigmoid().tolist()  # placeholder path scores
        grow = preds[1].sigmoid().item() > 0.5
        share = preds[2].sigmoid().item() > 0.5
        
        result = {"route": route, "grow": grow, "share": share}
        logger.debug(f"RoutingSuper predicted: {result}")
        return self._build_response(project, True, "_routing_graphproject", result)

    def _routing(self, obj: Any, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return self._default_result(obj)