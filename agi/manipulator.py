# agi/manipulator.py
from typing import List, Type, Optional, Dict, Any
from common.super.manipulator import Manipulator
from agi.project import GraphProject
from agi.super.growth import GrowthSuper
from agi.super.routing import RoutingSuper
from agi.super.sharing import SharingSuper
from common.utils.logging_setup import logger
import torch.nn as nn
from dgl.nn import GraphConv
import torch

class AGIManipulator(Manipulator):
    """Full AGI Manipulator – orchestrates all operations via registered Super classes.

    All AGI activities (growth, routing, sharing) are performed exclusively through
    registered Super instances. The Manipulator itself only provides the Orchestrator NN
    and facade for process_request.
    """
    def __init__(self, project: Optional[GraphProject] = None):
        from agi.node import NodeEntity
        from agi.cluster import ClusterContainer
        
        base_classes: List[Type] = [GraphProject, ClusterContainer, NodeEntity]
        super().__init__(managing_object=project, base_classes=base_classes)

        # Register full Super implementations
        self.register_operation(GrowthSuper(self), operation="growth")
        self.register_operation(RoutingSuper(self), operation="routing")
        self.register_operation(SharingSuper(self), operation="sharing")

        self.orchestrator_nn = self._build_orchestrator_nn()
        logger.info("AGIManipulator fully initialized with growth, routing, sharing operations")

    def _build_orchestrator_nn(self) -> nn.Module:
        """Build the Orchestrator GNN used by RoutingSuper."""
        return nn.Sequential(
            GraphConv(256, 128),   # query_vec + graph_state = 256 dim
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)       # [route_score, grow_prob, share_prob]
        )

    def clear(self) -> None:
        """Clean up resources."""
        self.orchestrator_nn = None
        super().clear_ops()
        super().clear_cache()
        super().clear_base_classes()
        logger.debug("AGIManipulator cleared")

    def __repr__(self) -> str:
        return f"AGIManipulator(operations={list(self._operations.keys())})"