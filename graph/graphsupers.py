# graph/graphsupers.py
from super.super import Super
from .graphnode import GraphNode
from .fractalgraph import FractalGraph
from typing import Dict, Any, Optional
import torch
import asyncio
from utils.logging_setup import logger

class CreateNodeSuper(Super):
    OPERATION = "create_node"

    def _create_node_fractalgraph(self, obj: FractalGraph, attributes: Dict[str, Any]) -> GraphNode:
        name = attributes.get("name")
        level = attributes.get("level", 0)
        if not name:
            raise ValueError("Name is required for create_node")
        node = obj.create_node(name, level, **{k: v for k, v in attributes.items() if k not in ["name", "level"]})
        return node

class DistributeKnowledgeSuper(Super):
    OPERATION = "distribute_knowledge"

    def _distribute_knowledge_fractalgraph(self, obj: FractalGraph, attributes: Dict[str, Any]) -> None:
        knowledge = attributes.get("knowledge", {})
        obj.distribute_knowledge(knowledge)

class SelfTrainSuper(Super):
    OPERATION = "self_train"

    def _self_train_fractalgraph(self, obj: FractalGraph, attributes: Dict[str, Any]) -> None:
        inputs = attributes.get("inputs")
        targets = attributes.get("targets")
        epochs = attributes.get("epochs", 10)
        if inputs is None or targets is None:
            raise ValueError("Inputs and targets are required for self_train")
        obj.self_train(inputs, targets, epochs)