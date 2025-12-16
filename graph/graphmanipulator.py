# graph/graphmanipulator.py
from mega.manipulator import Manipulator
from .fractalgraph import FractalGraph
from .graphsupers import CreateNodeSuper, DistributeKnowledgeSuper, SelfTrainSuper
from typing import Dict, Any, Optional
from utils.logging_setup import logger

class GraphManipulator(Manipulator):
    """Манипулятор для управления фрактальным графом.

    Наследуется от Manipulator, регистрирует супер-классы для операций.
    """

    def __init__(self, managing_object: Optional[FractalGraph] = None, **kwargs):
        """Инициализация GraphManipulator.

        Args:
            managing_object (Optional[FractalGraph]): Управляемый граф.
        """
        super().__init__(managing_object=managing_object, base_classes=[FractalGraph], **kwargs)
        # Регистрация супер-классов
        self.register_operation(CreateNodeSuper(), "create_node")
        self.register_operation(DistributeKnowledgeSuper(), "distribute_knowledge")
        self.register_operation(SelfTrainSuper(), "self_train")
        logger.debug("Initialized GraphManipulator with registered operations")