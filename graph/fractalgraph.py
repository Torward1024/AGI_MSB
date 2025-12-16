# graph/fractalgraph.py
from base.basecontainer import BaseContainer
from .graphnode import GraphNode, Connection, KnowledgeItem, ConnectionType
from utils.text_processor import TextProcessor
from typing import List, Dict, Any, Optional
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from utils.logging_setup import logger
import os
import json

class FractalGraph(BaseContainer[GraphNode]):
    """Фрактальный граф с иерархическими уровнями.

    Наследуется от BaseContainer[GraphNode], поддерживает фрактальность через подграфы в вершинах.
    Обеспечивает параллелизм в операциях.
    """
    name: str
    max_level: int  # Максимальный уровень иерархии
    executor: ThreadPoolExecutor  # Для параллелизма
    training_data: List[tuple]  # Накопленные данные для обучения: (inputs, targets)
    config: Dict[str, Any]  # Конфигурационные параметры

    def __init__(self, name: str, config: Dict[str, Any], items: Dict[str, GraphNode] = None, **kwargs):
        """Инициализация FractalGraph.

        Args:
            name (str): Имя графа.
            config (Dict[str, Any]): Конфигурационные параметры.
            items (Dict[str, GraphNode]): Начальные вершины.
        """
        super().__init__(name=name, items=items, **kwargs)
        self.config = config
        self.max_level = config.get('graph_params', {}).get('max_level', 5)
        max_workers = config.get('graph_params', {}).get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.training_data = []
        logger.debug(f"Initialized FractalGraph {name} with max_level {self.max_level}")
        # Загрузить состояние, если файлы существуют
        self.load_state()

    def add_node(self, node: GraphNode) -> None:
        """Добавить вершину с проверкой уровня.

        Args:
            node (GraphNode): Вершина.
        """
        if node.level > self.max_level:
            raise ValueError(f"Node level {node.level} exceeds max_level {self.max_level}")
        super().add(node, copy_items=False)

    def create_node(self, name: str, level: int, **kwargs) -> GraphNode:
        """Создать новую вершину.

        Args:
            name (str): Имя.
            level (int): Уровень.
            **kwargs: Дополнительные параметры.

        Returns:
            GraphNode: Созданная вершина.
        """
        network_params = self.config.get('network_params', {})
        input_size = network_params.get('input_size', 10)
        hidden_size = network_params.get('hidden_size', 50)
        output_size = network_params.get('output_size', 100)
        node = GraphNode(name=name, level=level, input_size=input_size, hidden_size=hidden_size, output_size=output_size, **kwargs)
        self.add_node(node)
        return node

    async def distribute_knowledge_async(self, knowledge: Dict[str, Any]) -> None:
        """Распределить знания асинхронно.

        Args:
            knowledge (Dict[str, Any]): Знания для распределения.
        """
        tasks = []
        for node in self.get_items():
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._update_node_knowledge, node, knowledge
            )
            tasks.append(task)
        await asyncio.gather(*tasks)
        logger.debug(f"Distributed knowledge to {len(tasks)} nodes")

    def distribute_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Распределить знания синхронно.

        Args:
            knowledge (Dict[str, Any]): Знания.
        """
        for node in self.get_items():
            self._update_node_knowledge(node, knowledge)
        # asyncio.run(self.distribute_knowledge_async(knowledge))

    def _update_node_knowledge(self, node: GraphNode, knowledge: Dict[str, Any]) -> None:
        """Обновить знания вершины.

        Args:
            node (GraphNode): Вершина.
            knowledge (Dict[str, Any]): Знания.
        """
        for key, value in knowledge.items():
            node.update_knowledge(key, value)

    async def self_train_async(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 10) -> None:
        """Самообучение асинхронно.

        Args:
            inputs (torch.Tensor): Входы.
            targets (torch.Tensor): Цели.
            epochs (int): Эпохи.
        """
        tasks = []
        for node in self.get_items():
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, node.train_neural_net, inputs, targets, epochs
            )
            tasks.append(task)
        await asyncio.gather(*tasks)
        logger.debug(f"Self-trained {len(tasks)} nodes")

    def self_train(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = None) -> None:
        """Самообучение синхронно.

        Args:
            inputs, targets, epochs: Параметры обучения.
        """
        if epochs is None:
            epochs = self.config.get('network_params', {}).get('epochs', 10)
        lr = self.config.get('network_params', {}).get('learning_rate', 0.01)
        for node in self.get_items():
            node.train_neural_net(inputs, targets, epochs, lr)
        # asyncio.run(self.self_train_async(inputs, targets, epochs))

    def get_nodes_by_level(self, level: int) -> List[GraphNode]:
        """Получить вершины по уровню.

        Args:
            level (int): Уровень.

        Returns:
            List[GraphNode]: Вершины.
        """
        return [node for node in self.get_items() if node.level == level]

    def propagate_to_subgraphs(self, action: callable) -> None:
        """Распространить действие на подграфы.

        Args:
            action (callable): Действие.
        """
        for node in self.get_items():
            if node.sub_graph:
                action(node.sub_graph)

    def save_graph_structure(self) -> None:
        """Сохранить структуру графа в JSON."""
        structure = {}
        for node in self.get_items():
            structure[node.name] = {
                'level': node.level,
                'connections': [conn.to_dict() for conn in node.connections],
                'knowledge': {k: v.to_dict() for k, v in node.knowledge.items()},
                # sub_graph игнорируем для простоты
            }
        with open("agi_structure.json", 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False)
        logger.debug(f"Saved graph structure for {self.name}")

    def load_graph_structure(self) -> None:
        """Загрузить структуру графа из JSON."""
        if os.path.exists("agi_structure.json"):
            with open("agi_structure.json", 'r', encoding='utf-8') as f:
                structure = json.load(f)
            for name, data in structure.items():
                connections = [Connection.from_dict(c) for c in data['connections']]
                knowledge = {k: KnowledgeItem.from_dict(v) for k, v in data['knowledge'].items()}
                node = GraphNode(name=name, level=data['level'], connections=connections, knowledge=knowledge)
                self.add_node(node)
            logger.debug(f"Loaded graph structure for {self.name}")

    def save_state(self) -> None:
        """Сохранить состояние всех вершин."""
        self.save_graph_structure()
        model_state = {}
        for node in self.get_items():
            model_state[node.name] = node.neural_net.state_dict()
        torch.save(model_state, 'agi_model.pth')
        logger.debug(f"Saved state for all nodes in FractalGraph {self.name}")

    def load_state(self) -> None:
        """Загрузить состояние всех вершин, если файлы существуют."""
        self.load_graph_structure()
        if os.path.exists('agi_model.pth'):
            model_state = torch.load('agi_model.pth')
            for node in self.get_items():
                if node.name in model_state:
                    node.neural_net.load_state_dict(model_state[node.name])
            logger.debug(f"Loaded state for all nodes in FractalGraph {self.name}")

    def train_on_accumulated_data(self, epochs: int = None) -> None:
        """Обучить все вершины на накопленных данных."""
        if not self.training_data:
            return
        if epochs is None:
            epochs = self.config.get('graph_params', {}).get('background_train_epochs', 5)
        inputs = torch.stack([d[0] for d in self.training_data])
        targets = torch.stack([d[1] for d in self.training_data])
        self.self_train(inputs, targets, epochs)
        self.training_data = []  # Очистить после обучения
        logger.debug(f"Trained on {len(inputs)} accumulated data points")

    def train_nodes_on_own_data(self, epochs: int = 1) -> None:
        """Обучить каждую вершину на ее собственных накопленных данных."""
        lr = self.config.get('network_params', {}).get('learning_rate', 0.01)
        for node in self.get_items():
            if node.training_data:
                inputs = torch.stack([d[0] for d in node.training_data])
                targets = torch.stack([d[1] for d in node.training_data])
                node.train_neural_net(inputs, targets, epochs, lr)
                node.training_data = []  # Очистить после обучения
                logger.debug(f"Trained node {node.name} on {len(inputs)} data points")

    def reset(self) -> None:
        """Сбросить граф: удалить все вершины и очистить файлы."""
        self.clear()
        if os.path.exists("agi_structure.json"):
            os.remove("agi_structure.json")
        if os.path.exists("agi_model.pth"):
            os.remove("agi_model.pth")
        logger.debug(f"Reset FractalGraph {self.name}")

    def get_status(self) -> str:
        """Получить статус графа: количество вершин, уровни, связи."""
        nodes = self.get_items()
        total_nodes = len(nodes)
        levels = {}
        connections = 0
        for node in nodes:
            levels[node.level] = levels.get(node.level, 0) + 1
            connections += len(node.connections)
        status = f"Граф '{self.name}':\n"
        status += f"Всего вершин: {total_nodes}\n"
        status += "Вершины по уровням:\n"
        for lvl in sorted(levels.keys()):
            status += f"  Уровень {lvl}: {levels[lvl]}\n"
        status += f"Общее количество связей: {connections}\n"
        return status

    def create_associative_connections(self, threshold: float = 0.5) -> None:
        """Создать ассоциативные связи между вершинами на основе семантического сходства.

        Args:
            threshold (float): Порог сходства.
        """
        nodes = self.get_items()
        for node in nodes:
            node.create_associative_connections(nodes, threshold)
        logger.debug(f"Created associative connections for {len(nodes)} nodes")

    def process_text_query(self, query: str) -> tuple:
        """Обработать текстовый запрос, используя существующие вершины и связи.

        Args:
            query (str): Запрос.

        Returns:
            tuple: (response, node, embedding, target_logits)
        """
        stop_words = self.config.get('stop_words', [])
        tokens = [t for t in TextProcessor.tokenize(query) if t not in stop_words]
        # Найти существующую вершину для обработки
        node = None
        for token in tokens:
            if token in [n.name for n in self.get_items()]:
                node = self.get(token)
                break
        if not node:
            # Создать новую вершину для первого токена или запроса
            if tokens:
                node_name = tokens[0]
            else:
                node_name = query.replace(' ', '_')[:20]
            if node_name not in [n.name for n in self.get_items()]:
                self.create_node(node_name, level=0)
            node = self.get(node_name)
        embedding = node.embed_text(query)
        output = node.process_data(embedding)
        # Активировать связанные вершины
        for conn in node.connections:
            if conn.target_name in [n.name for n in self.get_items()]:
                linked_node = self.get(conn.target_name)
                linked_output = linked_node.process_data(embedding)
                if conn.conn_type == ConnectionType.EXCITATORY:
                    output += linked_output * conn.weight
                elif conn.conn_type == ConnectionType.INHIBITORY:
                    output -= linked_output * conn.weight
        top_n = self.config.get('network_params', {}).get('top_n', 5)
        response = TextProcessor.generate_text(output, top_n)
        # Создать target на основе сгенерированного ответа для потенциального обучения
        target_logits = torch.zeros(TextProcessor.VOCAB_SIZE)
        tokens_resp = TextProcessor.tokenize(response)
        for token in tokens_resp:
            if token in TextProcessor.VOCAB:
                idx = TextProcessor.VOCAB[token]
                target_logits[idx] += 1
        return response, node, embedding, target_logits