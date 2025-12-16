# graph/graphnode.py
import torch
import torch.nn as nn
from base.baseentity import BaseEntity
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from utils.logging_setup import logger
from utils.text_processor import TextProcessor
import hashlib
import os
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from .fractalgraph import FractalGraph

class ConnectionType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"

class Connection:
    """Связь между вершинами с весом и типом."""
    def __init__(self, target_name: str, weight: float = 1.0, conn_type: ConnectionType = ConnectionType.EXCITATORY):
        self.target_name = target_name
        self.weight = weight
        self.conn_type = conn_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_name': self.target_name,
            'weight': self.weight,
            'conn_type': self.conn_type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Connection':
        return cls(
            target_name=data['target_name'],
            weight=data['weight'],
            conn_type=ConnectionType(data['conn_type'])
        )

class KnowledgeType(Enum):
    FACT = "fact"
    PATTERN = "pattern"
    ASSOCIATION = "association"

class KnowledgeItem:
    """Элемент знания с типом и историей обновлений."""
    def __init__(self, key: str, value: Any, k_type: KnowledgeType = KnowledgeType.FACT):
        self.key = key
        self.value = value
        self.k_type = k_type
        self.history: List[Dict[str, Any]] = []
        self.update(value)

    def update(self, new_value: Any) -> None:
        """Обновить значение и добавить в историю."""
        timestamp = datetime.now().isoformat()
        self.history.append({
            'timestamp': timestamp,
            'old_value': self.value,
            'new_value': new_value
        })
        self.value = new_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'value': self.value,
            'k_type': self.k_type.value,
            'history': self.history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        item = cls(
            key=data['key'],
            value=data['value'],
            k_type=KnowledgeType(data['k_type'])
        )
        item.history = data['history']
        return item

class MiniNeuralNet(nn.Module):
    """Простая мини-нейронная сеть для GraphNode."""
    def __init__(self, input_size: int = 10, hidden_size: int = 50, output_size: int = 100):
        super(MiniNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        logger.debug(f"MiniNeuralNet initialized with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class GraphNode(BaseEntity):
    """Вершина графа с мини-нейронной сетью.

    Наследуется от BaseEntity, содержит мини-нейронную сеть для обработки данных.
    Поддерживает связи с другими вершинами и может содержать подграф для фрактальности.
    """
    name: str
    level: int  # Уровень в иерархии
    connections: List[Connection]  # Список связей с весами и типами
    knowledge: Dict[str, KnowledgeItem]  # Структурированные знания с типами и историей
    sub_graph: Optional['FractalGraph']  # Подграф для фрактальности
    neural_net: MiniNeuralNet  # Мини-нейронная сеть
    training_data: List[tuple]  # Накопленные данные для обучения: (inputs, targets)

    def __init__(self, name: str, level: int = 0, connections: List[Connection] = None,
                   knowledge: Dict[str, KnowledgeItem] = None, sub_graph: Optional['FractalGraph'] = None,
                   input_size: int = 10, hidden_size: int = 50, output_size: int = 100, **kwargs):
        """Инициализация GraphNode.

        Args:
            name (str): Имя вершины.
            level (int): Уровень в иерархии.
            connections (List[Connection]): Список связей.
            knowledge (Dict[str, KnowledgeItem]): Знания вершины.
            sub_graph (Optional[FractalGraph]): Подграф.
            input_size, hidden_size, output_size: Размеры для нейронной сети.
        """
        # Для совместимости с BaseEntity, но переопределим
        super().__init__(name=name, level=level, connections=[], knowledge={}, sub_graph=sub_graph, **kwargs)
        self.connections = connections or []
        self.knowledge = knowledge or {}
        self.neural_net = MiniNeuralNet(input_size, hidden_size, output_size)
        self.training_data = []
        logger.debug(f"Initialized GraphNode {name} at level {level}")

    def process_data(self, data: torch.Tensor) -> torch.Tensor:
        """Обработка данных через нейронную сеть.

        Args:
            data (torch.Tensor): Входные данные.

        Returns:
            torch.Tensor: Выход сети.
        """
        with torch.no_grad():
            output = self.neural_net(data)
            logger.debug(f"process_data: output size = {output.size()}")
            return output

    def add_connection(self, node_name: str, weight: float = 1.0, conn_type: ConnectionType = ConnectionType.EXCITATORY) -> None:
        """Добавить связь с другой вершиной.

        Args:
            node_name (str): Имя вершины.
            weight (float): Вес связи.
            conn_type (ConnectionType): Тип связи.
        """
        if not any(conn.target_name == node_name for conn in self.connections):
            self.connections.append(Connection(node_name, weight, conn_type))
            logger.debug(f"Added connection to {node_name} with weight {weight} and type {conn_type.value} in GraphNode {self.name}")

    def remove_connection(self, node_name: str) -> None:
        """Удалить связь.

        Args:
            node_name (str): Имя вершины.
        """
        self.connections = [conn for conn in self.connections if conn.target_name != node_name]
        logger.debug(f"Removed connection to {node_name} in GraphNode {self.name}")

    def update_knowledge(self, key: str, value: Any, k_type: KnowledgeType = KnowledgeType.FACT) -> None:
        """Обновить знания.

        Args:
            key (str): Ключ.
            value (Any): Значение.
            k_type (KnowledgeType): Тип знания.
        """
        if key in self.knowledge:
            self.knowledge[key].update(value)
        else:
            self.knowledge[key] = KnowledgeItem(key, value, k_type)
        logger.debug(f"Updated knowledge {key} in GraphNode {self.name}")

    def train_neural_net(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 10, lr: float = 0.01) -> None:
        """Обучить нейронную сеть.

        Args:
            inputs (torch.Tensor): Входы.
            targets (torch.Tensor): Цели.
            epochs (int): Количество эпох.
            lr (float): Скорость обучения.
        """
        optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.neural_net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Node {self.name}, Epoch {epoch}, Loss: {loss.item()}")
        logger.debug(f"Trained neural net in GraphNode {self.name} for {epochs} epochs")

    def save_state(self) -> None:
        """Сохранить состояние нейронной сети в файл."""
        torch.save(self.neural_net.state_dict(), f"{self.name}.pth")
        logger.debug(f"Saved state for GraphNode {self.name}")

    def load_state(self) -> None:
        """Загрузить состояние нейронной сети из файла, если существует."""
        if os.path.exists(f"{self.name}.pth"):
            self.neural_net.load_state_dict(torch.load(f"{self.name}.pth"))
            logger.debug(f"Loaded state for GraphNode {self.name}")
        else:
            logger.debug(f"No saved state found for GraphNode {self.name}")

    def embed_text(self, text: str) -> torch.Tensor:
        """Embedding текста.

        Args:
            text (str): Текст.

        Returns:
            torch.Tensor: Embedding.
        """
        return TextProcessor.embed_text(text, self.neural_net.fc1.in_features)

    def generate_text(self, logits: torch.Tensor, top_n: int = 5) -> str:
        """Генерация текста из logits с softmax и топ-N.

        Args:
            logits (torch.Tensor): Logits.
            top_n (int): Количество топ слов.

        Returns:
            str: Текст.
        """
        logger.debug(f"generate_text: logits size = {logits.size()}")
        return TextProcessor.generate_text(logits, top_n)

    def get_embedding(self) -> torch.Tensor:
        """Получить embedding вершины на основе имени и knowledge."""
        text = self.name
        for item in self.knowledge.values():
            if isinstance(item.value, str):
                text += " " + item.value
        return self.embed_text(text)

    def create_associative_connections(self, other_nodes: List['GraphNode'], threshold: float = 0.5) -> None:
        """Создать ассоциативные связи на основе семантического сходства.

        Args:
            other_nodes (List[GraphNode]): Другие вершины.
            threshold (float): Порог сходства для создания связи.
        """
        my_emb = self.get_embedding()
        for node in other_nodes:
            if node.name == self.name:
                continue
            other_emb = node.get_embedding()
            similarity = torch.cosine_similarity(my_emb.unsqueeze(0), other_emb.unsqueeze(0)).item()
            if similarity > threshold:
                conn_type = ConnectionType.EXCITATORY if similarity > 0.7 else ConnectionType.INHIBITORY
                weight = similarity
                self.add_connection(node.name, weight, conn_type)
                logger.debug(f"Created associative connection from {self.name} to {node.name} with similarity {similarity}")