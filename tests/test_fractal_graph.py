import pytest
import torch
from graph.graphnode import GraphNode
from graph.fractalgraph import FractalGraph
from graph.graphmanipulator import GraphManipulator

class TestGraphNode:
    def test_initialization(self):
        node = GraphNode("test_node", level=1)
        assert node.name == "test_node"
        assert node.level == 1
        assert node.connections == []
        assert node.knowledge == {}
        assert node.sub_graph is None
        assert isinstance(node.neural_net, torch.nn.Module)

    def test_process_data(self):
        node = GraphNode("test_node")
        data = torch.randn(1, 10)
        output = node.process_data(data)
        assert output.shape == (1, 100)

    def test_add_connection(self):
        node = GraphNode("test_node")
        node.add_connection("node2")
        assert "node2" in node.connections

    def test_update_knowledge(self):
        node = GraphNode("test_node")
        node.update_knowledge("key1", "value1")
        assert node.knowledge["key1"] == "value1"

    def test_train_neural_net(self):
        node = GraphNode("test_node")
        inputs = torch.randn(10, 10)
        targets = torch.randn(10, 1)
        node.train_neural_net(inputs, targets, epochs=1)
        # Проверка, что сеть обучена (сложно проверить без метрик, но отсутствие ошибок - хорошо)

class TestFractalGraph:
    def test_initialization(self):
        graph = FractalGraph("test_graph", max_level=3)
        assert graph.name == "test_graph"
        assert graph.max_level == 3
        assert len(graph.get_items()) == 0

    def test_create_node(self):
        graph = FractalGraph("test_graph")
        node = graph.create_node("node1", level=0)
        assert node.name == "node1"
        assert node.level == 0
        assert graph.has_item("node1")

    def test_add_node_level_exceed(self):
        graph = FractalGraph("test_graph", max_level=2)
        node = GraphNode("node1", level=3)
        with pytest.raises(ValueError):
            graph.add_node(node)

    def test_distribute_knowledge(self):
        graph = FractalGraph("test_graph")
        node1 = graph.create_node("node1", level=0)
        node2 = graph.create_node("node2", level=0)
        knowledge = {"info": "test"}
        graph.distribute_knowledge(knowledge)
        assert node1.knowledge["info"] == "test"
        assert node2.knowledge["info"] == "test"

    def test_self_train(self):
        graph = FractalGraph("test_graph")
        node1 = graph.create_node("node1", level=0)
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        graph.self_train(inputs, targets, epochs=1)
        # Проверка без ошибок

    def test_get_nodes_by_level(self):
        graph = FractalGraph("test_graph")
        node1 = graph.create_node("node1", level=0)
        node2 = graph.create_node("node2", level=1)
        level0_nodes = graph.get_nodes_by_level(0)
        assert len(level0_nodes) == 1
        assert level0_nodes[0].name == "node1"

    def test_fractality(self):
        graph = FractalGraph("main_graph")
        sub_graph = FractalGraph("sub_graph")
        node = graph.create_node("node_with_sub", level=0, sub_graph=sub_graph)
        assert node.sub_graph.name == "sub_graph"

class TestGraphManipulator:
    def test_initialization(self):
        graph = FractalGraph("test_graph")
        manipulator = GraphManipulator(managing_object=graph)
        assert manipulator.get_managing_object() == graph
        assert "create_node" in manipulator.get_supported_operations()

    def test_create_node_via_process_request(self):
        graph = FractalGraph("test_graph")
        manipulator = GraphManipulator(managing_object=graph)
        request = {"operation": "create_node", "attributes": {"name": "node1", "level": 0}}
        result = manipulator.process_request(request)
        assert result["status"] == True
        assert graph.has_item("node1")

    def test_distribute_knowledge_via_process_request(self):
        graph = FractalGraph("test_graph")
        node = graph.create_node("node1", level=0)
        manipulator = GraphManipulator(managing_object=graph)
        request = {"operation": "distribute_knowledge", "attributes": {"knowledge": {"key": "value"}}}
        result = manipulator.process_request(request)
        assert result["status"] == True
        assert node.knowledge["key"] == "value"

    def test_self_train_via_process_request(self):
        graph = FractalGraph("test_graph")
        node = graph.create_node("node1", level=0)
        manipulator = GraphManipulator(managing_object=graph)
        inputs = torch.randn(5, 10)
        targets = torch.randn(5, 1)
        request = {"operation": "self_train", "attributes": {"inputs": inputs, "targets": targets, "epochs": 1}}
        result = manipulator.process_request(request)
        assert result["status"] == True