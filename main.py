# agi/main.py
from agi.project import GraphProject
from agi.node import NodeEntity
from agi.manipulator import AGIManipulator
from common.utils.logging_setup import logger
import torch
import sys

def main():
    """Main CLI application for interacting with the AGI system.

    Supports commands:
    - query: <text> - Process a query (dummy vector).
    - status - Get AGI status.
    - save: <path> - Save state to path.
    - load: <path> - Load state from path.
    - growth: cluster=<cluster> node_name=<name> - Grow a node in cluster.
    - routing: query=<text> - Route a dummy query.
    - sharing: cluster=<cluster> source=<node> target=<node> - Share knowledge.
    - train: <dummy> - Dummy train command.
    - exit - Quit.
    """
    project = GraphProject()
    manipulator = AGIManipulator(project)
    
    # Bootstrap example structure
    project.create_item("default_cluster")
    cluster = project.get_item("default_cluster")
    nn_config = {"layers": [128, 64, 128], "activation": "relu"}
    node1 = NodeEntity(name="node1", nn_config=nn_config, memory_dim=128, level=0)
    node2 = NodeEntity(name="node2", nn_config=nn_config, memory_dim=128, level=0)
    cluster.add(node1)
    cluster.add(node2)
    cluster.add_edge("node1", "node2")

    logger.info("AGI CLI started. Enter commands.")

    while True:
        cmd = input("> ").strip()
        if not cmd:
            continue
        parts = cmd.split()
        op = parts[0].lower()

        try:
            if op == "exit":
                break
            elif op == "status":
                result = manipulator.process_request({"operation": "status_graphproject", "obj": project})
                print(result["result"])
            elif op == "save:":
                path = parts[1]
                manipulator.process_request({"operation": "saveload", "obj": project, "attributes": {"mode": "save", "path": path}})
                print("Saved.")
            elif op == "load:":
                path = parts[1]
                manipulator.process_request({"operation": "saveload", "obj": project, "attributes": {"mode": "load", "path": path}})
                print("Loaded.")
            elif op == "growth:":
                attrs = dict(p.split('=') for p in parts[1:])
                cluster_name = attrs.get("cluster", "default_cluster")
                obj = project.get_item(cluster_name)
                attrs["name"] = attrs.get("node_name", "new_node")
                result = manipulator.process_request({"operation": "growth", "obj": obj, "attributes": attrs})
                print(result["result"])
            elif op == "routing:":
                query_text = ' '.join(parts[1:])
                query_vec = torch.randn(128)  # Dummy
                result = manipulator.process_request({"operation": "routing", "obj": project, "attributes": {"query_vector": query_vec}})
                print(result["result"])
            elif op == "sharing:":
                attrs = dict(p.split('=') for p in parts[1:])
                cluster_name = attrs.get("cluster", "default_cluster")
                source_name = attrs["source"]
                obj = project.get_item(cluster_name).get_item(source_name)
                attrs["target"] = attrs["target"]
                result = manipulator.process_request({"operation": "sharing", "obj": obj, "attributes": attrs})
                print(result["result"])
            elif op == "query:":
                query_text = ' '.join(parts[1:])
                query_vec = torch.randn(128)  # Dummy vectorization
                # Dummy propagation
                outputs = cluster.propagate(query_vec)
                print(f"Dummy response: {outputs}")
            elif op == "train:":
                # Dummy training
                input_vec = torch.randn(128)
                target = torch.randn(128)
                for node in cluster._items.values():
                    node.learn(input_vec, target)
                print("Dummy training completed.")
            else:
                print("Unknown command.")
        except Exception as e:
            print(f"Error: {e}")

    manipulator.clear()
    logger.info("AGI CLI exited.")

if __name__ == "__main__":
    main()