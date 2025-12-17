"""
Example usage of ClusterContainer
"""

import numpy as np
from datetime import datetime, timedelta
from agi.clustercontainer import ClusterContainer, ClusterType, ActivationPattern
from agi.nodeentity import NodeEntity
from agi.edgeentity import EdgeEntity, ConnectionType
from agi.knowledgechunk import KnowledgeChunk


def demonstrate_cluster_container():
    """Demonstrate ClusterContainer capabilities"""
    
    print("=== ClusterContainer Demonstration ===\n")
    
    # 1. Create different types of clusters
    print("1. Creating specialized clusters...")
    
    # Language processing cluster
    language_cluster = ClusterContainer(
        name="language_processing_center",
        cluster_type=ClusterType.LANGUAGE,
        specialization_vector=np.random.randn(512),
        activation_threshold=0.3,
        max_capacity=50
    )
    
    # Visual processing cluster
    visual_cluster = ClusterContainer(
        name="visual_processing_center",
        cluster_type=ClusterType.VISUAL,
        specialization_vector=np.random.randn(512),
        activation_threshold=0.4,
        max_capacity=60
    )
    
    # Integrative cluster (for higher-level processing)
    integrative_cluster = ClusterContainer(
        name="integrative_cortex",
        cluster_type=ClusterType.INTEGRATIVE,
        specialization_vector=np.random.randn(512),
        activation_threshold=0.2,
        max_capacity=100
    )
    
    print(f"   Created: {language_cluster}")
    print(f"   Created: {visual_cluster}")
    print(f"   Created: {integrative_cluster}\n")
    
    # 2. Add nodes to clusters
    print("2. Adding nodes to clusters...")
    
    # Add language processing nodes
    language_nodes = []
    for i in range(5):
        node = NodeEntity(
            name=f"lang_node_{i}",
            specialization=f"language_feature_{i}",
            input_dim=512,
            output_dim=512
        )
        language_cluster.add_node(node)
        language_nodes.append(node.name)
    
    # Add visual processing nodes
    visual_nodes = []
    for i in range(6):
        node = NodeEntity(
            name=f"vis_node_{i}",
            specialization=f"visual_feature_{i}",
            input_dim=1024,
            output_dim=1024
        )
        visual_cluster.add_node(node)
        visual_nodes.append(node.name)
    
    print(f"   Language cluster: {len(language_nodes)} nodes")
    print(f"   Visual cluster: {len(visual_nodes)} nodes")
    
    # Get stats to verify
    lang_stats = language_cluster.get_stats()
    vis_stats = visual_cluster.get_stats()
    
    print(f"   Language cluster composition: {lang_stats['composition']}")
    print(f"   Visual cluster composition: {vis_stats['composition']}\n")
    
    # 3. Create connections within clusters
    print("3. Creating intra-cluster connections...")
    
    # Create connections between language nodes (simplified)
    for i in range(len(language_nodes) - 1):
        edge = EdgeEntity(
            name=f"lang_edge_{i}",
            source_node=language_nodes[i],
            target_node=language_nodes[i + 1],
            weight=0.6,
            connection_type=ConnectionType.EXCITATORY
        )
        language_cluster.add(edge)
        language_cluster.intra_cluster_edges.append(edge.name)
    
    # Create connections between visual nodes
    for i in range(len(visual_nodes) - 1):
        edge = EdgeEntity(
            name=f"vis_edge_{i}",
            source_node=visual_nodes[i],
            target_node=visual_nodes[i + 1],
            weight=0.7,
            connection_type=ConnectionType.EXCITATORY
        )
        visual_cluster.add(edge)
        visual_cluster.intra_cluster_edges.append(edge.name)
    
    print(f"   Language cluster edges: {len(language_cluster.intra_cluster_edges)}")
    print(f"   Visual cluster edges: {len(visual_cluster.intra_cluster_edges)}\n")
    
    # 4. Connect clusters together
    print("4. Creating inter-cluster connections...")
    
    # Connect language and visual clusters to integrative cluster
    edge1_name = language_cluster.connect_clusters(
        target_cluster=integrative_cluster,
        connection_type="excitatory",
        weight=0.8
    )
    
    edge2_name = visual_cluster.connect_clusters(
        target_cluster=integrative_cluster,
        connection_type="excitatory",
        weight=0.9
    )
    
    # Create feedback connection
    edge3_name = integrative_cluster.connect_clusters(
        target_cluster=language_cluster,
        connection_type="modulatory",
        weight=0.3
    )
    
    print(f"   Language -> Integrative: {edge1_name}")
    print(f"   Visual -> Integrative: {edge2_name}")
    print(f"   Integrative -> Language (feedback): {edge3_name}")
    
    print(f"\n   Language cluster outputs: {language_cluster.output_clusters}")
    print(f"   Visual cluster outputs: {visual_cluster.output_clusters}")
    print(f"   Integrative cluster inputs: {integrative_cluster.input_clusters}\n")
    
    # 5. Demonstrate activation propagation
    print("5. Activation propagation demonstration...")
    
    # Create input signals for language cluster
    lang_inputs = {language_nodes[0]: 0.8, language_nodes[1]: 0.6}
    
    print(f"   Activating language cluster with {len(lang_inputs)} inputs...")
    lang_result = language_cluster.propagate_activation(
        input_signals=lang_inputs,
        max_depth=2,
        energy_budget=0.5
    )
    
    print(f"   Language cluster activation result:")
    print(f"     Status: {lang_result['status']}")
    print(f"     Output activation: {lang_result['output_activation']:.3f}")
    print(f"     Activated nodes: {len(lang_result['activated_nodes'])}")
    print(f"     Energy used: {lang_result['energy_used']:.3f}")
    
    # Activate visual cluster
    vis_inputs = {visual_nodes[0]: 0.9, visual_nodes[2]: 0.7}
    
    print(f"\n   Activating visual cluster...")
    vis_result = visual_cluster.propagate_activation(
        input_signals=vis_inputs,
        max_depth=3,
        energy_budget=0.6
    )
    
    print(f"   Visual cluster output activation: {vis_result['output_activation']:.3f}")
    print(f"   Visual cluster coherence: {vis_result['coherence']:.3f}\n")
    
    # 6. Demonstrate learning in cluster
    print("6. Cluster learning demonstration...")
    
    # Create learning experience data
    experience_data = {
        "training_pairs": [
            {
                "input": [0.1, 0.2, 0.3],
                "target": [0.4, 0.5, 0.6],
                "reward": 0.8
            },
            {
                "input": [0.3, 0.4, 0.5],
                "target": [0.6, 0.7, 0.8],
                "reward": 0.9
            }
        ]
    }
    
    print("   Teaching language cluster...")
    learning_result = language_cluster.learn_from_experience(
        experience_data=experience_data,
        learning_rate=0.01
    )
    
    print(f"   Learning results:")
    print(f"     Nodes updated: {learning_result['nodes_updated']}")
    print(f"     Edges updated: {learning_result['edges_updated']}")
    print(f"     Total reward: {learning_result['total_reward']:.3f}\n")
    
    # 7. Demonstrate knowledge storage and retrieval
    print("7. Knowledge management in cluster...")
    
    # Add knowledge chunks to integrative cluster
    for i in range(3):
        knowledge = KnowledgeChunk(
            name=f"cross_modal_knowledge_{i}",
            encoded_content=np.random.randn(768),
            source_node="integrative_cluster",
            confidence_score=0.7 + i * 0.1
        )
        integrative_cluster.add(knowledge)
    
    # Get updated stats
    int_stats = integrative_cluster.get_stats()
    print(f"   Integrative cluster knowledge chunks: {int_stats['composition']['knowledge_chunks']}")
    
    # 8. Demonstrate energy management
    print("\n8. Energy management demonstration...")
    
    print(f"   Before recharge:")
    print(f"     Language cluster energy: {language_cluster.energy_reserve:.3f}")
    print(f"     Visual cluster energy: {visual_cluster.energy_reserve:.3f}")
    
    # Recharge clusters
    new_energy_lang = language_cluster.recharge_energy(amount=0.2)
    new_energy_vis = visual_cluster.recharge_energy(amount=0.3)
    
    print(f"\n   After recharge:")
    print(f"     Language cluster energy: {new_energy_lang:.3f}")
    print(f"     Visual cluster energy: {new_energy_vis:.3f}\n")
    
    # 9. Demonstrate statistics and monitoring
    print("9. Cluster statistics and monitoring...")
    
    # Get comprehensive stats for all clusters
    clusters = [language_cluster, visual_cluster, integrative_cluster]
    
    for cluster in clusters:
        stats = cluster.get_stats()
        print(f"\n   {cluster.name} ({cluster.cluster_type.value}):")
        print(f"     Activation count: {stats['metrics']['activation_count']}")
        print(f"     Success rate: {stats['metrics']['success_rate']:.3f}")
        print(f"     Avg latency: {stats['metrics']['average_latency_ms']:.2f}ms")
        print(f"     Specialization score: {stats['metrics']['specialization_score']:.3f}")
        print(f"     Nodes: {stats['composition']['nodes']}/{stats['capacity']['max']}")
    
    # 10. Demonstrate serialization
    print("\n10. Serialization test...")
    
    # Serialize language cluster
    cluster_data = language_cluster.to_dict()
    print(f"   Serialized cluster keys: {list(cluster_data.keys())}")
    
    # Deserialize
    restored_cluster = ClusterContainer.from_dict(cluster_data)
    print(f"   Restored cluster: {restored_cluster}")
    
    # Compare
    original_stats = language_cluster.get_stats()
    restored_stats = restored_cluster.get_stats()
    
    print(f"\n   Comparison:")
    print(f"     Original nodes: {original_stats['composition']['nodes']}")
    print(f"     Restored nodes: {restored_stats['composition']['nodes']}")
    print(f"     Names match: {language_cluster.name == restored_cluster.name}")
    print(f"     Types match: {language_cluster.cluster_type == restored_cluster.cluster_type}")
    
    print("\n=== ClusterContainer Demonstration Complete ===")


def demonstrate_hierarchical_clusters():
    """Demonstrate fractal/hierarchical cluster organization"""
    
    print("\n=== Hierarchical Clusters Demonstration ===\n")
    
    # 1. Create a parent cluster
    print("1. Creating parent cluster with subclusters...")
    
    brain_hemisphere = ClusterContainer(
        name="left_hemisphere",
        cluster_type=ClusterType.INTEGRATIVE,
        max_capacity=1000  # Can contain many subclusters
    )
    
    # 2. Create specialized subclusters
    print("2. Creating specialized subclusters...")
    
    # Language area
    broca_area = ClusterContainer(
        name="broca_area",
        cluster_type=ClusterType.LANGUAGE,
        specialization_vector=np.random.randn(512),
        max_capacity=100
    )
    
    # Visual area
    v1_area = ClusterContainer(
        name="v1_visual_cortex",
        cluster_type=ClusterType.VISUAL,
        specialization_vector=np.random.randn(512),
        max_capacity=150
    )
    
    # Add nodes to subclusters
    for i in range(3):
        broca_area.add_node(NodeEntity(
            name=f"broca_neuron_{i}",
            specialization="speech_production"
        ))
        
        v1_area.add_node(NodeEntity(
            name=f"v1_neuron_{i}",
            specialization="edge_detection"
        ))
    
    # 3. Add subclusters to parent cluster
    print("3. Building hierarchy...")
    
    brain_hemisphere.add(broca_area)
    brain_hemisphere.add(v1_area)
    
    print(f"   Parent cluster: {brain_hemisphere}")
    print(f"   Contains {len(brain_hemisphere.get_items())} subclusters")
    print(f"   Subcluster types: {[c.cluster_type.value for c in brain_hemisphere.get_items()]}\n")
    
    # 4. Demonstrate hierarchical activation
    print("4. Hierarchical activation propagation...")
    
    # Activate Broca's area
    broca_inputs = {"broca_neuron_0": 0.8, "broca_neuron_1": 0.6}
    broca_result = broca_area.propagate_activation(
        input_signals=broca_inputs,
        max_depth=2
    )
    
    # Activate V1 area
    v1_inputs = {"v1_neuron_0": 0.9, "v1_neuron_2": 0.7}
    v1_result = v1_area.propagate_activation(
        input_signals=v1_inputs,
        max_depth=2
    )
    
    print(f"   Broca's area activation: {broca_result['output_activation']:.3f}")
    print(f"   V1 area activation: {v1_result['output_activation']:.3f}")
    
    # 5. Connect subclusters
    print("\n5. Connecting subclusters...")
    
    # Create connection between language and visual areas
    connection_name = broca_area.connect_clusters(
        target_cluster=v1_area,
        connection_type="excitatory",
        weight=0.6
    )
    
    print(f"   Created connection: {connection_name}")
    print(f"   Broca outputs to: {broca_area.output_clusters}")
    print(f"   V1 inputs from: {v1_area.input_clusters}\n")
    
    # 6. Demonstrate recursive statistics
    print("6. Recursive statistics...")
    
    # Get stats for entire hierarchy
    hemisphere_stats = brain_hemisphere.get_stats()
    
    print(f"   Hemisphere composition:")
    for item_type, count in hemisphere_stats["composition"].items():
        print(f"     {item_type}: {count}")
    
    # Get stats for each subcluster
    print(f"\n   Subcluster details:")
    for subcluster in brain_hemisphere.get_items():
        if isinstance(subcluster, ClusterContainer):
            sub_stats = subcluster.get_stats()
            print(f"     {subcluster.name}: {sub_stats['composition']['nodes']} nodes, "
                  f"activation: {sub_stats['current_activation']:.3f}")
    
    # 7. Demonstrate energy management in hierarchy
    print("\n7. Hierarchical energy management...")
    
    print(f"   Before recharge:")
    print(f"     Broca energy: {broca_area.energy_reserve:.3f}")
    print(f"     V1 energy: {v1_area.energy_reserve:.3f}")
    
    # Recharge parent cluster (should propagate to children)
    brain_hemisphere.recharge_energy(0.5)
    
    print(f"\n   After recharging parent:")
    print(f"     Broca energy: {broca_area.energy_reserve:.3f}")
    print(f"     V1 energy: {v1_area.energy_reserve:.3f}")
    
    # 8. Demonstrate fractal serialization
    print("\n8. Fractal serialization test...")
    
    # Serialize entire hierarchy
    hierarchy_data = brain_hemisphere.to_dict()
    
    print(f"   Serialized hierarchy contains:")
    print(f"     Parent cluster data: ✓")
    
    # Check if subclusters are included
    items = hierarchy_data.get("items", {})
    print(f"     Subclusters in data: {len(items)}")
    
    # Deserialize
    restored_hierarchy = ClusterContainer.from_dict(hierarchy_data)
    
    print(f"   Restored hierarchy: {restored_hierarchy}")
    print(f"   Restored subclusters: {len(restored_hierarchy.get_items())}")
    
    # Verify structure
    original_items = brain_hemisphere.get_items()
    restored_items = restored_hierarchy.get_items()
    
    print(f"   Structure preserved: {len(original_items) == len(restored_items)}")
    
    print("\n=== Hierarchical Demonstration Complete ===")


def demonstrate_cluster_evolution():
    """Demonstrate cluster growth and evolution"""
    
    print("\n=== Cluster Evolution Demonstration ===\n")
    
    # 1. Create an evolving cluster
    print("1. Creating evolving cluster...")
    
    evolving_cluster = ClusterContainer(
        name="evolving_knowledge_base",
        cluster_type=ClusterType.INTEGRATIVE,
        max_capacity=200,
        activation_threshold=0.2
    )
    
    # 2. Initial state
    print("2. Initial state...")
    initial_stats = evolving_cluster.get_stats()
    print(f"   Initial capacity: {initial_stats['capacity']['used']}/{initial_stats['capacity']['max']}")
    print(f"   Initial energy: {evolving_cluster.energy_reserve:.3f}\n")
    
    # 3. Simulate growth
    print("3. Simulating growth phase...")
    
    # Add initial nodes
    for i in range(10):
        node = NodeEntity(
            name=f"initial_neuron_{i}",
            specialization=f"basic_function_{i}"
        )
        evolving_cluster.add_node(node)
    
    print(f"   After initial growth: {evolving_cluster._node_count} nodes")
    
    # 4. Simulate learning phase
    print("\n4. Learning phase...")
    
    # Create and add knowledge chunks
    knowledge_added = 0
    for i in range(15):
        if evolving_cluster._knowledge_count < 50:  # Limit knowledge
            knowledge = KnowledgeChunk(
                name=f"learned_knowledge_{i}",
                encoded_content=np.random.randn(768),
                source_node="evolving_cluster",
                confidence_score=0.5 + i * 0.03
            )
            evolving_cluster.add(knowledge)
            knowledge_added += 1
    
    print(f"   Knowledge chunks added: {knowledge_added}")
    
    # 5. Simulate specialization
    print("\n5. Specialization phase...")
    
    # Add specialized nodes for different tasks
    specializations = ["pattern_recognition", "abstract_reasoning", 
                      "temporal_analysis", "spatial_reasoning"]
    
    for spec in specializations:
        for i in range(3):
            node = NodeEntity(
                name=f"{spec}_neuron_{i}",
                specialization=spec
            )
            evolving_cluster.add_node(node)
    
    print(f"   Specialized nodes added: {len(specializations) * 3}")
    
    # 6. Update cluster metrics through simulated usage
    print("\n6. Simulating usage and metric updates...")
    
    # Simulate multiple activations
    for i in range(20):
        # Create random inputs
        active_nodes = evolving_cluster.get_active_items()[:5]  # Get first 5 active nodes
        if active_nodes:
            input_signals = {node.name: 0.5 + np.random.random() * 0.5 
                           for node in active_nodes[:3]}
            
            # Propagate activation
            result = evolving_cluster.propagate_activation(
                input_signals=input_signals,
                max_depth=2,
                energy_budget=0.1
            )
            
            if i % 5 == 0:
                # Occasionally learn from experience
                evolving_cluster.learn_from_experience(
                    experience_data={"training_pairs": []},
                    learning_rate=0.01
                )
    
    # 7. Show evolved state
    print("\n7. Evolved state...")
    
    evolved_stats = evolving_cluster.get_stats()
    
    print(f"   Final composition:")
    for item_type, count in evolved_stats["composition"].items():
        print(f"     {item_type}: {count}")
    
    print(f"\n   Performance metrics:")
    print(f"     Activation count: {evolved_stats['metrics']['activation_count']}")
    print(f"     Success rate: {evolved_stats['metrics']['success_rate']:.3f}")
    print(f"     Specialization score: {evolved_stats['metrics']['specialization_score']:.3f}")
    print(f"     Adaptability score: {evolved_stats['metrics']['adaptability_score']:.3f}")
    
    print(f"\n   Capacity usage:")
    print(f"     Nodes: {evolved_stats['capacity']['used']}/{evolved_stats['capacity']['max']}")
    print(f"     Available: {evolved_stats['capacity']['available']}")
    
    print(f"\n   Energy state: {evolved_stats['energy_reserve']:.3f}")
    
    # 8. Demonstrate evolution through serialization
    print("\n8. Evolution snapshot...")
    
    # Take snapshot
    snapshot = evolving_cluster.to_dict()
    
    print(f"   Snapshot contains:")
    print(f"     Nodes: {len([k for k, v in snapshot.get('items', {}).items() 
                            if 'NodeEntity' in v.get('type', '')])}")
    print(f"     Knowledge chunks: {len([k for k, v in snapshot.get('items', {}).items() 
                                       if 'KnowledgeChunk' in v.get('type', '')])}")
    
    # Restore from snapshot
    restored_evolved = ClusterContainer.from_dict(snapshot)
    
    print(f"   Restored cluster: {restored_evolved}")
    print(f"   Evolution preserved: ✓")
    
    print("\n=== Evolution Demonstration Complete ===")


def demonstrate_cluster_network():
    """Demonstrate network of interconnected clusters"""
    
    print("\n=== Cluster Network Demonstration ===\n")
    
    # 1. Create a network of specialized clusters
    print("1. Creating cluster network...")
    
    # Create specialized clusters
    clusters = {
        "sensory": ClusterContainer(
            name="sensory_input",
            cluster_type=ClusterType.SENSORY_INPUT,
            max_capacity=80
        ),
        "perception": ClusterContainer(
            name="perceptual_processing",
            cluster_type=ClusterType.VISUAL,
            max_capacity=100
        ),
        "memory": ClusterContainer(
            name="memory_storage",
            cluster_type=ClusterType.MEMORY,
            max_capacity=150
        ),
        "executive": ClusterContainer(
            name="executive_control",
            cluster_type=ClusterType.EXECUTIVE,
            max_capacity=60
        ),
        "motor": ClusterContainer(
            name="motor_output",
            cluster_type=ClusterType.MOTOR_OUTPUT,
            max_capacity=70
        )
    }
    
    # Add nodes to each cluster
    for cluster_name, cluster in clusters.items():
        for i in range(5):
            node = NodeEntity(
                name=f"{cluster_name}_node_{i}",
                specialization=f"{cluster_name}_function"
            )
            cluster.add_node(node)
    
    print(f"   Created {len(clusters)} specialized clusters\n")
    
    # 2. Connect clusters in a processing pipeline
    print("2. Creating processing pipeline...")
    
    # Sensory -> Perception -> Memory -> Executive -> Motor
    connections = [
        ("sensory", "perception", 0.9, "excitatory"),
        ("perception", "memory", 0.8, "excitatory"),
        ("memory", "executive", 0.7, "excitatory"),
        ("executive", "motor", 0.9, "excitatory"),
        
        # Feedback connections
        ("executive", "memory", 0.4, "modulatory"),
        ("memory", "perception", 0.3, "modulatory")
    ]
    
    connection_names = []
    for source, target, weight, conn_type in connections:
        source_cluster = clusters[source]
        target_cluster = clusters[target]
        
        edge_name = source_cluster.connect_clusters(
            target_cluster=target_cluster,
            connection_type=conn_type,
            weight=weight
        )
        connection_names.append(edge_name)
    
    print(f"   Created {len(connections)} inter-cluster connections")
    print(f"   Pipeline: Sensory -> Perception -> Memory -> Executive -> Motor\n")
    
    # 3. Simulate information flow
    print("3. Simulating information flow...")
    
    # Start with sensory input
    sensory_inputs = {"sensory_node_0": 0.8, "sensory_node_1": 0.6}
    
    print("   Step 1: Sensory input")
    sensory_result = clusters["sensory"].propagate_activation(
        input_signals=sensory_inputs,
        max_depth=2
    )
    
    # Simulate propagation through network
    print(f"   Sensory output: {sensory_result['output_activation']:.3f}")
    
    # Continue through perception
    if sensory_result['status'] == 'success':
        perception_inputs = {f"perception_node_{i}": 0.7 for i in range(2)}
        
        print("\n   Step 2: Perceptual processing")
        perception_result = clusters["perception"].propagate_activation(
            input_signals=perception_inputs,
            max_depth=2
        )
        print(f"   Perception output: {perception_result['output_activation']:.3f}")
    
    # 4. Demonstrate network statistics
    print("\n4. Network-wide statistics...")
    
    total_nodes = 0
    total_edges = 0
    total_activation = 0.0
    
    for cluster_name, cluster in clusters.items():
        stats = cluster.get_stats()
        total_nodes += stats['composition']['nodes']
        total_edges += stats['composition']['edges']
        total_activation += stats['current_activation']
        
        print(f"   {cluster_name}: {stats['composition']['nodes']} nodes, "
              f"{stats['composition']['edges']} edges, "
              f"activation: {stats['current_activation']:.3f}")
    
    print(f"\n   Network totals:")
    print(f"     Total nodes: {total_nodes}")
    print(f"     Total edges: {total_edges}")
    print(f"     Avg activation: {total_activation / len(clusters):.3f}")
    
    # 5. Demonstrate coordinated learning
    print("\n5. Coordinated network learning...")
    
    # Create shared learning experience
    shared_experience = {
        "training_pairs": [
            {
                "input": list(range(10)),
                "target": list(range(10, 20)),
                "reward": 0.8
            }
        ]
    }
    
    learning_results = {}
    for cluster_name, cluster in clusters.items():
        result = cluster.learn_from_experience(
            experience_data=shared_experience,
            learning_rate=0.01
        )
        learning_results[cluster_name] = result
    
    print(f"   Network learning summary:")
    for cluster_name, result in learning_results.items():
        print(f"     {cluster_name}: {result['nodes_updated']} nodes updated, "
              f"reward: {result['total_reward']:.3f}")
    
    print("\n=== Network Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_cluster_container()
    demonstrate_hierarchical_clusters()
    demonstrate_cluster_evolution()
    demonstrate_cluster_network()