"""
Example usage of EdgeEntity
"""

import numpy as np
from datetime import datetime, timedelta
from agi.edgeentity import EdgeEntity, ConnectionType, LearningRule
from agi.nodeentity import NodeEntity


def demonstrate_edge_entity():
    """Demonstrate EdgeEntity capabilities"""
    
    print("=== EdgeEntity Demonstration ===\n")
    
    # 1. Create nodes first
    print("1. Creating nodes for edge connections...")
    
    node_a = NodeEntity(
        name="visual_cortex",
        specialization="image_processing",
        input_dim=1024,
        output_dim=512,
        use_cache=False
    )
    
    node_b = NodeEntity(
        name="semantic_processor",
        specialization="meaning_extraction",
        input_dim=512,
        output_dim=256,
        use_cache=False
    )
    
    node_c = NodeEntity(
        name="attention_controller",
        specialization="focus_management",
        input_dim=512,
        output_dim=512,
        use_cache=False
    )
    
    print(f"   Created nodes: {node_a.name}, {node_b.name}, {node_c.name}\n")
    
    # 2. Create different types of edges
    print("2. Creating different edge types...")
    
    # Excitatory edge (strengthens connection)
    excitatory_edge = EdgeEntity(
        name="vis_to_sem",
        source_node=node_a.name,
        target_node=node_b.name,
        weight=0.6,
        connection_type=ConnectionType.EXCITATORY,
        learning_rule=LearningRule.HEBB,
        transmission_delay=15.0
    )
    
    # Inhibitory edge (weakens connection)
    inhibitory_edge = EdgeEntity(
        name="attention_gate",
        source_node=node_c.name,
        target_node=node_b.name,
        weight=0.4,
        connection_type=ConnectionType.INHIBITORY,
        learning_rule=LearningRule.OJA,
        transmission_delay=5.0
    )
    
    # Modulatory edge (modulates connection)
    modulatory_edge = EdgeEntity(
        name="sem_modulator",
        source_node=node_b.name,
        target_node=node_a.name,
        weight=0.3,
        connection_type=ConnectionType.MODULATORY,
        learning_rule=LearningRule.STDP,
        transmission_delay=25.0
    )
    
    print(f"   Created excitatory edge: {excitatory_edge}")
    print(f"   Created inhibitory edge: {inhibitory_edge}")
    print(f"   Created modulatory edge: {modulatory_edge}\n")
    
    # 3. Demonstrate signal transmission
    print("3. Signal transmission demonstration...")
    
    # Transmit through excitatory edge
    print("   Excitatory edge transmission:")
    result = excitatory_edge.transmit(signal_strength=0.8)
    print(f"     Input: 0.8, Output: {result['output_strength']:.3f}")
    print(f"     Delay: {result['delay']}ms")
    
    # Transmit through inhibitory edge
    print("\n   Inhibitory edge transmission:")
    result = inhibitory_edge.transmit(signal_strength=0.6)
    print(f"     Input: 0.6, Output: {result['output_strength']:.3f}")
    print(f"     Note: Negative output indicates inhibition")
    
    # Transmit through modulatory edge
    print("\n   Modulatory edge transmission:")
    result = modulatory_edge.transmit(signal_strength=0.5)
    print(f"     Input: 0.5, Output: {result['output_strength']:.3f}")
    print(f"     Note: Modulatory edges enhance existing signals")
    
    # 4. Demonstrate learning (reinforcement)
    print("\n4. Edge learning (reinforcement) demonstration...")
    
    print("   Before reinforcement:")
    print(f"     Edge weight: {excitatory_edge.weight:.3f}")
    print(f"     Evolutionary score: {excitatory_edge.evolutionary_score:.3f}")
    
    # Apply reinforcement
    learning_result = excitatory_edge.reinforce(reinforcement_strength=0.2)
    
    print("\n   After positive reinforcement:")
    print(f"     Old weight: {learning_result['old_weight']:.3f}")
    print(f"     New weight: {learning_result['new_weight']:.3f}")
    print(f"     Weight change: {learning_result['weight_change']:.4f}")
    print(f"     New evolutionary score: {learning_result['evolutionary_score']:.3f}")
    
    # 5. Demonstrate transmission buffer
    print("\n5. Transmission buffer demonstration...")
    
    # Create edge with delay
    delayed_edge = EdgeEntity(
        name="delayed_connection",
        source_node=node_a.name,
        target_node=node_b.name,
        transmission_delay=100.0  # 100ms delay
    )
    
    # Transmit at time T
    transmit_time = datetime.now()
    delayed_edge.transmit(signal_strength=0.7, timestamp=transmit_time)
    
    print(f"   Transmitted at: {transmit_time.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"   Transmission delay: {delayed_edge.transmission_delay}ms")
    
    # Check immediately (should be empty)
    immediate_signals = delayed_edge.get_pending_signals()
    print(f"   Immediate check: {len(immediate_signals)} signals ready")
    
    # Check after delay
    future_time = transmit_time + timedelta(milliseconds=150)
    future_signals = delayed_edge.get_pending_signals(current_time=future_time)
    print(f"   After 150ms: {len(future_signals)} signals ready")
    
    if future_signals:
        signal = future_signals[0]
        print(f"   Signal strength: {signal['signal']:.3f}")
    
    # 6. Demonstrate statistics
    print("\n6. Edge statistics...")
    
    # Do more transmissions
    for i in range(5):
        excitatory_edge.transmit(signal_strength=0.3 + i * 0.1)
        if i % 2 == 0:
            excitatory_edge.reinforce(reinforcement_strength=0.05)
    
    stats = excitatory_edge.get_stats()
    
    print(f"   Usage frequency: {stats['usage_frequency']}")
    print(f"   Current weight: {stats['weight']:.3f}")
    print(f"   Weight history length: {stats['weight_history_length']}")
    print(f"   Age (days): {stats['age_days']:.2f}")
    print(f"   Evolutionary score: {stats['evolutionary_score']:.3f}")
    print(f"   Pending signals: {stats['pending_signals']}")
    print(f"   Total transmission: {stats['total_transmission']:.3f}")
    
    # 7. Demonstrate connection strength calculation
    print("\n7. Connection strength analysis...")
    
    strength = excitatory_edge.get_connection_strength()
    print(f"   Base weight: {excitatory_edge.weight:.3f}")
    print(f"   Bandwidth: {excitatory_edge.bandwidth:.3f}")
    print(f"   Plasticity factor: {excitatory_edge.plasticity_factor:.3f}")
    print(f"   Effective connection strength: {strength:.3f}")
    
    # 8. Demonstrate serialization
    print("\n8. Serialization test...")
    
    # Serialize edge
    edge_data = excitatory_edge.to_dict()
    print(f"   Serialized keys: {list(edge_data.keys())}")
    
    # Deserialize
    restored_edge = EdgeEntity.from_dict(edge_data)
    print(f"   Restored edge: {restored_edge}")
    
    # Compare
    original_stats = excitatory_edge.get_stats()
    restored_stats = restored_edge.get_stats()
    
    print(f"   Original weight: {original_stats['weight']:.3f}")
    print(f"   Restored weight: {restored_stats['weight']:.3f}")
    print(f"   Weights match: {abs(original_stats['weight'] - restored_stats['weight']) < 0.001}")
    
    print("\n=== Demonstration Complete ===")


def demonstrate_network_building():
    """Demonstrate building a small network with nodes and edges"""
    
    print("\n=== Network Building Demonstration ===\n")
    
    # Create a simple 3-node network
    input_node = NodeEntity(
        name="input_layer",
        specialization="sensory_input",
        input_dim=256,
        output_dim=128,
        use_cache=False
    )
    
    hidden_node = NodeEntity(
        name="hidden_layer",
        specialization="feature_extraction",
        input_dim=128,
        output_dim=64,
        use_cache=False
    )
    
    output_node = NodeEntity(
        name="output_layer",
        specialization="decision_making",
        input_dim=64,
        output_dim=32,
        use_cache=False
    )
    
    # Create edges
    edge1 = EdgeEntity(
        name="input_to_hidden",
        source_node=input_node.name,
        target_node=hidden_node.name,
        weight=0.7,
        connection_type=ConnectionType.EXCITATORY
    )
    
    edge2 = EdgeEntity(
        name="hidden_to_output",
        source_node=hidden_node.name,
        target_node=output_node.name,
        weight=0.8,
        connection_type=ConnectionType.EXCITATORY
    )
    
    # Feedback edge
    edge3 = EdgeEntity(
        name="output_to_hidden_feedback",
        source_node=output_node.name,
        target_node=hidden_node.name,
        weight=0.3,
        connection_type=ConnectionType.MODULATORY
    )
    
    print("Network structure:")
    print(f"  {input_node.name} → [{edge1.name}] → {hidden_node.name}")
    print(f"  {hidden_node.name} → [{edge2.name}] → {output_node.name}")
    print(f"  {output_node.name} → [{edge3.name}] → {hidden_node.name} (feedback)")
    print()
    
    # Simulate signal flow
    print("Simulating signal flow...")
    
    # Input signal
    input_signal = 0.9
    print(f"  Input signal strength: {input_signal}")
    
    # Transmit through first edge
    result1 = edge1.transmit(input_signal)
    hidden_input = result1['output_strength']
    print(f"  After edge1: {hidden_input:.3f}")
    
    # Process through hidden node (simplified)
    hidden_output = hidden_input * 0.8  # Simulate node processing
    print(f"  Hidden node output: {hidden_output:.3f}")
    
    # Transmit through second edge
    result2 = edge2.transmit(hidden_output)
    final_output = result2['output_strength']
    print(f"  Final output: {final_output:.3f}")
    
    # Feedback loop
    feedback_result = edge3.transmit(final_output * 0.5)
    print(f"  Feedback to hidden: {feedback_result['output_strength']:.3f}")
    
    print("\nNetwork statistics:")
    print(f"  Total edge usage: {edge1.usage_frequency + edge2.usage_frequency + edge3.usage_frequency}")
    print(f"  Average edge weight: {(edge1.weight + edge2.weight + edge3.weight) / 3:.3f}")
    
    print("\n=== Network Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_edge_entity()
    demonstrate_network_building()