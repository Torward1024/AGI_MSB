"""
Example usage of NodeEntity
"""

import numpy as np
import torch
from agi.nodeentity import NodeEntity

def demonstrate_node_entity():
    """Demonstrate NodeEntity capabilities"""
    
    print("=== NodeEntity Demonstration ===\n")
    
    # 1. Create specialized nodes
    print("1. Creating specialized nodes...")
    
    language_node = NodeEntity(
        name="language_processor",
        specialization="syntax_analysis",
        input_dim=512,
        output_dim=512,
        confidence_threshold=0.4,
        network_type="mlp"
    )
    
    vision_node = NodeEntity(
        name="vision_processor",
        specialization="object_recognition",
        input_dim=1024,
        output_dim=1024,
        confidence_threshold=0.5,
        network_type="transformer"
    )
    
    print(f"   Created: {language_node}")
    print(f"   Created: {vision_node}\n")
    
    # 2. Demonstrate forward pass
    print("2. Forward pass demonstration...")
    
    # Create a sample input
    sample_input = np.random.randn(512).astype(np.float32)
    
    # Process through language node
    result = language_node.forward(sample_input)
    
    print(f"   Input shape: {sample_input.shape}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Проверяем, есть ли статус (ошибка) или output не None
    if result.get('status') is not None:
        # Узел не активировался (inactive или low_confidence)
        print(f"   Status: {result['status']}")
        print(f"   Error: {result.get('error', 'No error message')}")
    else:
        # Успешная активация - выводим информацию о выводе
        if result['output'] is not None:
            print(f"   Output shape: {result['output'].shape}")
        else:
            print(f"   Output: None (проверьте логи)")
        print(f"   Memory matches: {result['memory_matches']}")
    print()
    
    # 3. Demonstrate learning
    print("3. Learning demonstration...")
    
    learning_result = language_node.learn_from_experience(
        input_vector=sample_input,
        target_output=np.random.randn(512),
        reward=0.8
    )
    
    print(f"   Learning loss: {learning_result['loss']:.4f}")
    print(f"   Memory size after learning: {learning_result['memory_size']}\n")
    
    # 4. Demonstrate memory operations
    print("4. Memory operations...")
    
    # Add some patterns to memory
    for i in range(5):
        pattern = np.random.randn(512)
        language_node.add_to_memory(
            vector=pattern,
            label=f"example_pattern_{i}",
            confidence=0.7 + i * 0.05,
            metadata={"category": f"type_{i % 3}"}
        )
    
    # Query memory
    query = np.random.randn(512)
    memory_results = language_node.query_memory(query, k=3, threshold=0.5)
    
    # Get stats to show pattern count
    stats = language_node.get_stats()
    pattern_count = stats['memory']['pattern_count']
    
    print(f"   Stored {pattern_count} patterns")
    print(f"   Memory query returned {len(memory_results)} matches")
    
    if memory_results:
        pattern, similarity = memory_results[0]
        print(f"   Best match: '{pattern.label}' with similarity {similarity:.3f}")
    print()
    
    # 5. Demonstrate connectivity
    print("5. Node connectivity...")
    
    language_node.connect_input("text_input", "text_preprocessor")
    language_node.connect_output("parsed_output", "semantic_analyzer")
    
    print(f"   Input slots: {language_node.input_slots}")
    print(f"   Output slots: {language_node.output_slots}\n")
    
    # 6. Get statistics
    print("6. Node statistics...")
    
    stats = language_node.get_stats()
    for key, value in stats.items():
        if key != "memory" and not isinstance(value, dict):
            print(f"   {key}: {value}")
    
    memory_stats = stats["memory"]
    print(f"   Memory patterns: {memory_stats['pattern_count']}")
    print(f"   Memory usage: {memory_stats['memory_usage_mb']:.2f} MB\n")
    
    # 7. Serialization demonstration
    print("7. Serialization test...")
    
    # Serialize node
    node_data = language_node.to_dict()
    print(f"   Serialized data keys: {list(node_data.keys())}")
    
    # Create new node from serialized data
    restored_node = NodeEntity.from_dict(node_data)
    print(f"   Restored node: {restored_node}")
    
    # Compare statistics
    original_stats = language_node.get_stats()
    restored_stats = restored_node.get_stats()
    
    print(f"   Original activations: {original_stats['activation_count']}")
    print(f"   Restored activations: {restored_stats['activation_count']}")
    print(f"   Memory patterns match: "
          f"{original_stats['memory']['pattern_count'] == restored_stats['memory']['pattern_count']}")
    
    print("\n=== Demonstration Complete ===")

def benchmark_performance():
    """Benchmark NodeEntity performance"""
    
    print("\n=== Performance Benchmark ===\n")
    
    import time
    
    # Create a test node
    test_node = NodeEntity(
        name="benchmark_node",
        input_dim=768,
        output_dim=768
    )
    
    # Test forward pass speed
    test_input = np.random.randn(768).astype(np.float32)
    
    start_time = time.time()
    iterations = 100
    
    for i in range(iterations):
        test_node.forward(test_input, use_memory=False)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations * 1000
    
    print(f"Forward pass (no memory): {avg_time:.2f} ms per iteration")
    
    # Test learning speed
    start_time = time.time()
    
    for i in range(iterations):
        test_node.learn_from_experience(
            input_vector=test_input,
            target_output=np.random.randn(768),
            reward=0.5
        )
    
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations * 1000
    
    print(f"Learning step: {avg_time:.2f} ms per iteration")
    
    # Memory operations
    start_time = time.time()
    
    for i in range(100):
        pattern = np.random.randn(768)
        test_node.add_to_memory(pattern, f"benchmark_{i}")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000
    
    print(f"Memory storage: {avg_time:.2f} ms per pattern")
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    demonstrate_node_entity()
    benchmark_performance()