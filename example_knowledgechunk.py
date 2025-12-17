"""
Example usage of KnowledgeChunk
"""

import numpy as np
from datetime import datetime, timedelta
from agi.knowledgechunk import (
    KnowledgeChunk, 
    CompressionMethod, 
    KnowledgeStatus,
    SemanticMetadata
)
from agi.nodeentity import NodeEntity


def demonstrate_knowledge_chunk():
    """Demonstrate KnowledgeChunk capabilities"""
    
    print("=== KnowledgeChunk Demonstration ===\n")
    
    # 1. Create a knowledge source node
    print("1. Creating knowledge source node...")
    
    physics_node = NodeEntity(
        name="physics_expert",
        specialization="theoretical_physics",
        input_dim=768,
        output_dim=768,
        use_cache=False
    )
    
    print(f"   Created node: {physics_node.name}\n")
    
    # 2. Create different types of knowledge chunks
    print("2. Creating knowledge chunks...")
    
    # Simple fact chunk
    fact_content = np.random.randn(768).astype(np.float32)
    fact_metadata = SemanticMetadata(
        domain="physics",
        subdomain="classical_mechanics",
        abstractness=0.2,  # Concrete fact
        certainty_level=0.95
    )
    
    fact_chunk = KnowledgeChunk(
        name="newtons_first_law",
        encoded_content=fact_content,
        source_node=physics_node.name,
        semantic_metadata=fact_metadata,
        confidence_score=0.95,
        creation_context={
            "derivation": "empirical_observation",
            "era": "17th_century",
            "discoverer": "Isaac Newton"
        },
        prerequisites=["basic_kinematics"],
        applications=["motion_prediction", "engineering_design"],
        utility_score=0.9,
        novelty_score=0.1,  # Well-known law
        complexity_score=0.3
    )
    
    # Abstract concept chunk
    concept_content = np.random.randn(1024).astype(np.float32) * 0.5
    concept_metadata = SemanticMetadata(
        domain="physics",
        subdomain="quantum_field_theory",
        abstractness=0.9,  # Very abstract
        certainty_level=0.7,
        temporal_validity="timeless",
        spatial_context="universal"
    )
    
    concept_chunk = KnowledgeChunk(
        name="quantum_superposition",
        encoded_content=concept_content,
        source_node=physics_node.name,
        semantic_metadata=concept_metadata,
        confidence_score=0.8,
        creation_context={
            "derivation": "mathematical_formalism",
            "era": "20th_century",
            "key_figures": ["Schrödinger", "Heisenberg"]
        },
        prerequisites=["quantum_mechanics_basics", "linear_algebra"],
        applications=["quantum_computing", "cryptography"],
        utility_score=0.6,
        novelty_score=0.4,
        complexity_score=0.8
    )
    
    print(f"   Created fact chunk: {fact_chunk}")
    print(f"   Created concept chunk: {concept_chunk}\n")
    
    # 3. Demonstrate validation
    print("3. Knowledge validation...")
    
    print("   Before validation:")
    print(f"     Fact confidence: {fact_chunk.confidence_score:.3f}")
    print(f"     Fact status: {fact_chunk.knowledge_status}")
    print(f"     Validations: {fact_chunk.validation_count}")
    
    # Validate multiple times
    validators = ["experiment_validation", "peer_review", "practical_application"]
    for validator in validators:
        result = fact_chunk.validate(
            validation_source=validator,
            validation_confidence=0.9 + np.random.random() * 0.1,
            validation_notes=f"Validated by {validator}"
        )
    
    print("\n   After validation:")
    print(f"     Fact confidence: {fact_chunk.confidence_score:.3f}")
    print(f"     Fact status: {fact_chunk.knowledge_status}")
    print(f"     Validations: {fact_chunk.validation_count}\n")
    
    # 4. Demonstrate utility updates
    print("4. Utility tracking...")
    
    # Simulate usage in different contexts
    usage_contexts = [
        ("physics_education", 0.95),
        ("engineering_calculation", 0.85),
        ("scientific_debate", 0.7),
        ("ai_training", 0.8)
    ]
    
    for context, success_rate in usage_contexts:
        result = fact_chunk.update_utility(
            usage_context=context,
            success_rate=success_rate
        )
    
    print(f"   After {fact_chunk.access_count} usages:")
    print(f"     Utility score: {fact_chunk.utility_score:.3f}")
    print(f"     Novelty score: {fact_chunk.novelty_score:.3f}")
    print(f"     Applications: {fact_chunk.applications}\n")
    
    # 5. Demonstrate relations
    print("5. Knowledge relations...")
    
    # Add relations between chunks
    fact_chunk.add_relation("basic_kinematics", relation_type="prerequisite")
    fact_chunk.add_relation("quantum_superposition", relation_type="semantic")
    fact_chunk.add_relation("relativity_theory", relation_type="related")
    
    concept_chunk.add_relation("newtons_first_law", relation_type="contradiction")
    
    print(f"   Fact chunk relations:")
    print(f"     Prerequisites: {fact_chunk.prerequisites}")
    print(f"     Related: {fact_chunk.related_chunks}")
    
    print(f"\n   Concept chunk contradictions: {concept_chunk.contradictions}\n")
    
    # 6. Demonstrate similarity calculation
    print("6. Similarity analysis...")
    
    # Create another similar chunk
    similar_content = fact_content + np.random.normal(0, 0.1, 768)
    similar_chunk = KnowledgeChunk(
        name="similar_fact",
        encoded_content=similar_content,
        source_node=physics_node.name
    )
    
    # Calculate similarities
    similarity = fact_chunk.calculate_similarity(similar_chunk, method="cosine")
    similarity_diff = fact_chunk.calculate_similarity(concept_chunk, method="cosine")
    
    print(f"   Similarity between related facts: {similarity:.3f}")
    print(f"   Similarity between different concepts: {similarity_diff:.3f}\n")
    
    # 7. Demonstrate compression
    print("7. Knowledge compression...")
    
    print(f"   Before compression:")
    print(f"     Size: {fact_chunk.original_size} → {fact_chunk.encoded_size} bytes")
    print(f"     Method: {fact_chunk.compression_method}")
    
    # Apply compression
    result = fact_chunk.compress(CompressionMethod.QUANTIZED)
    
    print(f"\n   After quantization:")
    print(f"     Size: {result['old_size']} → {result['new_size']} bytes")
    print(f"     Compression ratio: {result['compression_ratio']:.2f}")
    print(f"     New method: {result['new_method']}")
    
    # Check content type changed
    print(f"     Content dtype: {fact_chunk.encoded_content.dtype}\n")
    
    # 8. Demonstrate quality assessment
    print("8. Quality assessment...")
    
    fact_quality = fact_chunk.get_quality_score()
    concept_quality = concept_chunk.get_quality_score()
    
    print(f"   Fact chunk quality: {fact_quality:.3f}")
    print(f"   Concept chunk quality: {concept_quality:.3f}")
    
    # Get detailed stats
    fact_stats = fact_chunk.get_stats()
    print(f"\n   Fact chunk statistics:")
    print(f"     Compression ratio: {fact_stats['compression']['compression_ratio']:.2f}")
    print(f"     Relations: {fact_stats['relations']}")
    print(f"     Access frequency: {fact_stats['temporal']['access_frequency']:.2f}/hour")
    
    # 9. Demonstrate serialization
    print("\n9. Serialization test...")
    
    # Serialize chunk
    chunk_data = fact_chunk.to_dict()
    print(f"   Serialized keys: {list(chunk_data.keys())}")
    
    # Deserialize
    restored_chunk = KnowledgeChunk.from_dict(chunk_data)
    print(f"   Restored chunk: {restored_chunk}")
    
    # Compare
    assert np.allclose(fact_chunk.encoded_content, restored_chunk.encoded_content)
    print(f"   Content matches: ✓")
    print(f"   Confidence matches: {fact_chunk.confidence_score == restored_chunk.confidence_score}")
    
    print("\n=== Demonstration Complete ===")


def demonstrate_knowledge_evolution():
    """Demonstrate knowledge evolution through merging"""
    
    print("\n=== Knowledge Evolution Demonstration ===\n")
    
    # Create a knowledge node
    research_node = NodeEntity(
        name="research_ai",
        specialization="knowledge_synthesis",
        use_cache=False
    )
    
    # Create two related but different knowledge chunks
    print("1. Creating initial knowledge chunks...")
    
    # Version 1: Initial understanding
    content_v1 = np.random.randn(512).astype(np.float32)
    chunk_v1 = KnowledgeChunk(
        name="theory_v1",
        encoded_content=content_v1,
        source_node=research_node.name,
        confidence_score=0.6,
        creation_context={"stage": "initial_hypothesis", "year": 2020}
    )
    
    # Version 2: Refined understanding
    content_v2 = content_v1 + np.random.normal(0, 0.3, 512)  # Some evolution
    chunk_v2 = KnowledgeChunk(
        name="theory_v2",
        encoded_content=content_v2,
        source_node=research_node.name,
        confidence_score=0.8,
        creation_context={"stage": "refined_theory", "year": 2023}
    )
    
    print(f"   Created: {chunk_v1}")
    print(f"   Created: {chunk_v2}")
    
    # 2. Validate both
    print("\n2. Validating knowledge...")
    chunk_v1.validate("experiment_2021", 0.7)
    chunk_v1.validate("peer_review_2021", 0.65)
    
    chunk_v2.validate("experiment_2023", 0.85)
    chunk_v2.validate("peer_review_2023", 0.82)
    chunk_v2.validate("practical_test_2023", 0.88)
    
    print(f"   V1 confidence: {chunk_v1.confidence_score:.3f}, validations: {chunk_v1.validation_count}")
    print(f"   V2 confidence: {chunk_v2.confidence_score:.3f}, validations: {chunk_v2.validation_count}")
    
    # 3. Merge knowledge
    print("\n3. Merging knowledge...")
    merged_chunk = chunk_v1.merge_with(chunk_v2, merge_strategy="weighted_average")
    
    print(f"   Merged chunk: {merged_chunk}")
    print(f"   Merge confidence: {merged_chunk.confidence_score:.3f}")
    print(f"   Merge version: {merged_chunk.version}")
    print(f"   Previous versions: {merged_chunk.previous_version}")
    
    # 4. Track utility evolution
    print("\n4. Utility evolution...")
    
    # Simulate usage over time
    contexts_v1 = [("research_2021", 0.6), ("education_2021", 0.7)]
    contexts_v2 = [("research_2023", 0.8), ("industry_2023", 0.85)]
    
    for context, success in contexts_v1:
        chunk_v1.update_utility(context, success)
    
    for context, success in contexts_v2:
        chunk_v2.update_utility(context, success)
        merged_chunk.update_utility(context, success)
    
    print(f"   V1 utility: {chunk_v1.utility_score:.3f}")
    print(f"   V2 utility: {chunk_v2.utility_score:.3f}")
    print(f"   Merged utility: {merged_chunk.utility_score:.3f}")
    
    # 5. Check obsolescence
    print("\n5. Obsolescence check...")
    
    # Make v1 old
    chunk_v1.creation_time = datetime.now() - timedelta(days=60)
    chunk_v1.utility_score = 0.1  # Low utility
    
    is_v1_obsolete = chunk_v1.is_obsolete(min_utility=0.2, max_age_days=30)
    is_v2_obsolete = chunk_v2.is_obsolete(min_utility=0.2, max_age_days=30)
    
    print(f"   V1 obsolete: {is_v1_obsolete} (age: 60 days, utility: {chunk_v1.utility_score:.3f})")
    print(f"   V2 obsolete: {is_v2_obsolete} (age: 0 days, utility: {chunk_v2.utility_score:.3f})")
    
    print("\n=== Evolution Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_knowledge_chunk()
    demonstrate_knowledge_evolution()