"""
AGI Knowledge Chunk - Atomic Knowledge Unit
Represents a compressed, encoded piece of knowledge with semantic metadata,
relational links, and pragmatic metrics for efficient storage and diffusion.
"""

from typing import Dict, Any, Optional, List, Union, Set
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import zlib
from common.base.baseentity import BaseEntity
from common.utils.logging_setup import logger


class CompressionMethod(Enum):
    """Methods for knowledge compression"""
    NONE = "none"           # No compression
    QUANTIZED = "quantized" # Quantization (8-bit, 16-bit)
    AUTOENCODER = "ae"      # Autoencoder compression
    HASH = "hash"           # Cryptographic hash (for identity)
    DIM_REDUCTION = "pca"   # Dimensionality reduction


class KnowledgeStatus(Enum):
    """Status of knowledge validation"""
    RAW = "raw"             # New, unvalidated knowledge
    VALIDATED = "validated" # Multiple validations passed
    CONFLICTING = "conflicting" # Conflicting with other knowledge
    DEPRECATED = "deprecated"   # Outdated knowledge


@dataclass
class SemanticMetadata:
    """Semantic metadata for knowledge chunk"""
    domain: str                     # Knowledge domain (e.g., "physics", "language")
    subdomain: str                  # Subdomain (e.g., "quantum_mechanics", "syntax")
    abstractness: float            # 0.0=concrete, 1.0=abstract
    certainty_level: float         # 0.0=speculative, 1.0=certain
    temporal_validity: Optional[str] = None  # "past", "present", "future", "timeless"
    spatial_context: Optional[str] = None    # Geographic/cultural context


class KnowledgeChunk(BaseEntity):
    """
    AGI Knowledge Chunk - atomic unit of knowledge.
    
    Represents a compressed, encoded piece of knowledge with:
    1. Encoded content vector
    2. Semantic metadata and source information
    3. Relational links to other knowledge
    4. Pragmatic usage metrics
    
    Attributes:
        # Core content
        encoded_content (np.ndarray): Vector representation (768-1024 dim)
        compression_method (str): Method used for compression
        original_size (int): Size before compression (bytes)
        encoded_size (int): Size after compression (bytes)
        
        # Semantic metadata
        source_node (str): ID of node that created this chunk
        creation_context (Dict[str, Any]): Conditions of creation
        confidence_score (float): Source confidence (0.0-1.0)
        validation_count (int): How many times validated
        knowledge_status (str): Status of knowledge (raw/validated/etc)
        semantic_metadata (SemanticMetadata): Semantic classification
        
        # Relational links
        prerequisites (List[str]): Required knowledge chunks (by ID)
        applications (List[str]): Where this knowledge can be applied
        related_chunks (List[str]): Semantically similar chunks
        contradictions (List[str]): Conflicting knowledge chunks
        
        # Pragmatic metrics
        utility_score (float): Frequency of use (0.0-1.0)
        novelty_score (float): Novelty/uniqueness (0.0-1.0)
        complexity_score (float): Computational complexity (0.0-1.0)
        coherence_score (float): Internal consistency (0.0-1.0)
        
        # Temporal tracking
        creation_time (datetime): When created
        last_accessed (datetime): Last time used
        last_updated (datetime): Last modification
        access_count (int): Total accesses
        
        # Versioning
        version (int): Version number
        previous_version (Optional[str]): ID of previous version
    """
    
    # Core content
    encoded_content: np.ndarray
    compression_method: str
    original_size: int
    encoded_size: int
    
    # Semantic metadata
    source_node: str
    creation_context: Dict[str, Any]
    confidence_score: float
    validation_count: int
    knowledge_status: str
    semantic_metadata: SemanticMetadata
    
    # Relational links
    prerequisites: List[str]
    applications: List[str]
    related_chunks: List[str]
    contradictions: List[str]
    
    # Pragmatic metrics
    utility_score: float
    novelty_score: float
    complexity_score: float
    coherence_score: float
    
    # Temporal tracking
    creation_time: datetime
    last_accessed: datetime
    last_updated: datetime
    access_count: int
    
    # Versioning
    version: int
    previous_version: Optional[str]
    
    def __init__(self,
                 name: str,
                 encoded_content: np.ndarray,
                 source_node: str,
                 compression_method: Union[str, CompressionMethod] = CompressionMethod.NONE,
                 confidence_score: float = 0.5,
                 creation_context: Optional[Dict[str, Any]] = None,
                 semantic_metadata: Optional[SemanticMetadata] = None,
                 original_size: Optional[int] = None,
                 prerequisites: Optional[List[str]] = None,
                 applications: Optional[List[str]] = None,
                 utility_score: float = 0.1,
                 novelty_score: float = 0.9,
                 complexity_score: float = 0.5,
                 isactive: bool = True,
                 use_cache: bool = False,
                 **kwargs):
        """
        Initialize a KnowledgeChunk.
        
        Args:
            name: Unique name/ID for the chunk
            encoded_content: Vector representation of knowledge
            source_node: ID of creating node
            compression_method: Compression method used
            confidence_score: Initial confidence (0.0-1.0)
            creation_context: Context of creation
            semantic_metadata: Semantic classification
            original_size: Size before compression (bytes)
            prerequisites: Required knowledge chunks
            applications: Where knowledge can be applied
            utility_score: Initial utility score (0.0-1.0)
            novelty_score: Initial novelty score (0.0-1.0)
            complexity_score: Initial complexity (0.0-1.0)
            isactive: Whether chunk is active
            use_cache: Enable caching for to_dict()
            **kwargs: Additional BaseEntity parameters
        """
        
        # Convert enum to string if needed
        if isinstance(compression_method, CompressionMethod):
            compression_method = compression_method.value
        
        # Set original size if not provided
        if original_size is None:
            original_size = encoded_content.nbytes
        
        # Calculate encoded size
        encoded_size = encoded_content.nbytes
        
        # Set default semantic metadata
        if semantic_metadata is None:
            semantic_metadata = SemanticMetadata(
                domain="general",
                subdomain="unspecified",
                abstractness=0.5,
                certainty_level=confidence_score
            )
        
        # Set default creation context
        if creation_context is None:
            creation_context = {
                "mode": "automatic",
                "trigger": "learning",
                "environment": "default"
            }
        
        # Get current time
        current_time = datetime.now()
        
        # Initialize BaseEntity first
        super().__init__(
            name=name,
            isactive=isactive,
            use_cache=use_cache,
            **kwargs
        )
        
        # Set core content
        self.encoded_content = encoded_content.copy()
        self.compression_method = compression_method
        self.original_size = original_size
        self.encoded_size = encoded_size
        
        # Semantic metadata
        self.source_node = source_node
        self.creation_context = creation_context
        self.confidence_score = max(0.0, min(1.0, confidence_score))
        self.validation_count = 0
        self.knowledge_status = KnowledgeStatus.RAW.value
        self.semantic_metadata = semantic_metadata
        
        # Relational links
        self.prerequisites = prerequisites or []
        self.applications = applications or []
        self.related_chunks = []
        self.contradictions = []
        
        # Pragmatic metrics
        self.utility_score = max(0.0, min(1.0, utility_score))
        self.novelty_score = max(0.0, min(1.0, novelty_score))
        self.complexity_score = max(0.0, min(1.0, complexity_score))
        self.coherence_score = 0.7  # Initial coherence
        
        # Temporal tracking
        self.creation_time = current_time
        self.last_accessed = current_time
        self.last_updated = current_time
        self.access_count = 0
        
        # Versioning
        self.version = 1
        self.previous_version = None
        
        # Calculate initial coherence based on confidence and complexity
        self._update_coherence_score()
        
        logger.info(f"Created KnowledgeChunk '{name}' from node '{source_node}' "
                   f"with confidence {confidence_score:.3f}, size={encoded_size} bytes")
    
    def get_content_hash(self) -> str:
        """
        Generate a unique hash for the content.
        
        Returns:
            SHA-256 hash string
        """
        # Convert content to bytes
        if self.compression_method == CompressionMethod.QUANTIZED.value:
            # For quantized arrays, use the raw bytes
            content_bytes = self.encoded_content.tobytes()
        else:
            # For float arrays, convert to string representation
            content_str = np.array2string(self.encoded_content, precision=6, separator=',', 
                                         suppress_small=True)
            content_bytes = content_str.encode('utf-8')
        
        # Add metadata to hash
        metadata = f"{self.source_node}{self.version}{self.creation_time}"
        full_bytes = content_bytes + metadata.encode('utf-8')
        
        return hashlib.sha256(full_bytes).hexdigest()[:16]  # 16 chars for brevity
    
    def calculate_similarity(self, 
                           other_chunk: 'KnowledgeChunk',
                           method: str = "cosine") -> float:
        """
        Calculate similarity between this chunk and another.
        
        Args:
            other_chunk: Another KnowledgeChunk
            method: Similarity method ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score (0.0-1.0)
        """
        vec1 = self.encoded_content.flatten()
        vec2 = other_chunk.encoded_content.flatten()
        
        # Ensure vectors have same length
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
        
        if method == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
            
        elif method == "euclidean":
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(vec1 - vec2)
            max_distance = np.linalg.norm(vec1) + np.linalg.norm(vec2)
            if max_distance == 0:
                return 1.0
            return 1.0 - (distance / max_distance)
            
        elif method == "dot":
            # Dot product normalized
            dot_product = np.dot(vec1, vec2)
            max_dot = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if max_dot == 0:
                return 0.0
            return dot_product / max_dot
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def validate(self, 
                validation_source: str,
                validation_confidence: float,
                validation_notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate this knowledge chunk.
        
        Args:
            validation_source: ID of validating node
            validation_confidence: Confidence of validation (0.0-1.0)
            validation_notes: Optional notes about validation
            
        Returns:
            Dictionary with validation results
        """
        old_confidence = self.confidence_score
        old_status = self.knowledge_status
        
        # Update confidence (weighted average)
        total_validations = self.validation_count + 1
        self.confidence_score = (
            (self.confidence_score * self.validation_count + validation_confidence)
            / total_validations
        )
        
        self.validation_count += 1
        
        # Update status based on validation count and confidence
        if self.validation_count >= 3 and self.confidence_score > 0.7:
            self.knowledge_status = KnowledgeStatus.VALIDATED.value
        elif self.validation_count >= 2 and self.confidence_score < 0.3:
            self.knowledge_status = KnowledgeStatus.CONFLICTING.value
        elif self.validation_count >= 1 and self.confidence_score > 0.5:
            self.knowledge_status = KnowledgeStatus.VALIDATED.value
        
        self.last_updated = datetime.now()
        
        logger.debug(f"KnowledgeChunk '{self.name}' validated by '{validation_source}', "
                    f"confidence: {old_confidence:.3f}->{self.confidence_score:.3f}")
        
        return {
            "chunk_id": self.name,
            "old_confidence": old_confidence,
            "new_confidence": self.confidence_score,
            "old_status": old_status,
            "new_status": self.knowledge_status,
            "validation_count": self.validation_count,
            "validator": validation_source,
            "validation_notes": validation_notes
        }
    
    def update_utility(self, 
                      usage_context: str,
                      success_rate: float = 1.0) -> Dict[str, Any]:
        """
        Update utility metrics based on usage.
        
        Args:
            usage_context: Context in which chunk was used
            success_rate: How successful was the usage (0.0-1.0)
            
        Returns:
            Dictionary with updated metrics
        """
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Update utility score (exponential moving average)
        utility_decay = 0.9
        self.utility_score = (
            utility_decay * self.utility_score + 
            (1 - utility_decay) * success_rate
        )
        
        # Update novelty score (decays with usage)
        novelty_decay = 0.95
        self.novelty_score *= novelty_decay
        
        # Update coherence based on successful usage
        if success_rate > 0.7:
            coherence_gain = 0.05
            self.coherence_score = min(1.0, self.coherence_score + coherence_gain)
        
        # Add to applications if new context
        if usage_context not in self.applications:
            self.applications.append(usage_context)
        
        logger.debug(f"KnowledgeChunk '{self.name}' utility updated: "
                    f"utility={self.utility_score:.3f}, "
                    f"novelty={self.novelty_score:.3f}, "
                    f"accesses={self.access_count}")
        
        return {
            "chunk_id": self.name,
            "utility_score": self.utility_score,
            "novelty_score": self.novelty_score,
            "access_count": self.access_count,
            "applications": self.applications
        }
    
    def add_relation(self, 
                    related_chunk_id: str,
                    relation_type: str = "semantic",
                    similarity_score: Optional[float] = None) -> None:
        """
        Add relation to another knowledge chunk.
        
        Args:
            related_chunk_id: ID of related chunk
            relation_type: Type of relation ("semantic", "prerequisite", "application")
            similarity_score: Optional similarity score
        """
        if related_chunk_id == self.name:
            logger.warning(f"Cannot relate chunk to itself: {self.name}")
            return
        
        if relation_type == "prerequisite":
            if related_chunk_id not in self.prerequisites:
                self.prerequisites.append(related_chunk_id)
        elif relation_type == "application":
            if related_chunk_id not in self.applications:
                self.applications.append(related_chunk_id)
        elif relation_type == "semantic":
            if related_chunk_id not in self.related_chunks:
                self.related_chunks.append(related_chunk_id)
        elif relation_type == "contradiction":
            if related_chunk_id not in self.contradictions:
                self.contradictions.append(related_chunk_id)
        
        self.last_updated = datetime.now()
        logger.debug(f"Added {relation_type} relation from '{self.name}' to '{related_chunk_id}'")
    
    def merge_with(self, 
                  other_chunk: 'KnowledgeChunk',
                  merge_strategy: str = "weighted_average") -> 'KnowledgeChunk':
        """
        Merge this chunk with another chunk.
        
        Args:
            other_chunk: Another KnowledgeChunk to merge with
            merge_strategy: How to merge ("weighted_average", "union", "intersection")
            
        Returns:
            New merged KnowledgeChunk
        """
        if merge_strategy == "weighted_average":
            # Weighted average of content based on confidence
            weight1 = self.confidence_score
            weight2 = other_chunk.confidence_score
            total_weight = weight1 + weight2
            
            if total_weight == 0:
                total_weight = 1
            
            # Average the vectors
            new_content = (
                (self.encoded_content * weight1 + other_chunk.encoded_content * weight2)
                / total_weight
            )
            
        elif merge_strategy == "union":
            # Take the more confident vector
            if self.confidence_score >= other_chunk.confidence_score:
                new_content = self.encoded_content.copy()
            else:
                new_content = other_chunk.encoded_content.copy()
                
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Create new merged chunk
        merged_name = f"merged_{self.get_content_hash()[:8]}_{other_chunk.get_content_hash()[:8]}"
        
        # Use higher confidence score
        new_confidence = max(self.confidence_score, other_chunk.confidence_score)
        
        # Combine metadata
        combined_context = {
            **self.creation_context,
            **other_chunk.creation_context,
            "merged_from": [self.name, other_chunk.name],
            "merge_strategy": merge_strategy
        }
        
        # Create new semantic metadata
        new_semantic = SemanticMetadata(
            domain=f"{self.semantic_metadata.domain}/{other_chunk.semantic_metadata.domain}",
            subdomain="merged",
            abstractness=(self.semantic_metadata.abstractness + 
                         other_chunk.semantic_metadata.abstractness) / 2,
            certainty_level=new_confidence
        )
        
        # Create merged chunk
        merged_chunk = KnowledgeChunk(
            name=merged_name,
            encoded_content=new_content,
            source_node=f"merge_of_{self.source_node}_{other_chunk.source_node}",
            compression_method=self.compression_method,
            confidence_score=new_confidence,
            creation_context=combined_context,
            semantic_metadata=new_semantic,
            original_size=self.original_size + other_chunk.original_size,
            prerequisites=list(set(self.prerequisites + other_chunk.prerequisites)),
            applications=list(set(self.applications + other_chunk.applications)),
            utility_score=(self.utility_score + other_chunk.utility_score) / 2,
            novelty_score=(self.novelty_score + other_chunk.novelty_score) / 2,
            complexity_score=(self.complexity_score + other_chunk.complexity_score) / 2
        )
        
        # Update versioning
        merged_chunk.version = max(self.version, other_chunk.version) + 1
        merged_chunk.previous_version = f"{self.name},{other_chunk.name}"
        
        logger.info(f"Merged '{self.name}' with '{other_chunk.name}' into '{merged_name}'")
        return merged_chunk
    
    def compress(self, 
                method: Union[str, CompressionMethod],
                target_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Compress the knowledge chunk.
        
        Args:
            method: Compression method to use
            target_size: Target size in bytes (optional)
            
        Returns:
            Dictionary with compression results
        """
        if isinstance(method, CompressionMethod):
            method = method.value
        
        old_size = self.encoded_size
        old_method = self.compression_method
        
        if method == CompressionMethod.QUANTIZED.value:
            # 8-bit quantization
            min_val = np.min(self.encoded_content)
            max_val = np.max(self.encoded_content)
            
            if max_val > min_val:
                # Normalize to [0, 255]
                normalized = (self.encoded_content - min_val) / (max_val - min_val)
                self.encoded_content = (normalized * 255).astype(np.uint8)
            else:
                self.encoded_content = np.zeros_like(self.encoded_content, dtype=np.uint8)
            
            # Store quantization params in context
            self.creation_context["quantization_params"] = {
                "min": float(min_val),
                "max": float(max_val),
                "dtype": "uint8"
            }
            
        elif method == CompressionMethod.HASH.value:
            # Hash-based compression (only store hash)
            content_hash = self.get_content_hash()
            self.encoded_content = np.array([ord(c) for c in content_hash[:64]], 
                                          dtype=np.uint8)
            self.creation_context["hash_representation"] = True
        
        # Update compression info
        self.compression_method = method
        self.encoded_size = self.encoded_content.nbytes
        
        compression_ratio = old_size / self.encoded_size if self.encoded_size > 0 else 1.0
        
        logger.debug(f"Compressed '{self.name}' with {method}: "
                    f"{old_size}->{self.encoded_size} bytes, "
                    f"ratio={compression_ratio:.2f}")
        
        return {
            "chunk_id": self.name,
            "old_method": old_method,
            "new_method": method,
            "old_size": old_size,
            "new_size": self.encoded_size,
            "compression_ratio": compression_ratio,
            "original_size": self.original_size
        }
    
    def _update_coherence_score(self) -> None:
        """Update coherence score based on internal consistency"""
        
        factors = []
        
        # Confidence factor
        factors.append(self.confidence_score)
        
        # Validation factor
        validation_factor = min(1.0, self.validation_count / 5.0)
        factors.append(validation_factor)
        
        # Consistency factor (prerequisites vs contradictions)
        if self.prerequisites and not self.contradictions:
            consistency_factor = 0.8
        elif self.contradictions and not self.prerequisites:
            consistency_factor = 0.4
        else:
            consistency_factor = 0.6
        factors.append(consistency_factor)
        
        # Complexity factor (moderate complexity is best)
        if 0.3 <= self.complexity_score <= 0.7:
            complexity_factor = 0.9
        elif self.complexity_score < 0.1 or self.complexity_score > 0.9:
            complexity_factor = 0.5
        else:
            complexity_factor = 0.7
        factors.append(complexity_factor)
        
        # Calculate weighted average
        weights = [0.4, 0.3, 0.2, 0.1]  # Confidence is most important
        self.coherence_score = sum(f * w for f, w in zip(factors, weights)) / sum(weights)
    
    def get_quality_score(self) -> float:
        """
        Calculate overall quality score.
        
        Returns:
            Quality score (0.0-1.0)
        """
        factors = {
            "confidence": self.confidence_score * 0.3,
            "utility": self.utility_score * 0.25,
            "coherence": self.coherence_score * 0.2,
            "validation": min(1.0, self.validation_count / 10.0) * 0.15,
            "novelty": self.novelty_score * 0.1
        }
        
        quality = sum(factors.values())
        
        # Penalize conflicting knowledge
        if self.knowledge_status == KnowledgeStatus.CONFLICTING.value:
            quality *= 0.5
        
        return min(1.0, quality)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive chunk statistics"""
        
        # Calculate age in hours
        age_hours = (datetime.now() - self.creation_time).total_seconds() / 3600
        
        # Calculate access frequency (accesses per hour)
        if age_hours > 0:
            access_frequency = self.access_count / age_hours
        else:
            access_frequency = 0.0
        
        return {
            "name": self.name,
            "source_node": self.source_node,
            "isactive": self.isactive,
            "knowledge_status": self.knowledge_status,
            "confidence_score": self.confidence_score,
            "validation_count": self.validation_count,
            "quality_score": self.get_quality_score(),
            "compression": {
                "method": self.compression_method,
                "original_size": self.original_size,
                "encoded_size": self.encoded_size,
                "compression_ratio": self.original_size / max(1, self.encoded_size)
            },
            "semantic": {
                "domain": self.semantic_metadata.domain,
                "subdomain": self.semantic_metadata.subdomain,
                "abstractness": self.semantic_metadata.abstractness,
                "certainty": self.semantic_metadata.certainty_level
            },
            "relations": {
                "prerequisites": len(self.prerequisites),
                "applications": len(self.applications),
                "related_chunks": len(self.related_chunks),
                "contradictions": len(self.contradictions)
            },
            "metrics": {
                "utility_score": self.utility_score,
                "novelty_score": self.novelty_score,
                "complexity_score": self.complexity_score,
                "coherence_score": self.coherence_score
            },
            "temporal": {
                "age_hours": age_hours,
                "access_count": self.access_count,
                "access_frequency": access_frequency,
                "last_accessed": self.last_accessed,
                "last_updated": self.last_updated
            },
            "versioning": {
                "version": self.version,
                "previous_version": self.previous_version
            }
        }
    
    def get_compression_efficiency(self) -> float:
        """Get compression efficiency ratio"""
        if self.encoded_size == 0:
            return 1.0
        return self.original_size / self.encoded_size
    
    def is_obsolete(self, 
                   min_utility: float = 0.1,
                   max_age_days: float = 30.0) -> bool:
        """
        Check if chunk is obsolete.
        
        Args:
            min_utility: Minimum utility threshold
            max_age_days: Maximum age in days
            
        Returns:
            True if chunk is obsolete
        """
        age_days = (datetime.now() - self.creation_time).total_seconds() / (24 * 3600)
        
        if self.utility_score < min_utility and age_days > max_age_days:
            return True
        
        if self.knowledge_status == KnowledgeStatus.DEPRECATED.value:
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization"""
        data = super().to_dict()
        
        # Convert numpy array to list
        if "encoded_content" in data and isinstance(data["encoded_content"], np.ndarray):
            data["encoded_content"] = data["encoded_content"].tolist()
        
        # Convert datetime fields to ISO strings
        datetime_fields = ["creation_time", "last_accessed", "last_updated"]
        for field in datetime_fields:
            if field in data and isinstance(data[field], datetime):
                data[field] = data[field].isoformat()
        
        # Convert SemanticMetadata dataclass to dict
        if "semantic_metadata" in data:
            data["semantic_metadata"] = {
                "domain": self.semantic_metadata.domain,
                "subdomain": self.semantic_metadata.subdomain,
                "abstractness": self.semantic_metadata.abstractness,
                "certainty_level": self.semantic_metadata.certainty_level,
                "temporal_validity": self.semantic_metadata.temporal_validity,
                "spatial_context": self.semantic_metadata.spatial_context
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeChunk':
        """Create chunk from dictionary (deserialization)"""
        
        # Extract and convert numpy array
        encoded_content = np.array(data["encoded_content"])
        
        # Extract semantic metadata
        semantic_data = data.get("semantic_metadata", {})
        semantic_metadata = SemanticMetadata(
            domain=semantic_data.get("domain", "general"),
            subdomain=semantic_data.get("subdomain", "unspecified"),
            abstractness=semantic_data.get("abstractness", 0.5),
            certainty_level=semantic_data.get("certainty_level", 0.5),
            temporal_validity=semantic_data.get("temporal_validity"),
            spatial_context=semantic_data.get("spatial_context")
        )
        
        # Handle datetime fields
        datetime_fields = ["creation_time", "last_accessed", "last_updated"]
        for field in datetime_fields:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Create chunk instance
        chunk = cls(
            name=data["name"],
            encoded_content=encoded_content,
            source_node=data["source_node"],
            compression_method=data.get("compression_method", CompressionMethod.NONE.value),
            confidence_score=data.get("confidence_score", 0.5),
            creation_context=data.get("creation_context", {}),
            semantic_metadata=semantic_metadata,
            original_size=data.get("original_size", encoded_content.nbytes),
            prerequisites=data.get("prerequisites", []),
            applications=data.get("applications", []),
            utility_score=data.get("utility_score", 0.1),
            novelty_score=data.get("novelty_score", 0.9),
            complexity_score=data.get("complexity_score", 0.5),
            isactive=data.get("isactive", True),
            use_cache=data.get("_use_cache", False)
        )
        
        # Set additional fields
        chunk.validation_count = data.get("validation_count", 0)
        chunk.knowledge_status = data.get("knowledge_status", KnowledgeStatus.RAW.value)
        chunk.related_chunks = data.get("related_chunks", [])
        chunk.contradictions = data.get("contradictions", [])
        chunk.coherence_score = data.get("coherence_score", 0.7)
        chunk.access_count = data.get("access_count", 0)
        chunk.version = data.get("version", 1)
        chunk.previous_version = data.get("previous_version")
        
        # Set datetime fields
        chunk.creation_time = data.get("creation_time", datetime.now())
        chunk.last_accessed = data.get("last_accessed", datetime.now())
        chunk.last_updated = data.get("last_updated", datetime.now())
        
        logger.info(f"Restored KnowledgeChunk '{chunk.name}' from dict")
        return chunk
    
    def __repr__(self) -> str:
        """String representation of KnowledgeChunk"""
        stats = self.get_stats()
        quality = stats["quality_score"]
        
        return (
            f"KnowledgeChunk(name='{self.name}', "
            f"source='{self.source_node}', "
            f"quality={quality:.3f}, "
            f"confidence={self.confidence_score:.3f}, "
            f"utility={self.utility_score:.3f}, "
            f"size={self.encoded_size}/{self.original_size} bytes)"
        )