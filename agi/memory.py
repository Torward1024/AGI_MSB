# agi/memory.py
import torch
import numpy as np
from common.utils.logging_setup import logger

class SimpleVectorMemory:
    """Simple in-memory vector store using Torch for similarity search.

    Replaces FAISS with a list-based storage and cdist computation.
    Supports add (embeddings) and search (kNN with L2 distance).
    """
    def __init__(self, dim: int):
        self.vectors: list[torch.Tensor] = []
        self.dim = dim
        logger.debug(f"SimpleVectorMemory initialized with dim={dim}")

    def add(self, vec: np.ndarray | torch.Tensor) -> None:
        """Add a vector to memory."""
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec.astype('float32'))
        else:
            vec = vec.float()
        self.vectors.append(vec.view(-1))
        logger.debug(f"Added vector; total={len(self.vectors)}")

    def search(self, query: np.ndarray | torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors using L2 distance."""
        if not self.vectors:
            logger.warning("Memory empty; returning zeros")
            return np.zeros((k, self.dim), dtype='float32'), np.full(k, np.inf, dtype='float32')
        
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query.astype('float32'))
        else:
            query = query.float()
        query = query.view(-1)
        
        stack = torch.stack(self.vectors)
        dists = torch.cdist(query.unsqueeze(0), stack)[0]
        indices = torch.argsort(dists)[:k]
        
        nearest_vectors = stack[indices].numpy()
        nearest_dists = dists[indices].numpy()
        logger.debug(f"Searched k={k}; found {len(indices)}")
        return nearest_vectors, nearest_dists

    def get_all(self) -> list[torch.Tensor]:
        """Return all stored vectors."""
        return self.vectors[:]

    def set_all(self, vectors: list[torch.Tensor]) -> None:
        """Set memory from list of vectors."""
        self.vectors = [v.view(-1).float() for v in vectors]
        logger.debug(f"Set {len(self.vectors)} vectors")

    def clear(self) -> None:
        """Clear memory."""
        self.vectors = []
        logger.debug("Memory cleared")