# utils/text_processor.py
import re
import torch
from utils.logging_setup import logger

class TextProcessor:
    """Простой процессор текста для embedding и генерации."""
    VOCAB = {}
    VOCAB_SIZE = 100  # Будет переопределяться из конфига

    @staticmethod
    def set_vocab_size(size: int):
        TextProcessor.VOCAB_SIZE = size

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    @staticmethod
    def embed_text(text: str, size: int = 10) -> torch.Tensor:
        tokens = TextProcessor.tokenize(text)
        embedding = torch.zeros(size)
        for token in tokens:
            if token not in TextProcessor.VOCAB:
                if len(TextProcessor.VOCAB) < TextProcessor.VOCAB_SIZE:
                    TextProcessor.VOCAB[token] = len(TextProcessor.VOCAB)
                else:
                    # Если словарь полон, игнорируем новые слова
                    continue
            idx = TextProcessor.VOCAB[token]
            embedding[idx % size] += 1
        return embedding

    @staticmethod
    def generate_text(logits: torch.Tensor, top_n: int = 5) -> str:
        """Генерация текста с softmax и топ-N выбором."""
        probs = torch.softmax(logits, dim=0)
        logger.debug(f"generate_text static: probs size = {probs.size()}")
        if probs.size(0) < top_n:
            logger.warning(f"probs size {probs.size(0)} < top_n {top_n}, adjusting top_n to {probs.size(0)}")
            top_n = probs.size(0)
        top_probs, top_indices = torch.topk(probs, top_n)
        words = []
        for idx in top_indices:
            idx = idx.item()
            for word, v_idx in TextProcessor.VOCAB.items():
                if v_idx == idx:
                    words.append(word)
                    break
        return ' '.join(words)