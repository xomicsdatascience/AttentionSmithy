import torch
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingTorch

class RotaryEmbedding:
    def __init__(self, dim):
        self.rotary = RotaryEmbeddingTorch(dim=dim)

    def __call__(self, matrix, *args, **kwargs):
        return self.rotary.to(matrix.device).rotate_queries_or_keys(matrix)
