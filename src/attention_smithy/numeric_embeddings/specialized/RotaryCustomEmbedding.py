import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange
from typing import Optional
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import MatrixModificationStrategyBase

class RotaryCustomEmbedding(MatrixModificationStrategyBase):
    """
    Rotary Embedding wrapper that allows applying custom scalar position values
    (e.g., [0.4, 2.6, 10.1]) instead of standard integer positions (0, 1, 2, ...).

    Applies rotary embeddings to input tensors like queries or keys using those custom positions.
    """

    def __init__(self, head_dimension: int):
        """
        Args:
            dim (int): Embedding dimension per attention head (must be divisible by 2).
        """
        super().__init__()
        self.rotary = RotaryEmbedding(dim=head_dimension, cache_if_possible=False)

    def modify_matrix(self, target_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(target_matrix, **kwargs)

    def forward(
        self,
        target_matrix: torch.Tensor,
        rotary_custom_values: torch.Tensor,
        seq_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            target_matrix (torch.Tensor): The input Q or K matrix of shape (..., seq_len, dim).
            rotary_custom_values (torch.Tensor): Float tensor of shape (seq_len,) specifying custom position values.
            seq_dim (int, optional): The dimension in `tensor` corresponding to sequence length. Defaults to -2.

        Returns:
            torch.Tensor: Tensor with rotary embeddings applied.
        """
        seq_dim = seq_dim if seq_dim is not None else -2
        seq_len = target_matrix.shape[seq_dim]

        if rotary_custom_values.shape[0] != seq_len:
            raise ValueError(f"rotary_custom_values length {rotary_custom_values.shape[0]} must match sequence length {seq_len}")

        freqs = self.rotary.forward(rotary_custom_values, seq_len=seq_len)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, target_matrix, seq_dim=seq_dim)