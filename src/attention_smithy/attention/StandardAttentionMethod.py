import torch
import torch.nn as nn
from torch.nn import functional as F


class StandardAttentionMethod(nn.Module):
    """
    A PyTorch module implementing the attention mechanism described in the "Attention Is All You Need" paper.
        The query, key and value matrices have already been multiplied by their respective weights before passing
        through this module.
    """

    def __init__(self, dropout: float = 0.0):
        """
        Initializes the standard attention module.

        Args:
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """

        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        k,
        q,
        v,
        **kwargs,
    ):
        """
        Forward pass of the Big Bird attention module.

        Args:
            q (torch.Tensor): The query tensor embedding, of shape (batch_size, num_heads, query_length, head_dim)
                Note that the embedding dimension can be calculated by multiplying (num_heads * head_dim)
            k (torch.Tensor): The key tensor embedding, of shape (batch_size, num_heads, kv_length, head_dim)
            v (torch.Tensor): The value tensor embedding, of shape (batch_size, num_heads, kv_length, head_dim)
        Returns:
            torch.Tensor: The output tensor, of shape (batch_size, num_heads, query_length, head_dim)
        """
        attn_scores = torch.einsum("bnqd, bnkd -> bnqk", q, k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        attn_output = torch.einsum("bnqk, bnkd -> bnqd", attn_probs, v)
        return attn_output, attn_probs
