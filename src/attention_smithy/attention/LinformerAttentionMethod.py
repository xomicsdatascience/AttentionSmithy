import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinformerAttentionMethod(nn.Module):
    """
    Linformer-style attention, compatible with the MultiheadAttention interface.
    """

    def __init__(self, embedding_dim, sequence_length, k=256, dropout=0.0, share_kv=False):
        """
        Args:
            embedding_dim (int): Total embedding dimension (num_heads * head_dim).
            sequence_length (int): Maximum sequence length.
            k (int): Projected key/value dimension.
            dropout (float): Dropout rate.
            share_kv (bool): Whether to share key/value projections.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.k = k
        self.share_kv = share_kv

        # Projection matrices: (seq_len, k)
        self.proj_k = nn.Parameter(self._init_proj(sequence_length, k))
        if not share_kv:
            self.proj_v = nn.Parameter(self._init_proj(sequence_length, k))

        self.dropout_layer = nn.Dropout(dropout)

    def _init_proj(self, seq_len, k):
        tensor = torch.zeros(seq_len, k)
        std = 1 / math.sqrt(k)
        return tensor.uniform_(-std, std)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        numeric_embedding_manager,
        padding_and_loss_attention_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        Args:
            q, k, v: (batch, heads, seq_len, head_dim)
            padding_and_loss_attention_mask: (batch, kv_len), 1 = keep, 0 = pad
        Returns:
            attention_outputs: (batch, heads, query_len, head_dim)
            attention_probabilities: (batch, heads, query_len, kv_len) [expanded approx.]
        """
        b, h, q_len, d_h = q.shape
        _, _, kv_len, _ = k.shape
        assert kv_len <= self.sequence_length, f"kv_len={kv_len} exceeds max seq length {self.sequence_length}"

        proj_k = self.proj_k[:kv_len, :]  # (kv_len, k)
        proj_v = proj_k if self.share_kv else self.proj_v[:kv_len, :]  # (kv_len, k)

        # --- Pre-projection masking on k and v
        if padding_and_loss_attention_mask is not None:
            # shape: (batch, 1, kv_len, 1) → broadcast across heads, head_dim
            mask_exp = padding_and_loss_attention_mask.unsqueeze(1).unsqueeze(-1)
            k = k * mask_exp
            v = v * mask_exp

        # --- Project keys and values over sequence length
        k_proj = torch.einsum('bhnd,nk->bhkd', k, proj_k)  # (batch, heads, k, head_dim)
        v_proj = torch.einsum('bhnd,nk->bhkd', v, proj_v)  # (batch, heads, k, head_dim)

        # --- Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k_proj) / math.sqrt(d_h)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)

        out = torch.einsum('bhqk,bhkd->bhqd', attn_probs, v_proj)  # (batch, heads, query_len, head_dim)
        return out, attn_probs