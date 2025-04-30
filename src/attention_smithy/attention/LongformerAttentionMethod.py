import torch
import torch.nn as nn
import torch.nn.functional as F

def chunk(x, window_size):
    batch_heads, seq_len, head_dim = x.size()
    x = x.view(batch_heads, seq_len // (window_size * 2), window_size * 2, head_dim)
    chunk_shape = list(x.size())
    chunk_shape[1] = chunk_shape[1] * 2 - 1
    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_shape, stride=chunk_stride)

def mask_start_end_extra_local_tokens(attn_scores, window_size):
    batch_heads, seq_len, window = attn_scores.shape
    assert window == 2 * window_size + 1, f"Unexpected attention window shape {window} for window size {window_size}"

    center = window_size

    full_mask = torch.zeros(seq_len, window, dtype=torch.bool, device=attn_scores.device)

    for i in range(seq_len):
        left = max(0, center - i)
        right = max(0, i + window_size + 1 - seq_len)
        if left > 0:
            full_mask[i, :left] = True  # mask left-side overflow
        if right > 0:
            full_mask[i, -right:] = True  # mask right-side overflow

    # Apply the mask to attn_scores
    attn_scores = attn_scores.masked_fill(full_mask.unsqueeze(0), float("-inf"))

    return attn_scores

class LongformerAttentionMethod(nn.Module):
    def __init__(self, attention_window: int, dropout: float = 0.0):
        super().__init__()
        self.attention_window = attention_window
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v, numeric_embedding_manager, attention_mask=None, **kwargs):
        batch_size, num_heads, seq_len, head_dim = q.size()
        assert q.size() == k.size() == v.size()
        assert seq_len % (2 * self.attention_window) == 0, "Sequence length must be divisible by 2w"

        # Flatten head dimension
        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # Compute local attention scores
        attn_scores = self._sliding_chunks_matmul_qk(q, k, self.attention_window)
        #attn_scores = self._naive_sliding_matmul_qk(q, k, self.attention_window)
        attn_scores = mask_start_end_extra_local_tokens(attn_scores, self.attention_window)

        # Apply attention mask to center of window
        if attention_mask is not None:
            key_padding_mask = attention_mask == -1  # shape: (B, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len)
            key_padding_mask = key_padding_mask.reshape(batch_size * num_heads, seq_len)
            center_idx = self.attention_window
            attn_scores[:, :, center_idx] = attn_scores[:, :, center_idx].masked_fill(
                key_padding_mask, float('-inf')
            )

        # Compute probabilities and clean up NaNs
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = self.dropout_layer(attn_probs)

        # Compute local context
        context = self._sliding_chunks_matmul_pv(attn_probs, v, self.attention_window, batch_size, num_heads)
        context = context.view(batch_size, num_heads, seq_len, head_dim)
        return context, attn_probs

    def _sliding_chunks_matmul_qk(self, q, k, window_size):
        _, _, head_dimension = q.shape
        chunk_q = chunk(q, window_size)
        chunk_k = chunk(k, window_size)
        attn_scores = torch.einsum("bcxd,bcyd->bcxy", chunk_q, chunk_k) / torch.sqrt(torch.tensor(head_dimension, dtype=torch.float32))
        batch_heads, chunks, _, _ = attn_scores.size()
        diagonal_attn = attn_scores.new_full((batch_heads, chunks + 1, window_size, 2 * window_size + 1), float('-inf'))
        # Edge-safe copying logic
        if window_size > 1 and attn_scores.shape[2] > window_size:
            diagonal_attn[:, :-1, :, window_size:] = attn_scores[:, :, :window_size, :window_size + 1]
            num_rows = attn_scores[:, -1].shape[1]
            diagonal_attn[:, -1, :, window_size:] = attn_scores[:, -1, -window_size:, :window_size + 1]
            diagonal_attn[:, 1:, :, :window_size] = attn_scores[:, :, -(window_size + 1):-1, window_size + 1:]
            diagonal_attn[:, 0, 1:window_size, 1:window_size] = attn_scores[:, 0, :window_size - 1, 1 - window_size:]
        else:
            diagonal_attn[:, :-1, :, window_size:] = attn_scores[:, :, :window_size, :window_size + 1]
            diagonal_attn[:, -1, :, window_size:] = attn_scores[:, -1, window_size:, :window_size + 1]
        final_attn_scores = diagonal_attn.view(batch_heads, -1, 2 * window_size + 1)
        return final_attn_scores

    def _naive_sliding_matmul_qk(self, q, k, window_size):
        """
        This function was written solely for debugging purposes. It is significantly less efficient
            than `_sliding_chunks_matmul_qk`, and you will need to disable the call to `mask_start_end_extra_local_tokens`
            if you use it.
        """
        batch_heads, seq_len, head_dim = q.shape
        scores = torch.full((batch_heads, seq_len, 2 * window_size + 1), float('-inf'), device=q.device)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            actual_len = end - start
            local_q = q[:, i, :]  # (B, D)
            local_k = k[:, start:end, :]  # (B, L, D)
            raw_scores = torch.einsum('bd, bld -> bl', local_q, local_k) / torch.sqrt(
                torch.tensor(head_dim, dtype=torch.float32))
            scores[:, i, window_size - (i - start):window_size + (end - i)] = raw_scores
        return scores

    def _sliding_chunks_matmul_pv(self, attn_probs, v, window_size, batch_size, num_heads):
        batch_heads, seq_len, head_dim = v.size()
        attn_probs = attn_probs.view(batch_size, num_heads, seq_len, 2 * window_size + 1)
        v = v.view(batch_size, num_heads, seq_len, head_dim)

        # Pad values to allow sliding window
        v_padded = F.pad(v, (0, 0, window_size, window_size), value=0.0)
        v_chunks = v_padded.unfold(dimension=2, size=2 * window_size + 1, step=1)
        v_chunks = v_chunks.permute(0, 1, 2, 4, 3)  # (B, H, L, D, 2w+1)
        context = torch.einsum("bhlw,bhlwd->bhld", attn_probs, v_chunks)

        return context.view(batch_size * num_heads, seq_len, head_dim)
