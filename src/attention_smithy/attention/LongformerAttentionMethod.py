import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerAttentionMethod(nn.Module):
    def __init__(self, attention_window: int, dropout: float = 0.0):
        super().__init__()
        self.attention_window = attention_window
        self.dropout_layer = nn.Dropout(dropout)
        self.query_global = nn.Identity()  # or replace with nn.Linear(...)

    def forward(self, q, k, v, numeric_embedding_manager, attention_mask=None, **kwargs):
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        scale = head_dim ** 0.5

        assert attention_mask is not None

        # Merge heads into batch dim for easier handling
        q = q.view(batch_size * num_heads, seq_len, head_dim)
        k = k.view(batch_size * num_heads, seq_len, head_dim)
        v = v.view(batch_size * num_heads, seq_len, head_dim)

        # (B * N, T)
        is_global = attention_mask > 0
        is_global = is_global.unsqueeze(1).expand(-1, num_heads, -1).reshape(batch_size * num_heads, seq_len)
        num_global = is_global.sum(dim=1)  # (B * N,)
        max_global = num_global.max().item()

        if max_global == 0:
            context = torch.zeros_like(q).view(batch_size, num_heads, seq_len, head_dim)
            attn_probs = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
            return context, attn_probs

        # Step 1: All tokens attend to global tokens

        # Extract (flat) indices of global tokens
        b_idx, t_idx = is_global.nonzero(as_tuple=True)
        global_kv = torch.zeros(batch_size * num_heads, max_global, head_dim, device=device)
        global_kv_mask = torch.zeros(batch_size * num_heads, max_global, dtype=torch.bool, device=device)

        # Create index ranges
        global_index_pos = torch.zeros_like(is_global, dtype=torch.long)
        global_counts = num_global.tolist()
        # Generate per-row global indices (cumsum trick)
        global_index_pos = is_global.cumsum(dim=1) - 1  # 0-based index for each global token
        global_index_pos = global_index_pos.masked_fill(~is_global, 0)  # don't care for non-global
        # Now scatter keys and values into padded tensors
        global_kv[b_idx, global_index_pos[b_idx, t_idx]] = k[b_idx, t_idx]

        global_v = torch.zeros(batch_size * num_heads, max_global, head_dim, device=device)
        global_v[b_idx, global_index_pos[b_idx, t_idx]] = v[b_idx, t_idx]
        # Mask indicating valid entries in padded global_kv
        global_kv_mask[b_idx, global_index_pos[b_idx, t_idx]] = True

        # Compute attention scores: all tokens -> global tokens
        q_scaled = q / scale
        attn_scores = torch.bmm(q_scaled, global_kv.transpose(1, 2))  # (B*N, T, G)
        # Mask out padding in global positions
        attn_scores = attn_scores.masked_fill(~global_kv_mask.unsqueeze(1), float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = self.dropout_layer(attn_probs)
        context = torch.bmm(attn_probs, global_v)  # (B*N, T, H)

        # Step 2: Global tokens attend to all tokens

        # Apply global projection to q if desired
        q_global = self.query_global(q)  # (B*N, T, H)

        # Gather global queries
        qg = q_global[b_idx, t_idx]  # (num_global_tokens, H)
        kg = k[b_idx]                # (num_global_tokens, T, H)
        vg = v[b_idx]                # (num_global_tokens, T, H)

        attn_scores_g = torch.bmm(qg.unsqueeze(1), kg.transpose(1, 2)).squeeze(1) / scale  # (G, T)
        attn_probs_g = F.softmax(attn_scores_g, dim=-1, dtype=torch.float32)
        attn_probs_g = self.dropout_layer(attn_probs_g)
        context_g = torch.bmm(attn_probs_g.unsqueeze(1), vg).squeeze(1)  # (G, H)

        # Overwrite context for global tokens
        context[b_idx, t_idx] = context_g

        # Reshape back to (B, N, T, H)
        context = context.view(batch_size, num_heads, seq_len, head_dim)
        return context, attn_probs.view(batch_size, num_heads, seq_len, -1)

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerAttentionMethod(nn.Module):
    def __init__(self, attention_window: int, dropout: float = 0.0):
        super().__init__()
        self.attention_window = attention_window
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v, numeric_embedding_manager, attention_mask=None, **kwargs):
        batch_size, num_heads, seq_len, head_dim = q.size()
        assert q.size() == k.size() == v.size()
        assert seq_len % (2 * self.attention_window) == 0, "Sequence length must be divisible by 2w"

        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_heads, seq_len, head_dim)

        attn_scores = _sliding_chunks_matmul_qk(q, k, self.attention_window)
        attn_scores = _mask_start_end_extra_local_tokens(attn_scores, self.attention_window)

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = self.dropout_layer(attn_probs)

        context = _sliding_chunks_matmul_pv(attn_probs, v, self.attention_window, batch_size, num_heads)
        context = context.view(batch_size, num_heads, seq_len, head_dim)
        return context, attn_probs
'''
def _slice_attention_score_chunks_into_exact_windows(attn_scores_in_chunks, window_size):
    diagonal = _skew_query_key_matrix(attn_scores_in_chunks, direction=(0, 0, 0, 1), padding_value=-float("inf"))
    batch_heads, chunks, _, _ = attn_scores_in_chunks.size()
    diagonal_attn = diagonal.new_full((batch_heads, chunks + 1, window_size, 2 * window_size + 1), float('-inf'))
    diagonal_attn[:, :-1, :, window_size:] = diagonal[:, :, :window_size, :window_size + 1]
    diagonal_attn[:, -1, :, window_size:] = diagonal[:, -1, window_size:, :window_size + 1]
    diagonal_attn[:, 1:, :, :window_size] = diagonal[:, :, -(window_size + 1):-1, window_size + 1:]
    diagonal_attn[:, 0, 1:window_size, 1:window_size] = diagonal[:, 0, :window_size - 1, 1 - window_size:]
    final_attention_scores = diagonal_attn.view(batch_heads, -1, 2 * window_size + 1)
    return final_attention_scores

def _skew_query_key_matrix(x, direction, padding_value):
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded

def _mask_start_end_extra_local_tokens(attn_scores, window_size):
    batch_heads, seq_len, window = attn_scores.shape
    assert window == 2 * window_size + 1, f"Unexpected attention window shape {window} for window size {window_size}"

    center = window_size
    full_mask = torch.zeros(seq_len, window, dtype=torch.bool, device=attn_scores.device)

    for i in range(seq_len):
        left = max(0, center - i)
        right = max(0, i + window_size + 1 - seq_len)
        if left > 0:
            full_mask[i, :left] = True
        if right > 0:
            full_mask[i, -right:] = True

    attn_scores = attn_scores.masked_fill(full_mask.unsqueeze(0), float("-inf"))
    return attn_scores

def _sliding_chunks_matmul_qk(q, k, window_size):
    _, _, head_dimension = q.shape
    chunk_q = _prepare_strided_chunks_for_attention(q, window_size)
    chunk_k = _prepare_strided_chunks_for_attention(k, window_size)
    attn_scores_in_chunks = torch.einsum("bcxd,bcyd->bcxy", chunk_q, chunk_k) / torch.sqrt(torch.tensor(head_dimension, dtype=torch.float32))
    return _slice_attention_score_chunks_into_exact_windows(attn_scores_in_chunks, window_size)

def _prepare_strided_chunks_for_attention(x, window_size):
    batch_heads, seq_len, head_dim = x.size()
    x = x.view(batch_heads, seq_len // (window_size * 2), window_size * 2, head_dim)
    chunk_shape = list(x.size())
    chunk_shape[1] = chunk_shape[1] * 2 - 1
    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_shape, stride=chunk_stride)

def _naive_sliding_matmul_qk(q, k, window_size):
    batch_heads, seq_len, head_dim = q.shape
    scores = torch.full((batch_heads, seq_len, 2 * window_size + 1), float('-inf'), device=q.device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        actual_len = end - start
        local_q = q[:, i, :]
        local_k = k[:, start:end, :]
        raw_scores = torch.einsum('bd, bld -> bl', local_q, local_k) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        scores[:, i, window_size - (i - start):window_size + (end - i)] = raw_scores
    return scores

def _sliding_chunks_matmul_pv(attn_probs, v, window_size, batch_size, num_heads):
    batch_heads, seq_len, head_dim = v.size()
    attn_probs = attn_probs.view(batch_size, num_heads, seq_len, 2 * window_size + 1)
    v = v.view(batch_size, num_heads, seq_len, head_dim)

    v_padded = F.pad(v, (0, 0, window_size, window_size), value=0.0)
    v_chunks = v_padded.unfold(dimension=2, size=2 * window_size + 1, step=1)
    v_chunks = v_chunks.permute(0, 1, 2, 4, 3)
    context = torch.einsum("bhlw,bhlwd->bhld", attn_probs, v_chunks)

    return context.view(batch_size * num_heads, seq_len, head_dim)