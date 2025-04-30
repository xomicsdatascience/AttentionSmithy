import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerAttentionMethod(nn.Module):
    def __init__(self, attention_window: int, dropout: float = 0.0):
        super().__init__()
        self.attention_window = attention_window
        self.dropout_layer = nn.Dropout(dropout)

        # Separate projection for global queries (mimicking original Longformer)
        self.query_global = nn.Identity()  # replace with nn.Linear if needed

    def forward(self, q, k, v, numeric_embedding_manager, attention_mask=None, **kwargs):
        print('\n'*10)
        batch_size, num_heads, seq_len, head_dim = q.size()
        assert attention_mask is not None, "attention_mask is required for global attention"

        # Shape: (batch_size * num_heads, seq_len, head_dim)
        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # (batch_size, seq_len) -> (batch_size * num_heads, seq_len)
        is_global = attention_mask > 0
        is_global = is_global.unsqueeze(1).expand(-1, num_heads, -1).reshape(batch_size * num_heads, seq_len)

        # All tokens attend to global tokens
        global_indices = is_global.nonzero(as_tuple=False)  # [num_global, 1+1]
        context = torch.zeros_like(q)

        if global_indices.numel() == 0:
            # No global tokens — return zeros
            context = context.view(batch_size, num_heads, seq_len, head_dim)
            attn_probs = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=q.device)
            return context, attn_probs

        # Step 1: All tokens attending to global tokens
        # Select global k/v: [batch*num_heads, num_global, head_dim]
        global_k = torch.zeros(batch_size * num_heads, seq_len, head_dim, device=q.device)
        global_v = torch.zeros_like(global_k)

        for b in range(batch_size * num_heads):
            idxs = is_global[b].nonzero(as_tuple=False).squeeze(-1)
            global_k[b, :len(idxs)] = k[b, idxs]
            global_v[b, :len(idxs)] = v[b, idxs]
        # Compute attention: [B*N, T, H] @ [B*N, H, G] -> [B*N, T, G]
        # Here we do full attention *to* global tokens
        attn_scores = torch.bmm(q, global_k.transpose(1, 2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))

        mask = torch.arange(seq_len, device=q.device).unsqueeze(0) >= is_global.sum(dim=1, keepdim=True)
        mask = mask.expand(batch_size * num_heads, -1)  # shape: [B*N, G]
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = self.dropout_layer(attn_probs)
        # Context for all tokens: weighted sum over global values
        context = torch.bmm(attn_probs, global_v)

        # Step 2: Global tokens attending to all tokens (overwrite output)
        q_global = self.query_global(q)  # Identity unless replaced with Linear

        for b in range(batch_size * num_heads):
            idxs = is_global[b].nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            qg = q_global[b, idxs]  # [num_global, head_dim]
            attn_scores_g = torch.matmul(qg, k[b].transpose(0, 1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))# [num_global, T]
            attn_probs_g = F.softmax(attn_scores_g, dim=-1, dtype=torch.float32)
            attn_probs_g = self.dropout_layer(attn_probs_g)
            context_g = torch.matmul(attn_probs_g, v[b])  # [num_global, head_dim]
            context[b, idxs] = context_g

        # Reshape back to original: [B, N, T, H]
        context = context.view(batch_size, num_heads, seq_len, head_dim)

        # Optional: For testing — build a dummy attention matrix with zeros
        # You can optionally return attn_probs if needed
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