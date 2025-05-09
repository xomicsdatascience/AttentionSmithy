import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class LongformerAttentionMethod(nn.Module):
    def __init__(self, attention_window: int, dropout: float = 0.0, embedding_dimension: int = None, use_global_weights: bool = False):
        super().__init__()
        self.attention_window = attention_window
        self.dropout_layer = nn.Dropout(dropout)
        self.use_global_weights = use_global_weights

        if self.use_global_weights:
            if embedding_dimension is None:
                raise ValueError("embedding_dimension must be provided when use_global_weights is True")
            self.global_query_weights = nn.Linear(embedding_dimension, embedding_dimension)
            self.global_key_weights = nn.Linear(embedding_dimension, embedding_dimension)
            self.global_value_weights = nn.Linear(embedding_dimension, embedding_dimension)

    def _forward_inner(
        self,
        q, k, v,
        numeric_embedding_manager,
        global_attention_mask,
        padding_and_loss_attention_mask,
        input_query, input_key, input_value,
        **kwargs
    ):
        device, scale = self._variable_setting_and_basic_assertions(k, q, v)
        k, q, v = self._reshape_inputs(k, q, v)
        is_global, padding_mask = self._prepare_masks(device, global_attention_mask, padding_and_loss_attention_mask)

        local_attn_scores = self._compute_local_attention(is_global, k, padding_mask, q)

        max_global = self._determine_max_global_of_each_sample(is_global)
        if max_global == 0:
            context = self._finalize_local_attention_only(local_attn_scores, v)
            return context, None

        batch_head_idx, token_idx = is_global.nonzero(as_tuple=True)

        global_keys_sparse, global_values_sparse, global_mask, global_token_positions = self._slim_down_kv_matrices_to_just_global_position_versions(
            batch_head_idx, device, is_global, k, token_idx, v, max_global)

        self._apply_padding_to_all_t_global(batch_head_idx, global_mask, global_token_positions, padding_mask, token_idx)

        global_attn_scores = self._compute_all_t_global_attention_scores(global_keys_sparse, global_mask, q, scale)

        used_key, used_query, used_value = self._determine_matries_to_use_for_global_t_global_calculations(input_key, input_query, input_value, k,
                                                                                q, v)

        global_key, global_query, global_value = self._select_global_t_global_tokens_only(batch_head_idx, token_idx, used_key,
                                                                                 used_query, used_value)

        global_out_scores = self._calculate_global_t_global_attention(global_key, global_query, scale)

        global_out_scores, global_value = self._apply_padding_to_global_t_global_attention_scores(batch_head_idx,
                                                                                                  global_out_scores,
                                                                                                  global_value,
                                                                                                  padding_mask)

        context = self._combine_all_context_outputs(batch_head_idx, global_attn_scores, global_out_scores, global_value,
                                                    global_values_sparse, local_attn_scores, max_global, token_idx, v)

        context = context.view(self.batch_size, self.num_heads, self.sequence_length, self.head_dimension)

        return context, None

    def _combine_all_context_outputs(self, batch_head_idx, global_attn_scores, global_out_scores, global_value,
                                     global_values_sparse, local_attn_scores, max_global, token_idx, v):
        context = self._combine_local_and_all_t_global_contexts(global_attn_scores, global_values_sparse,
                                                                local_attn_scores,
                                                                max_global, v)
        context_global = self._calculate_global_t_global_output(global_out_scores, global_value)
        context[batch_head_idx, token_idx] = context_global
        return context

    def _calculate_global_t_global_output(self, global_out_scores, global_value):
        global_out_probs = F.softmax(global_out_scores, dim=-1)
        global_out_probs = self.dropout_layer(global_out_probs)
        context_global = torch.bmm(global_out_probs.unsqueeze(1), global_value).squeeze(1)
        return context_global

    def _apply_padding_to_global_t_global_attention_scores(self, batch_head_idx, global_out_scores, global_value,
                                                           padding_mask):
        if padding_mask is not None:
            kv_mask = padding_mask[batch_head_idx]
            global_out_scores = global_out_scores.masked_fill(kv_mask == 0, float('-inf'))
            global_value = global_value.masked_fill(kv_mask.unsqueeze(-1) == 0, 0.0)
        return global_out_scores, global_value

    def _calculate_global_t_global_attention(self, global_key, global_query, scale):
        global_out_scores = torch.bmm(global_query.unsqueeze(1), global_key.transpose(1, 2)).squeeze(1) / scale
        return global_out_scores

    def _select_global_t_global_tokens_only(self, batch_head_idx, token_idx, used_key, used_query, used_value):
        global_query = used_query[batch_head_idx, token_idx]
        global_key = used_key[batch_head_idx]
        global_value = used_value[batch_head_idx]
        return global_key, global_query, global_value

    def _determine_matries_to_use_for_global_t_global_calculations(self, input_key, input_query, input_value, k, q, v):
        if self.use_global_weights:

            global_query_all_tokens = self.global_query_weights(input_query)
            global_key_all_tokens = self.global_key_weights(input_key)
            global_value_all_tokens = self.global_value_weights(input_value)

            # Reshape raw inputs
            global_query_all_tokens = global_query_all_tokens.view(self.batch_size, self.sequence_length,
                                                                   self.num_heads, self.head_dimension).transpose(1, 2)
            global_key_all_tokens = global_key_all_tokens.view(self.batch_size, self.sequence_length, self.num_heads,
                                                               self.head_dimension).transpose(1, 2)
            global_value_all_tokens = global_value_all_tokens.view(self.batch_size, self.sequence_length,
                                                                   self.num_heads, self.head_dimension).transpose(1, 2)

            # Flatten
            global_key_all_tokens, global_query_all_tokens, global_value_all_tokens = self._reshape_inputs(
                global_key_all_tokens, global_query_all_tokens, global_value_all_tokens)

            used_query = global_query_all_tokens
            used_key = global_key_all_tokens
            used_value = global_value_all_tokens

        else:
            used_query = q
            used_key = k
            used_value = v
        return used_key, used_query, used_value

    def _combine_local_and_all_t_global_contexts(self, global_attn_scores, global_values_sparse, local_attn_scores,
                                           max_global, v):
        attn_scores = torch.cat([global_attn_scores, local_attn_scores], dim=-1)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        global_probs = attn_probs[:, :, :max_global]
        local_probs = attn_probs[:, :, max_global:]
        context_local = _sliding_chunks_matmul_pv(local_probs, v, self.attention_window, self.batch_size,
                                                  self.num_heads)
        context_global = torch.bmm(global_probs, global_values_sparse)
        context = context_local + context_global
        return context

    def _compute_all_t_global_attention_scores(self, global_keys_sparse, global_mask, q, scale):
        q_scaled = q / scale
        global_attn_scores = torch.bmm(q_scaled, global_keys_sparse.transpose(1, 2))
        global_attn_scores = global_attn_scores.masked_fill(~global_mask.unsqueeze(1), float('-inf'))
        return global_attn_scores

    def _apply_padding_to_all_t_global(self, batch_head_idx, global_mask, global_token_positions, padding_mask, token_idx):
        if padding_mask is not None:
            padding_global = padding_mask[batch_head_idx, token_idx].bool()
            global_mask[batch_head_idx, global_token_positions[batch_head_idx, token_idx]] &= padding_global

    def _slim_down_kv_matrices_to_just_global_position_versions(self, batch_head_idx, device, is_global, k, token_idx, v, max_global):
        global_token_positions = is_global.cumsum(dim=1) - 1
        global_token_positions = global_token_positions.masked_fill(~is_global, 0)
        global_keys_sparse = torch.zeros(self.batch_size * self.num_heads, max_global, self.head_dimension,
                                         device=device)
        global_values_sparse = torch.zeros_like(global_keys_sparse)
        global_mask = torch.zeros(self.batch_size * self.num_heads, max_global, dtype=torch.bool, device=device)
        global_keys_sparse[batch_head_idx, global_token_positions[batch_head_idx, token_idx]] = k[
            batch_head_idx, token_idx]
        global_values_sparse[batch_head_idx, global_token_positions[batch_head_idx, token_idx]] = v[
            batch_head_idx, token_idx]
        global_mask[batch_head_idx, global_token_positions[batch_head_idx, token_idx]] = True
        return global_keys_sparse, global_values_sparse, global_mask, global_token_positions

    def _determine_max_global_of_each_sample(self, is_global):
        max_global = is_global.sum(dim=1).max().item()
        return max_global

    def _finalize_local_attention_only(self, local_attn_scores, v):
        attn_probs = F.softmax(local_attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        context = _sliding_chunks_matmul_pv(attn_probs, v, self.attention_window, self.batch_size, self.num_heads)
        context = context.view(self.batch_size, self.num_heads, self.sequence_length, self.head_dimension)
        return context

    def _compute_local_attention(self, is_global, k, padding_mask, q):
        local_attn_scores = _sliding_chunks_matmul_qk(q, k, self.attention_window)
        local_attn_scores = _mask_local_attention_edges_and_globals_and_padding(local_attn_scores,
                                                                                self.attention_window,
                                                                                is_global=is_global,
                                                                                padding_mask=padding_mask)
        return local_attn_scores

    def _prepare_masks(self, device, global_attention_mask, padding_and_loss_attention_mask):
        if padding_and_loss_attention_mask is not None:
            padding_mask = padding_and_loss_attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1).reshape(
                self.batch_size * self.num_heads, self.sequence_length)
        else:
            padding_mask = None
        if global_attention_mask is None:
            is_global = torch.zeros(self.batch_size * self.num_heads, self.sequence_length, dtype=torch.bool,
                                    device=device)
        else:
            is_global = global_attention_mask.bool().unsqueeze(1).expand(-1, self.num_heads, -1).reshape(
                self.batch_size * self.num_heads, self.sequence_length).to(device)
        return is_global, padding_mask

    def _reshape_inputs(self, k, q, v):
        q = q.reshape(self.batch_size * self.num_heads, self.sequence_length, self.head_dimension)
        k = k.reshape(self.batch_size * self.num_heads, self.sequence_length, self.head_dimension)
        v = v.reshape(self.batch_size * self.num_heads, self.sequence_length, self.head_dimension)
        return k, q, v

    def _variable_setting_and_basic_assertions(self, k, q, v):
        self.batch_size, self.num_heads, self.sequence_length, self.head_dimension = q.size()
        device = q.device
        scale = self.head_dimension ** 0.5
        assert q.size() == k.size() == v.size()
        assert self.sequence_length % (2 * self.attention_window) == 0, "Sequence length must be divisible by 2w"
        return device, scale

    def forward(
        self,
        q, k, v,
        numeric_embedding_manager,
        global_attention_mask=None,
        padding_and_loss_attention_mask=None,
        input_query=None, input_key=None, input_value=None,
        **kwargs
    ):
        return checkpoint(
            self._forward_inner,
            q, k, v, numeric_embedding_manager, global_attention_mask, padding_and_loss_attention_mask,
            input_query, input_key, input_value,
            **kwargs,
            use_reentrant=False,
        )

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

def _mask_local_attention_edges_and_globals_and_padding(attn_scores, window_size, is_global=None, padding_mask=None):
    batch_heads, seq_len, window = attn_scores.shape
    assert window == 2 * window_size + 1, f"Unexpected attention window shape {window} for window size {window_size}"
    device = attn_scores.device
    center = window_size

    # Step 1: Sequence boundary masking (fully vectorized)
    token_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    window_offsets = torch.arange(-window_size, window_size + 1, device=device).unsqueeze(0)
    rel_positions = token_indices + window_offsets

    mask_out_of_bounds = (rel_positions < 0) | (rel_positions >= seq_len)
    full_mask = mask_out_of_bounds  # (T, W)

    attn_scores.masked_fill_(full_mask.unsqueeze(0), float('-inf'))

    # Step 2: Mask global tokens from local attention
    if is_global is not None:
        is_global = is_global.bool()

        # Lookup global token status for each relative position
        rel_positions = rel_positions.clamp(min=0, max=seq_len - 1).to(device)
        global_mask = torch.gather(
            is_global.unsqueeze(1).expand(-1, seq_len, -1),  # (B*N, T, T)
            dim=2,
            index=rel_positions.unsqueeze(0).expand(batch_heads, -1, -1)  # (B*N, T, W)
        )  # → (B*N, T, W)

        attn_scores.masked_fill_(global_mask, float('-inf'))

    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        padding_mask_expanded = torch.gather(
            padding_mask.unsqueeze(1).expand(-1, seq_len, -1),  # (B*N, T, T)
            dim=2,
            index=rel_positions.unsqueeze(0).expand(batch_heads, -1, -1)
        )
        attn_scores.masked_fill_(~padding_mask_expanded, float('-inf'))


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
    """
    This function was written solely for debugging purposes. It is significantly less efficient than
         _sliding_chunks_matmul_qk`, and you will need to disable the call to `mask_start_end_extra_local_tokens`
         if you use it.
    """
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