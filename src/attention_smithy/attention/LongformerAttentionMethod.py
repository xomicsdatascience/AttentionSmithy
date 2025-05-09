import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import namedtuple

AttentionInputs = namedtuple('AttentionInputs', ['q', 'k', 'v'])
OriginalInputs = namedtuple('OriginalInputs', ['query', 'key', 'value'])
MasksAndGlobals = namedtuple('MasksAndGlobals', ['global_mask', 'padding_mask', 'is_global', 'max_global', 'batch_head_idx', 'token_idx'])

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

    def _set_context(self, q):
        self._ctx = {
            'batch_size': q.size(0),
            'num_heads': q.size(1),
            'seq_len': q.size(2),
            'head_dim': q.size(3),
            'device': q.device,
            'scale': q.size(3) ** 0.5,
        }

    def forward(self, q, k, v, numeric_embedding_manager, global_attention_mask=None, padding_and_loss_attention_mask=None, input_query=None, input_key=None, input_value=None, **kwargs):
        self._set_context(q)
        k, q, v = self._reshape_inputs(k, q, v)

        attn_inputs = AttentionInputs(q, k, v)
        orig_inputs = OriginalInputs(input_query, input_key, input_value)

        masks_and_globals = self._prepare_masks_and_globals(global_attention_mask, padding_and_loss_attention_mask)

        return checkpoint(
            self._forward_inner,
            attn_inputs,
            numeric_embedding_manager,
            masks_and_globals,
            orig_inputs,
            **kwargs,
            use_reentrant=False,
        )

    def _forward_inner(self, attn_inputs, numeric_embedding_manager, masks_and_globals, orig_inputs, **kwargs):
        self._basic_assertions(attn_inputs)
        local_attn_scores = self._compute_local_attention(attn_inputs, masks_and_globals)

        if masks_and_globals.max_global == 0:
            context = self._finalize_local_attention_only(local_attn_scores, attn_inputs.v)
            return context, None

        global_keys_sparse, global_values_sparse, global_mask_sparse, global_token_positions = self._slim_down_kv_matrices_to_just_global_position_versions(
            attn_inputs, masks_and_globals)

        self._apply_padding_to_all_t_global(global_token_positions, global_mask_sparse, masks_and_globals)

        global_attn_scores = self._compute_all_t_global_attention_scores(global_keys_sparse, global_mask_sparse, attn_inputs)

        used_key, used_query, used_value = self._determine_matries_to_use_for_global_t_global_calculations(attn_inputs, orig_inputs)

        global_key, global_query, global_value = self._select_global_t_global_tokens_only(used_key, used_query, used_value, masks_and_globals)

        global_out_scores = self._calculate_global_t_global_attention(global_key, global_query)

        global_out_scores, global_value = self._apply_padding_to_global_t_global_attention_scores(global_out_scores,
                                                                                                  global_value,
                                                                                                  masks_and_globals)

        context = self._combine_all_context_outputs(global_attn_scores, global_out_scores, global_value,
                                                    global_values_sparse, local_attn_scores, attn_inputs, masks_and_globals)

        context = context.view(self._ctx["batch_size"], self._ctx["num_heads"], self._ctx["seq_len"], self._ctx["head_dim"])

        return context, None

    def _combine_all_context_outputs(self, global_attn_scores, global_out_scores, global_value,
                                     global_values_sparse, local_attn_scores, attn_inputs, masks_and_globals):
        context = self._combine_local_and_all_t_global_contexts(global_attn_scores, global_values_sparse,
                                                                local_attn_scores, attn_inputs, masks_and_globals)
        context_global = self._calculate_global_t_global_output(global_out_scores, global_value)
        context[masks_and_globals.batch_head_idx, masks_and_globals.token_idx] = context_global
        return context

    def _calculate_global_t_global_output(self, global_out_scores, global_value):
        global_out_probs = F.softmax(global_out_scores, dim=-1)
        global_out_probs = self.dropout_layer(global_out_probs)
        context_global = torch.bmm(global_out_probs.unsqueeze(1), global_value).squeeze(1)
        return context_global

    def _apply_padding_to_global_t_global_attention_scores(self, global_out_scores, global_value, masks_and_globals):
        if masks_and_globals.padding_mask is not None:
            kv_mask = masks_and_globals.padding_mask[masks_and_globals.batch_head_idx]
            global_out_scores = global_out_scores.masked_fill(kv_mask == 0, float('-inf'))
            global_value = global_value.masked_fill(kv_mask.unsqueeze(-1) == 0, 0.0)
        return global_out_scores, global_value

    def _calculate_global_t_global_attention(self, global_key, global_query):
        global_out_scores = torch.bmm(global_query.unsqueeze(1), global_key.transpose(1, 2)).squeeze(1) / self._ctx["scale"]
        return global_out_scores

    def _select_global_t_global_tokens_only(self, used_key, used_query, used_value, masks_and_globals):
        global_query = used_query[masks_and_globals.batch_head_idx, masks_and_globals.token_idx]
        global_key = used_key[masks_and_globals.batch_head_idx]
        global_value = used_value[masks_and_globals.batch_head_idx]
        return global_key, global_query, global_value

    def _determine_matries_to_use_for_global_t_global_calculations(self, attn_inputs, orig_inputs):
        if self.use_global_weights:

            global_query_all_tokens = self.global_query_weights(orig_inputs.query)
            global_key_all_tokens = self.global_key_weights(orig_inputs.key)
            global_value_all_tokens = self.global_value_weights(orig_inputs.value)

            # Reshape raw inputs
            global_query_all_tokens = global_query_all_tokens.view(self._ctx["batch_size"], self._ctx["seq_len"],
                                                                   self._ctx["num_heads"], self._ctx["head_dim"]).transpose(1, 2)
            global_key_all_tokens = global_key_all_tokens.view(self._ctx["batch_size"], self._ctx["seq_len"], self._ctx["num_heads"],
                                                               self._ctx["head_dim"]).transpose(1, 2)
            global_value_all_tokens = global_value_all_tokens.view(self._ctx["batch_size"], self._ctx["seq_len"],
                                                                   self._ctx["num_heads"], self._ctx["head_dim"]).transpose(1, 2)

            # Flatten
            global_key_all_tokens, global_query_all_tokens, global_value_all_tokens = self._reshape_inputs(
                global_key_all_tokens, global_query_all_tokens, global_value_all_tokens)

            used_query = global_query_all_tokens
            used_key = global_key_all_tokens
            used_value = global_value_all_tokens

        else:
            used_query = attn_inputs.q
            used_key = attn_inputs.k
            used_value = attn_inputs.v
        return used_key, used_query, used_value

    def _combine_local_and_all_t_global_contexts(self, global_attn_scores, global_values_sparse, local_attn_scores, attn_inputs, masks_and_globals):
        attn_scores = torch.cat([global_attn_scores, local_attn_scores], dim=-1)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        global_probs = attn_probs[:, :, :masks_and_globals.max_global]
        local_probs = attn_probs[:, :, masks_and_globals.max_global:]
        context_local = _sliding_chunks_matmul_pv(local_probs, attn_inputs.v, self.attention_window, self._ctx["batch_size"],
                                                  self._ctx["num_heads"])
        context_global = torch.bmm(global_probs, global_values_sparse)
        context = context_local + context_global
        return context

    def _compute_all_t_global_attention_scores(self, global_keys_sparse, global_mask_sparse, attn_inputs):
        q_scaled = attn_inputs.q / self._ctx["scale"]
        global_attn_scores = torch.bmm(q_scaled, global_keys_sparse.transpose(1, 2))
        global_attn_scores = global_attn_scores.masked_fill(~global_mask_sparse.unsqueeze(1), float('-inf'))
        return global_attn_scores

    def _apply_padding_to_all_t_global(self, global_token_positions, global_mask_sparse, masks_and_globals):
        if masks_and_globals.padding_mask is not None:
            padding_global = masks_and_globals.padding_mask[masks_and_globals.batch_head_idx, masks_and_globals.token_idx].bool()
            global_mask_sparse[masks_and_globals.batch_head_idx, global_token_positions[masks_and_globals.batch_head_idx, masks_and_globals.token_idx]] &= padding_global

    def _slim_down_kv_matrices_to_just_global_position_versions(self, attn_inputs, masks_and_globals):
        global_token_positions = masks_and_globals.is_global.cumsum(dim=1) - 1
        global_token_positions = global_token_positions.masked_fill(~masks_and_globals.is_global, 0)
        global_keys_sparse = torch.zeros(self._ctx["batch_size"] * self._ctx["num_heads"], masks_and_globals.max_global, self._ctx["head_dim"],
                                         device=self._ctx["device"])
        global_values_sparse = torch.zeros_like(global_keys_sparse)
        global_mask = torch.zeros(self._ctx["batch_size"] * self._ctx["num_heads"], masks_and_globals.max_global, dtype=torch.bool, device=self._ctx["device"])
        global_keys_sparse[masks_and_globals.batch_head_idx, global_token_positions[masks_and_globals.batch_head_idx, masks_and_globals.token_idx]] = attn_inputs.k[
            masks_and_globals.batch_head_idx, masks_and_globals.token_idx]
        global_values_sparse[masks_and_globals.batch_head_idx, global_token_positions[masks_and_globals.batch_head_idx, masks_and_globals.token_idx]] = attn_inputs.v[
            masks_and_globals.batch_head_idx, masks_and_globals.token_idx]
        global_mask[masks_and_globals.batch_head_idx, global_token_positions[masks_and_globals.batch_head_idx, masks_and_globals.token_idx]] = True
        return global_keys_sparse, global_values_sparse, global_mask, global_token_positions

    def _determine_max_global_of_each_sample(self, is_global):
        max_global = is_global.sum(dim=1).max().item()
        return max_global

    def _finalize_local_attention_only(self, local_attn_scores, v):
        attn_probs = F.softmax(local_attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        context = _sliding_chunks_matmul_pv(attn_probs, v, self.attention_window, self._ctx["batch_size"], self._ctx["num_heads"])
        context = context.view(self._ctx["batch_size"], self._ctx["num_heads"], self._ctx["seq_len"], self._ctx["head_dim"])
        return context

    def _compute_local_attention(self, attn_inputs, masks):
        local_attn_scores = _sliding_chunks_matmul_qk(attn_inputs.q, attn_inputs.k, self.attention_window)
        local_attn_scores = _mask_local_attention_edges_and_globals_and_padding(local_attn_scores,
                                                                                self.attention_window,
                                                                                is_global=masks.is_global,
                                                                                padding_mask=masks.padding_mask)
        return local_attn_scores

    def _prepare_masks_and_globals(self, global_attention_mask, padding_and_loss_attention_mask):
        is_global, padding_mask = self._prepare_masks(self._ctx["device"], global_attention_mask,
                                                      padding_and_loss_attention_mask)
        max_global = self._determine_max_global_of_each_sample(is_global)
        batch_head_idx, tuple_idx, = is_global.nonzero(as_tuple=True)
        masks_and_globals = MasksAndGlobals(global_attention_mask, padding_mask, is_global, max_global, batch_head_idx,
                                            tuple_idx)
        return masks_and_globals


    def _prepare_masks(self, device, global_attention_mask, padding_and_loss_attention_mask):
        if padding_and_loss_attention_mask is not None:
            padding_mask = padding_and_loss_attention_mask.unsqueeze(1).expand(-1, self._ctx["num_heads"], -1).reshape(
                self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"])
        else:
            padding_mask = None
        if global_attention_mask is None:
            is_global = torch.zeros(self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"], dtype=torch.bool,
                                    device=self._ctx["device"])
        else:
            is_global = global_attention_mask.bool().unsqueeze(1).expand(-1, self._ctx["num_heads"], -1).reshape(
                self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"]).to(self._ctx["device"])
        return is_global, padding_mask

    def _reshape_inputs(self, k, q, v):
        q = q.reshape(self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"], self._ctx["head_dim"])
        k = k.reshape(self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"], self._ctx["head_dim"])
        v = v.reshape(self._ctx["batch_size"] * self._ctx["num_heads"], self._ctx["seq_len"], self._ctx["head_dim"])
        return k, q, v

    def _basic_assertions(self, attn_inputs):
        assert attn_inputs.q.size() == attn_inputs.k.size() == attn_inputs.v.size()
        assert self._ctx["seq_len"] % (2 * self.attention_window) == 0, "Sequence length must be divisible by 2w"

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