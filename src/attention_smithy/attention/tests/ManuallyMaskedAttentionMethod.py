import torch
from attention_smithy.attention import StandardAttentionMethod


class ManuallyMaskedAttentionMethod(StandardAttentionMethod):
    """
    Extends StandardAttentionMethod to support an explicit (query_length x kv_length) binary attention mask,
    which overrides any default masking like padding or causal.
    """

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            numeric_embedding_manager,
            manual_attention_mask: torch.Tensor = None,
            **kwargs,
    ):
        attention_scores = self._calculate_query_by_key_attention_scores(q, k)
        attention_scores += numeric_embedding_manager.create_bias_tensor(
            attention_score_matrix=attention_scores, query=q, key=k, **kwargs
        )

        if manual_attention_mask is not None:
            attention_scores = self._apply_manual_mask(attention_scores, manual_attention_mask)
        else:
            attention_scores = self._apply_masking_to_attention_scores(
                attention_scores, kwargs.get("padding_and_loss_attention_mask", None)
            )
        attention_probabilities = self._reduce_attention_scores_to_probabilities(attention_scores)
        attention_outputs = torch.matmul(attention_probabilities, v)
        return attention_outputs, attention_probabilities

    def _apply_manual_mask(self, attention_scores, manual_mask):
        """
        manual_mask:
            - (query_length, kv_length): apply same mask across all batches and heads
            - (batch_size, query_length, kv_length): apply per-batch mask, shared across heads
        """
        batch_size, num_heads, query_length, key_length = attention_scores.shape

        if manual_mask.ndim == 2:
            if manual_mask.shape != (query_length, key_length):
                raise ValueError(
                    f"2D manual_attention_mask must have shape (query_length, key_length), got {manual_mask.shape}"
                )
            manual_mask = manual_mask.to(device=attention_scores.device, dtype=attention_scores.dtype)
            expanded_mask = manual_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, query, key)
        elif manual_mask.ndim == 3:
            if manual_mask.shape != (batch_size, query_length, key_length):
                raise ValueError(
                    f"3D manual_attention_mask must have shape (batch_size, query_length, key_length), got {manual_mask.shape}"
                )
            manual_mask = manual_mask.to(device=attention_scores.device, dtype=attention_scores.dtype)
            expanded_mask = manual_mask.unsqueeze(1)  # (batch, 1, query, key)
        else:
            raise ValueError(
                f"manual_attention_mask must be 2D or 3D, got {manual_mask.ndim}D"
            )

        attention_scores = attention_scores.masked_fill(expanded_mask == 0, float("-inf"))
        return attention_scores