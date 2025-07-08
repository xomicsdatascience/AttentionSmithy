from attention_smithy.numeric_embeddings.ALiBiEmbedding import ALiBiEmbedding
import torch

class ALiBiCustomEmbedding(ALiBiEmbedding):
    """
    Applies ALiBi-based attention bias using custom-defined distances.
    """

    def forward(self, alibi_query_values: torch.Tensor, alibi_key_values: torch.Tensor, value_to_not_apply_linear_bias_toward: int = None, **kwargs) -> torch.Tensor:
        """
        Computes the ALiBi bias tensor using custom query and key values.
        Args:
            alibi_query_values (torch.Tensor): The values corresponding to the query matrix that
                should denote customized "distance" from other tokens, of shape (batch_size, query_length).
            alibi_key_values (torch.Tensor): The values corresponding to the key matrix that
                should denote customized "distance" from other tokens, of shape (batch_size, kv_length).
            value_to_not_apply_linear_bias_toward (int, optional): A specific value to exclude from biasing.
        Returns:
            torch.Tensor: ALiBi bias tensor applied to the attention scores.
        """
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(
            alibi_query_values[:, :, None], alibi_key_values[:, None, :]
        )
        attention_bias = purely_negative_distance_matrix[:, None, :, :] * self.slope_m_values[None, :, :, :]

        if value_to_not_apply_linear_bias_toward is not None:
            self._negate_linear_bias_on_specified_value(
                attention_bias, alibi_query_values, alibi_key_values, value_to_not_apply_linear_bias_toward
            )

        return attention_bias

    def _negate_linear_bias_on_specified_value(self, attention_bias, alibi_query_values, alibi_key_values, value):
        """
        Removes ALiBi bias from specific query/key values.
        """
        query_mask = alibi_query_values == value
        key_mask = alibi_key_values == value
        combined_mask = query_mask[:, :, None] | key_mask[:, None, :]
        combined_mask = combined_mask[:, None, :, :]
        attention_bias.masked_fill_(combined_mask, 0)