import math
import torch
import warnings
from torch import nn
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import AttentionBiasStrategyBase


class ALiBiEmbedding(AttentionBiasStrategyBase):
    """
    Base class for ALiBi embeddings, adjusting attention scores to prioritize nearby tokens.
    """

    def __init__(self, number_of_heads: int, slope_degree: int = 2) -> None:
        """
        Args:
            number_of_heads (int): The number of heads used in multihead attention.
            slope_degree (int, optional): Determines the degree of difference between heads. Defaults to 2.
        """
        super().__init__()
        slopes = self._get_slopes(number_of_heads, slope_degree)
        self.register_buffer('slope_m_values', slopes)

    def _determine_negative_distance_matrix(self, query_values, kv_values):
        distance_matrix = query_values - kv_values
        purely_negative_distance_matrix = torch.where(distance_matrix > 0, -distance_matrix, distance_matrix)
        return purely_negative_distance_matrix

    def _get_slopes(self, number_of_heads, slope_degree):
        slopes = (1 / slope_degree) ** torch.arange(1, number_of_heads + 1)
        return slopes.view(number_of_heads, 1, 1)

    def create_bias_tensor(self, **kwargs) -> torch.Tensor:
        return self.forward(**kwargs)


class ALiBiPositionEmbedding(ALiBiEmbedding):
    """
    Applies ALiBi-based attention bias based on token positions.
    """

    def __init__(self, number_of_heads: int, slope_degree: int = 2, allow_cross_attention: bool = False) -> None:
        """
        Args:
            number_of_heads (int): See parent class.
            slope_degree (int, optional): See parent class.
            allow_cross_attention (bool, optional): If True, enables cross-attention. Defaults to False.
        """
        super().__init__(number_of_heads, slope_degree)
        self.allow_cross_attention = allow_cross_attention

    def forward(self, query: torch.Tensor, key: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Computes the ALiBi bias tensor.
        Args:
            kwargs["query_length"] (int): Length of the query sequence.
            kwargs["kv_length"] (int): Length of the key sequence.
        Returns:
            torch.Tensor: ALiBi attention bias of shape (num_heads, query_length, key_length), broadcast across the batch.
        """
        query_length = query.shape[2]
        kv_length = key.shape[2]
        if query_length != kv_length and not self.allow_cross_attention:
            warnings.warn(
                'ALiBi position encoding used, but not enabled for cross attention by default. Set `allow_cross_attention=True` in initiliazation of ALiBiPositionEmbedding if you want to change this behavior (not recommended).')
            return torch.zeros((self.slope_m_values.shape[0], query_length, kv_length),
                               device=self.slope_m_values.device)

        query_positions = torch.arange(query_length, device=self.slope_m_values.device)[:, None]
        kv_positions = torch.arange(kv_length, device=self.slope_m_values.device)[None, :]
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(query_positions, kv_positions)

        return purely_negative_distance_matrix * self.slope_m_values

class ALiBiCustomEmbedding(ALiBiEmbedding):
    """
    Applies ALiBi-based attention bias using custom-defined distances.
    """

    def forward(self, alibi_query_values: torch.Tensor, alibi_key_values: torch.Tensor, value_to_not_apply_linear_bias_toward: int = None, **kwargs) -> torch.Tensor:
        """
        Computes the ALiBi bias tensor using custom query and key values.
        Args:
            custom_query_values (torch.Tensor): Custom distance values for queries.
            custom_key_values (torch.Tensor): Custom distance values for keys.
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
