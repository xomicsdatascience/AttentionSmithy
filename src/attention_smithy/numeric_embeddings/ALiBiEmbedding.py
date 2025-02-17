import math
import torch
import warnings


class ALiBiEmbedding:
    """
    Attention with Linear Biases (ALiBi) embedding class using a technique described in the paper
        "TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION".
        The intention is to not encode numeric (position) values directly, but rather adjust
        attention scores to prioritize attending to nearby tokens. The exact distance prioritization
        explicitly differs across heads in multihead attention to enable varying ranges of attention.
    WHEN APPLIED: After attention scores have been computed, but before conversion to probablities
        via softmax.
    NOTE: ALiBi is generally used as a positional encoding method. However, it can apply to
        customized distances as well. Thus there are two child classes in this file depending
        on the desired use case.
    """

    def __init__(self,
                 number_of_heads: int,
                 slope_degree: int = 2,
                 ) -> None:
        """
        Args:
            number_of_heads (int): The expected number of heads used in multihead attention.
            slope_degree (int, optional): The degree of difference between heads. This
                corresponds to the calculation for "m" in the paper. See "slope_m_values"
                attribute for an example. Defaults to 2 (1/2).
        Attributes:
            slope_m_values (torch.Tensor): A tensor containing a single scale value for
                each head. For 4 heads and a slope_degree of 2, this would be
                [0.5, 0.25, 0.125, 0.0625]. For 2 heads and a slope_degree of 4,
                this would be [0.25, 0.0625]. Of shape (number_of_heads, 1, 1).
        """

        self.slope_m_values = self._get_slopes(number_of_heads=number_of_heads, slope_degree=slope_degree)

    def _determine_negative_distance_matrix(self, query_values, kv_values):
        distance_matrix = query_values - kv_values
        purely_negative_distance_matrix = torch.where(distance_matrix > 0, -distance_matrix, distance_matrix)
        return purely_negative_distance_matrix

    def _get_slopes(self, number_of_heads, slope_degree):
        slopes = (1 / slope_degree) ** torch.arange(1, number_of_heads + 1)
        return slopes.view(number_of_heads, 1, 1)

class ALiBiPositionEmbedding(ALiBiEmbedding):
    def __init__(self,
                 number_of_heads: int,
                 slope_degree: int = 2,
                 allow_cross_attention: bool = False,
                 ) -> None:
        """
        Args:
            number_of_heads (int): See parent class.
            slope_degree (int, optional): See parent class.
            allow_cross_attention (bool, optional): Determines whether or not alibi position embedding should apply
                to cross attention blocks (where the query and key sequence lengths are different). To the author's
                knowledge, allowing this is untested and likely to end in error, but the option is available for future
                testing.
        Attributes:
            slope_m_values (torch.Tensor): A tensor containing a single scale value for
                each head. For 4 heads and a slope_degree of 2, this would be
                [0.5, 0.25, 0.125, 0.0625]. For 2 heads and a slope_degree of 4,
                this would be [0.25, 0.0625]. Of shape (number_of_heads, 1, 1).
        """

        super().__init__(number_of_heads, slope_degree)
        self.allow_cross_attention = allow_cross_attention
        self.has_warned_that_cross_attention_was_skipped = False

    def __call__(self,
                 query_length: int,
                 kv_length: int,
                 ) -> torch.Tensor:
        """
        Args:
            query_length (int): The sequence length of the query matrix
            kv_length (int): The sequence length of the key (and, technically, value)
                matrix.
        Raises:
            ValueError: ALiBi positional embeddings address self-attention
                mechanisms. This error may be removed if cross-attention applications
                become obvious.
        Returns:
            torch.Tensor: A series of query-by-key matrices, one for each head.
                No positive values exist in these matrices - the idea is to pay
                "less" attention to tokens that are farther away. How far depends
                on the head. Applies across samples in a batch. Of shape
                (number_of_heads, query_length, kv_length), the last three dimensions
                of the attention_score value in the standard attention method.
                It is broadcast across samples in a batch.
        """
        if query_length != kv_length and not self.allow_cross_attention:
            warnings.warn(
                'ALiBi position encoding used, but not enabled for cross attention by default. Set `allow_cross_attention=True` in initiliazation of ALiBiPositionEmbedding if you want to change this behavior (not recommended).')
        if query_length != kv_length and self.allow_cross_attention:
                return torch.zeros((query_length, kv_length))
        query_positions = torch.arange(query_length)[:, None]
        kv_positions = torch.arange(kv_length)[None, :]
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(query_positions, kv_positions)
        return purely_negative_distance_matrix.to(self.slope_m_values.device) * self.slope_m_values

class ALiBiCustomEmbedding(ALiBiEmbedding):
    def __call__(self,
                 custom_query_values: torch.Tensor,
                 custom_key_values: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Args:
            custom_query_values (torch.Tensor): The values corresponding to the query matrix that
                should denote customized "distance" from other tokens, of shape (batch_size, query_length).
            custom_key_values (torch.Tensor): The values corresponding to the key matrix that
                should denote customized "distance" from other tokens, of shape (batch_size, kv_length).
        Returns:
            torch.Tensor: see ALiBiPositionEmbedding __call__ return value. The
                only distinction is that this output is of shape
                (batch_size, number_of_heads, query_length, kv_length), corresponding to
                the exact shape of the attention_score value in the standard attention method.
        """
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(custom_query_values[:, :, None], custom_key_values[:, None, :])
        device = custom_query_values.device
        return purely_negative_distance_matrix[:, None, :, :] * self.slope_m_values[None, :, :, :]