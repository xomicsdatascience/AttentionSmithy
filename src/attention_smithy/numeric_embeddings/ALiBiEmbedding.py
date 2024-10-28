import math
import torch

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
                 num_heads: int,
                 slope_degree: int = 2
                 ) -> None:
        """
        Args:
            num_heads (int): The expected number of heads used in multihead attention.
            slope_degree (int, optional): The degree of difference between heads. This
                corresponds to the calculation for "m" in the paper. See "slope_m_values"
                attribute for an example. Defaults to 2 (1/2).
        Attributes:
            slope_m_values (torch.Tensor): A tensor containing a single scale value for
                each head. For 4 heads and a slope_degree of 2, this would be
                [0.5, 0.25, 0.125, 0.0625]. For 2 heads and a slope_degree of 4,
                this would be [0.25, 0.0625]. Of shape (num_heads, 1, 1).
        """

        self.slope_m_values = self._get_slopes(num_heads=num_heads, slope_degree=slope_degree)

    def _determine_negative_distance_matrix(self, query_values, kv_values):
        distance_matrix = query_values - kv_values
        purely_negative_distance_matrix = torch.where(distance_matrix > 0, -distance_matrix, distance_matrix)
        return purely_negative_distance_matrix

    def _get_slopes(self, num_heads, slope_degree):
        slopes = (1 / slope_degree) ** torch.arange(1, num_heads + 1)
        return slopes.view(num_heads, 1, 1)

class ALiBiPositionEmbedding(ALiBiEmbedding):
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
                (num_heads, query_length, kv_length), the last three dimensions
                of the attention_score value in the standard attention method.
                It is broadcast across samples in a batch.
        """
        if query_length != kv_length:
            raise ValueError(f"ALiBi Position Embedding failed. Query and Key sequence length must be identical, as in self-attention. Query length: {query_length}, Key length: {kv_length}")
        query_positions = torch.arange(query_length)[:, None]
        kv_positions = torch.arange(kv_length)[None, :]
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(query_positions, kv_positions)
        return purely_negative_distance_matrix * self.slope_m_values

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
                (batch_size, num_heads, query_length, kv_length), corresponding to
                the exact shape of the attention_score value in the standard attention method.
        """
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(custom_query_values[:, :, None], custom_key_values[:, None, :])
        return purely_negative_distance_matrix[:, None, :, :] * self.slope_m_values[None, :, :, :]