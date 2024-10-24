import math
import torch

class ALiBiEmbedding:
    def __init__(self, num_heads, slope_degree=2):
        self.slope_m_values = self._get_slopes(num_heads=num_heads, slope_degree=slope_degree)

    def _determine_negative_distance_matrix(self, query_values, kv_values):
        distance_matrix = query_values - kv_values
        purely_negative_distance_matrix = torch.where(distance_matrix > 0, -distance_matrix, distance_matrix)
        return purely_negative_distance_matrix

    def _get_slopes(self, num_heads, slope_degree):
        slopes = (1 / slope_degree) ** torch.arange(1, num_heads + 1)
        return slopes.view(num_heads, 1, 1)

class ALiBiPositionEmbedding(ALiBiEmbedding):
    def __call__(self, query_length, kv_length):
        if query_length != kv_length:
            raise ValueError(f"ALiBi Position Embedding failed. Query and Key sequence length must be identical, as in self-attention. Query length: {query_length}, Key length: {kv_length}")
        query_positions = torch.arange(query_length)[:, None]
        kv_positions = torch.arange(kv_length)[None, :]
        purely_negative_distance_matrix = self._determine_negative_distance_matrix(query_positions, kv_positions)
        return purely_negative_distance_matrix * self.slope_m_values

