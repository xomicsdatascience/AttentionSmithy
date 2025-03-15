import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingTorch
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import MatrixModificationStrategyBase


class RotaryPositionEmbedding(MatrixModificationStrategyBase):
    """
    A class that adjusts the query/key matrices to reflect position using rotary embeddings.

    This transformation is applied after query/key/values are multiplied by their respective
    weights in attention but before attention scores are calculated.

    NOTE: This implementation uses the external package `rotary_embedding_torch`.
    """

    def __init__(self, head_dimension: int) -> None:
        """
        Initializes the rotary embedding class.

        Args:
            head_dimension (int): The expected dimension size for each attention head.
        """
        super().__init__()
        self.rotary = RotaryEmbeddingTorch(dim=head_dimension)

    def modify_matrix(self, target_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Implements the abstract method from `MatrixModificationStrategyBase` to apply rotary embeddings.

        Args:
            kwargs (dict): Must contain 'target_matrix' (torch.Tensor).

        Returns:
            torch.Tensor: Modified matrix.
        """
        return self.forward(target_matrix)

    def forward(self, target_matrix: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary embeddings to the input matrix (query or key separately).

        Args:
            target_matrix (torch.Tensor): Input query or key matrix of shape
                (batch_size, sequence_length, embedding_dimension).

        Returns:
            torch.Tensor: Adjusted matrix of the same shape.
        """
        return self.rotary.rotate_queries_or_keys(target_matrix)