import torch
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingTorch

class RotaryPositionEmbedding:
    """
    A class that adjusts the query/key matrices to reflect position.
        Performed by breaking a token embedding into groups of 2, treats
        each group as a set of coordinates, then incrementally adjusts the angle of those
        coordinates, with variance depending on the position of the token.
    WHEN APPLIED: After query/key/values are multiplied by their respective weights
        in attention, but before attention scores are calculated.
    NOTE: This is largely performed with an external package, rotary_embedding_torch.
    """

    def __init__(self,
                 head_dimension: int
                 ) -> None:
        """
        Initializes the embedding class.

        Args:
            head_dimension (int): The expected dimension size for each head.
        """

        self.rotary = RotaryEmbeddingTorch(dim=head_dimension)

    def __call__(self,
                 input: torch.Tensor,
                 *args,
                 **kwargs
                 ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): An input query or key watrix, of shape
                (batch_size, sequence_length, embedding_dimension)
        Returns:
            torch.Tensor: Adjusted output, of shape
                (batch_size, sequence_length, embedding_dimension)
        """
        return self.rotary.rotate_queries_or_keys(input)
