import torch
from rotary_embedding_torch import RotaryEmbedding as RotaryEmbeddingTorch

class RotaryEmbedding:
    """
    A class that adjusts the query/key matrices to reflect position immediately preceding
        attention calculations. Performed by breaking a token embedding into groups of 2, treats
        each group as a set of coordinates, then incrementally adjusts the angle of those
        coordinates, with variance depending on the position of the token.
    WHEN APPLIED: After query/key/values are multiplied by their respective weights
        in attention, but before attention scores are calculated.
    NOTE: This is largely performed with an external package.
    """

    def __init__(self, embedding_dimension):
        """
        Initializes the embedding class.

        Args:
            embedding_dimension (int): The expected dimension size for each token embedding.
        """

        self.rotary = RotaryEmbeddingTorch(dim=embedding_dimension)

    def __call__(self, input, *args, **kwargs):
        """
        Args:
            input (torch.Tensor): An input query or key watrix, of shape
                (batch_size, sequence_length, embedding_dimension)
        Returns:
            torch.Tensor: Adjusted output, of shape
                (batch_size, sequence_length, embedding_dimension)
        """
        return self.rotary.to(input.device).rotate_queries_or_keys(input)
