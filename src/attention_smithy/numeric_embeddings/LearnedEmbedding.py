import torch
from torch import nn

class LearnedEmbedding(nn.Module):
    """
    A pytorch module for learned positional embeddings.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dimension: int,
                 padding_idx: int = 0
                 ):
        """
        Args:
            vocab_size (int): The number of values in the expected vocabulary.
            embedding_dimension (int): The token embedding dimension size.
            padding_idx (int): The index assigned to ignorable padding tokens.
        Attributes:
            embedding (nn.Embedding): a module to pass position or other sequences
                to for embedding.
        """
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=padding_idx)

    def forward(self,
                x: torch.Tensor,
                ):
        """
        Args:
            x (torch.Tensor): An input tensor of values to be encoded, of shape
                (batch_size, sequence_length).
        Returns:
            torch.Tensor: An output tensor of shape
                (batch_size, sequence_length, embedding_dimension)
        """
        return embedding(x)