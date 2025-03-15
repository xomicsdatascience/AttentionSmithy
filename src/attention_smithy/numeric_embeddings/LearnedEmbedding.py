import torch
from torch import nn
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import NumericEmbeddingStrategyBase


class LearnedPositionEmbedding(NumericEmbeddingStrategyBase):
    """
    A PyTorch module for learned positional embeddings implementing NumericEmbeddingStrategyBase.
    """

    def __init__(self, max_sequence_length: int, embedding_dimension: int) -> None:
        """
        Args:
            max_sequence_length (int): The maximum sequence length for any input.
            embedding_dimension (int): The token embedding dimension size.
        Attributes:
            embedding (nn.Embedding): A module to embed position indices.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_sequence_length, embedding_dimension)

    def create_positional_or_custom_embedding(self, **kwargs) -> torch.Tensor:
        """
        Generates learned positional embeddings for a given sequence length.

        Args:
            kwargs["token_embedding"] (torch.Tensor): Input tensor (batch_size, sequence_length, embedding_dimension).
        Returns:
            torch.Tensor: Learned positional encoding of shape (sequence_length, embedding_dimension).
        """
        return self.forward(kwargs["token_embedding"])

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        Standard forward function aligning with nn.Module usage.

        Args:
            token_embedding (torch.Tensor): Input tensor (batch_size, sequence_length, embedding_dimension).
        Returns:
            torch.Tensor: Learned positional encoding of shape (sequence_length, embedding_dimension).
        """
        sequence_length = token_embedding.shape[1]
        position_ids = torch.arange(sequence_length, device=self.embedding.weight.device)
        return self.embedding(position_ids)
