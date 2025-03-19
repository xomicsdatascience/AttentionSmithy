import torch
from torch import nn
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import NumericEmbeddingStrategyBase

class SinusoidalEmbedding(NumericEmbeddingStrategyBase):
    """
    A class that encodes numeric values as tokens. It relies on sinusoidal equations described in the original
        "Attention Is All You Need" paper.
    WHEN APPLIED: Generally added to token embeddings at the beginning of the model.
    NOTE: These numeric encodings typically encode position (0, 1, 2 etc.). However, it could be used to
        encode custom numeric values as well. Thus there are two child classes in this file depending
        on the desired use case.
    """

    def __init__(self, embedding_dimension: int) -> None:
        """
        Args:
            embedding_dimension (int): The expected dimension size for each token embedding.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension

    def _create_exponent(self, device):
        return torch.arange(0, self.embedding_dimension, 2, device=device) / self.embedding_dimension

    def _find_sin_and_cos_embeddings_of_given_values(self, values):
        exponent = self._create_exponent(values.device)
        output_values = values / (10_000 ** exponent)
        return torch.sin(output_values), torch.cos(output_values)

    def create_positional_or_custom_embedding(self, token_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(token_embedding, **kwargs)

class SinusoidalPositionEmbedding(SinusoidalEmbedding):
    """
    Encodes sequence position as a sinusoidal embedding.
    """

    def __init__(self, embedding_dimension: int) -> None:
        """
        Args:
            embedding_dimension (int): The expected dimension size for each token embedding.
        """
        super().__init__(embedding_dimension)

    def forward(self, token_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            token_embedding (torch.Tensor): The embedded input tensor, of shape (batch_size, sequence_length, embedding_dimension)
        Returns:
            torch.Tensor: Positional encoding (sequence_length, embedding_dimension)
        """
        _, sequence_length, _ = token_embedding.shape
        positions = torch.arange(sequence_length, device=token_embedding.device).unsqueeze(1)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(positions)

        positional_encoding = torch.zeros(sequence_length, self.embedding_dimension, device=token_embedding.device)
        positional_encoding[:, 0::2] = sin
        positional_encoding[:, 1::2] = cos

        return positional_encoding