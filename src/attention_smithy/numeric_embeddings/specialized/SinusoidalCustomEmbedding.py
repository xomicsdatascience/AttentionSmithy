from attention_smithy.numeric_embeddings.SinusoidalEmbedding import SinusoidalEmbedding
import torch

class SinusoidalCustomEmbedding(SinusoidalEmbedding):
    """
    Encodes custom numeric input as a sinusoidal embedding.
    """

    def forward(self, token_embedding: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            token_embedding (torch.Tensor): The embedded input tensor, of shape (batch_size, sequence_length, embedding_dimension)
            custom_values (torch.Tensor): The provided custom values to be encoded, of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: Custom value encoding, shape (batch_size, sequence_length, embedding_dimension).
        """
        custom_values = kwargs.get("custom_values")
        custom_encoding = torch.zeros_like(token_embedding)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(custom_values.unsqueeze(2))

        custom_encoding[:, :, 0::2] = sin
        custom_encoding[:, :, 1::2] = cos

        return custom_encoding