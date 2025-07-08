import torch
from torch import nn
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import NumericEmbeddingStrategyBase

class ContinuousValueEmbedding(NumericEmbeddingStrategyBase):
    """
    Encodes continuous scalar values into a high-dimensional embedding space using
    a two-layer MLP with a tanh non-linearity.
    This approach allows numeric values (e.g., scalar features or non-position numbers)
    to be encoded in a learnable, non-linear way.

    See https://doi.org/10.1145/3516367.
    """

    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()
        hidden_dim = int(embedding_dimension ** 0.5)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dimension, bias=False)
        )

    def forward(self, continuous_values: torch.Tensor, **kwargs) -> torch.Tensor:
        continuous_values = continuous_values.unsqueeze(-1)
        return self.net(continuous_values)

    def create_positional_or_custom_embedding(self, continuous_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self(continuous_values, **kwargs)