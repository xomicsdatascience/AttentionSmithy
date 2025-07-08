import torch
from typing import Union
from attention_smithy.numeric_embeddings.abstract_embedding_strategies import (
    NumericEmbeddingStrategyBase,
    MatrixModificationStrategyBase,
    AttentionBiasStrategyBase,
)
from torch import nn
from typing import Union, List

class NumericEmbeddingManager(nn.Module):
    """
    NumericEmbeddingManager keeps all numeric-based encoding in a single accessible location.

    Multiple positional embedding strategies have been proposed, and are often addressed
        interchangably. For example, you might use sinusoidal positional embeddings, or
        you might use rotary positional embeddings. It is easy to address this distinction
        in speech. However, in code, these positional embedding methods differ wildly in
        implementation. This is often the case with any numeric encoding, positional or
        otherwise.

    This class serves to promote a psuedo "interchangability" for experimentation
        with different numeric encoding types. The idea is to determine at the start of
        the program which numeric encodings the programmer wants to use, and they are
        implemented at their appropriate location. This is mostly used for position,
        but if the programmer wants to apply positional techniques to custom numeric
        values instead, that option is available as well.
    """
    def __init__(
        self,
        embedding_strategies: List[Union[NumericEmbeddingStrategyBase, MatrixModificationStrategyBase, AttentionBiasStrategyBase]]
    ) -> None:
        """
        Attributes:
            embedding_strategies:
                List[Union[NumericEmbeddingStrategyBase, MatrixModificationStrategyBase, AttentionBiasStrategyBase]]
                A list of numeric embedding strategies, primarily used for position encodings. This manager class
                will call the requisite strategies at the appropriate times.
        """
        super().__init__()
        for i, strategy in enumerate(embedding_strategies):
            setattr(self, f"embedding_strategy_{i}", strategy)

    def _get_strategies_by_type(self, strategy_type):
        """Helper function to find strategies of a specific type dynamically."""
        return [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("embedding_strategy_") and isinstance(getattr(self, attr), strategy_type)
        ]

    def create_positional_or_custom_embedding(self, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(kwargs["token_embedding"])
        for embedding_strategy in self._get_strategies_by_type(NumericEmbeddingStrategyBase):
            output += embedding_strategy.create_positional_or_custom_embedding(**kwargs)
        return output

    def modify_matrix(self, target_matrix: torch.Tensor, **kwargs) -> torch.Tensor:
        for embedding_strategy in self._get_strategies_by_type(MatrixModificationStrategyBase):
            target_matrix = embedding_strategy.modify_matrix(target_matrix, **kwargs)
        return target_matrix

    def create_bias_tensor(self, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(kwargs["attention_score_matrix"])
        for embedding_strategy in self._get_strategies_by_type(AttentionBiasStrategyBase):
            output += embedding_strategy.create_bias_tensor(**kwargs)
        return output







