import torch
from typing import Union
from attention_smithy.numeric_embeddings import (
    ALiBiPositionEmbedding,
    ALiBiCustomEmbedding,
    LearnedPositionEmbedding,
    RotaryPositionEmbedding,
    SinusoidalPositionEmbedding,
    SinusoidalCustomEmbedding,
)
from torch import nn


class NoAddEmbedding:
    """
    Takes any input and returns 0.
    """
    def __call__(self, *args, **kwargs):
        return 0

class PassthroughEmbedding:
    """
    Takes any input and returns the input.
    """
    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return x

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
    def __init__(self,
                 sinusoidal_position: Union[NoAddEmbedding, SinusoidalPositionEmbedding] = NoAddEmbedding(),
                 sinusoidal_custom: Union[NoAddEmbedding, SinusoidalCustomEmbedding] = NoAddEmbedding(),
                 learned_position: Union[NoAddEmbedding, LearnedPositionEmbedding] = NoAddEmbedding(),
                 rotary_position: Union[PassthroughEmbedding, RotaryPositionEmbedding] = PassthroughEmbedding(),
                 alibi_position: Union[NoAddEmbedding, ALiBiPositionEmbedding] = NoAddEmbedding(),
                 alibi_custom: Union[NoAddEmbedding, ALiBiCustomEmbedding] = NoAddEmbedding(),
                 ) -> None:
        """
        Args:
            sinusoidal_position: SinusoidalPositionEmbedding() instance. If not set,
                will just return 0.
            sinusoidal_custom: SinusoidalCustomEmbedding() instance. If not set,
                will just return 0.
            learned_position: LearnedPositionEmbedding() instance. If not set,
                will just return 0.
            rotary_position: RotaryPositionEmbedding() instance. If not set,
                will return the given input unchanged.
            alibi_position: ALiBiPositionEmbedding() instance. If not set,
                will just return 0.
            alibi_custom: ALiBiCustomEmbedding() instance. If not set,
                will just return 0.

        """
        super().__init__()
        self.sinusoidal_position = sinusoidal_position
        self.sinusoidal_custom = sinusoidal_custom
        self.learned_position = learned_position
        self.rotary_position = rotary_position
        self.alibi_position = alibi_position
        self.alibi_custom = alibi_custom

    def calculate_sinusoidal_and_learned_tokenizations(self,
                                                       x: torch.Tensor,
                                                       sinusoidal_custom_values: torch.Tensor=None,
                                                       **kwargs
                                                       ) -> torch.Tensor:
        output = torch.zeros_like(x)
        output += self.sinusoidal_position(x)
        output += self.sinusoidal_custom(x, sinusoidal_custom_values)
        output += self.learned_position(x)
        return output

    def apply_rotation_to_query_and_key_matrices(self,
                                                 query: torch.Tensor,
                                                 key: torch.Tensor,
                                                 ) -> torch.Tensor:
        return self.rotary_position(query), self.rotary_position(key)

    def calculate_alibi_attention_score_distances(self,
                                                  query: torch.Tensor,
                                                  key: torch.Tensor,
                                                  alibi_query_values: torch.Tensor = None,
                                                  alibi_key_values: torch.Tensor = None,
                                                  **kwargs
                                                  ) -> torch.Tensor:
        batch_size, num_heads, query_sequence_length, _ = query.shape
        _, _, key_sequence_length, _ = key.shape
        output = torch.zeros((batch_size, num_heads, query_sequence_length, key_sequence_length), device=query.device)
        output += self.alibi_position(query_sequence_length, key_sequence_length)
        output += self.alibi_custom(alibi_query_values, alibi_key_values)
        return output







