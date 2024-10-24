import torch

class _NoAddEmbedding:
    """
    Takes any input an returns 0.
    """
    def __call__(self, *args, **kwargs):
        return 0

class _PassthroughEmbedding:
    """
    Takes any input and returns the input.
    """
    def __call__(self, x, *args, **kwargs):
        return x

class NumericEmbeddingFacade:
    """
    NumericEmbeddingFacade keeps all numeric-based encoding in a single accessible location.

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
                 sinusoidalPosition=_NoAddEmbedding(),
                 sinusoidalCustom=_NoAddEmbedding(),
                 learnedPosition=_NoAddEmbedding(),
                 rotaryPosition=_PassthroughEmbedding(),
                 alibiPosition=_NoAddEmbedding(),
                 alibiCustom=_NoAddEmbedding(),
                 ):
        """
        Args:
            sinusoidalPosition: SinusoidalPositionEmbedding() instance. If not set,
                will just return 0.
            sinusoidalCustom: SinusoidalCustomEmbedding() instance. If not set,
                will just return 0.
            learnedPosition: LearnedEmbedding() instance. If not set,
                will just return 0.
            rotaryPosition: RotaryEmbedding() instance. If not set,
                will return the given input unchanged.
            alibiPosition: ALiBiPositionEmbedding() instance. If not set,
                will just return 0.
            alibiCustom: ALiBiCustomEmbedding() instance. If not set,
                will just return 0.

        """
        self.sinusoidalPosition = sinusoidalPosition
        self.sinusoidalCustom = sinusoidalCustom
        self.learnedPosition = learnedPosition
        self.rotaryPosition = rotaryPosition
        self.alibiPosition = alibiPosition
        self.alibiCustom = alibiCustom

    def calculate_sinusoidal_and_learned_tokenizations(self, x, sinusoidal_custom_values=None, learned_values=None, **kwargs):
        output = torch.zeros_like(x)
        output += self.sinusoidalPosition(x)
        output += self.sinusoidalCustom(x, sinusoidal_custom_values)
        output += self.learnedPosition(learned_values)
        return output

    def apply_rotation_to_matrix(self, matrix):
        return self.rotaryPosition(matrix)

    def calculate_alibi_attention_score_distances(self, query, key, alibi_query_values=None, alibi_key_values=None, **kwargs):
        batch_size, num_heads, query_sequence_length, _ = query.shape
        _, _, key_sequence_length, _ = key.shape
        output = torch.zeros((batch_size, num_heads, query_sequence_length, embedding_dimension))
        output += self.alibiPosition(query_sequence_length, key_sequence_length)
        output += self.alibiCustom(alibi_query_values, alibi_key_values)
        return output







