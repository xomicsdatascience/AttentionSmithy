import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy
from attention_smithy.generators.GreedyGenerator import GreedyGenerator
from attention_smithy.generators.BeamGenerator import BeamGenerator

class GeneratorContext:
    """

    """
    def __init__(self,
                 method: str = "greedy",
                 src_embedding: torch.Tensor = None,
                 maximum_sequence_length: int = 1000,
                 **kwargs,
                 ) -> None:
        """
        Args:
            method (str): A string indicating the type of generator to be used.
                Options are currently "greedy" search and "beam" search.
            src_embedding (torch.Tensor): For encoder-decoder models, an input
                is encoded a single time while the decoder is iterated over
                multiple times. This allows for users to first encode an input
                and provide it for repeated reference by the decoder. Of size
                (batch_size, src_sequence_length, embedding_dimension)
        Attributes:
            _strategy (GeneratorStrategy): A child of the GeneratorStrategy
                abstract class.
            src_embedding (torch.Tensor): See description above.
        """
        if method == "greedy":
            self._strategy = GreedyGenerator(**kwargs)
        if method == "beam":
            self._strategy = BeamGenerator(**kwargs)
        self.src_embedding = src_embedding
        self.maximum_sequence_length = maximum_sequence_length

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        **kwargs,
    ):
        """
        Args:
            tgt_input (torch.Tensor): An initial target input of token IDs, of shape
                (1, tgt_sequence_length). If there is no provided initial target
                input, this value should be torch.Tensor([[start_token]]).
        Returns:
            torch.Tensor: 1D sequence of token integers.
        """
        return self._strategy.generate_sequence(
            model,
            end_token,
            tgt_input,
            src_embedding=self.src_embedding,
            maximum_sequence_length=self.maximum_sequence_length,
            **kwargs,
        )
