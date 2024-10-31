import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy
from attention_smithy.generators.GreedyGenerator import GreedyGenerator
from attention_smithy.generators.BeamGenerator import BeamGenerator

class GeneratorContext:
    """
    A sequence generator class. When provided a model designed to output a sequence
        of tokens (often words, as in machine translation or GPT models), this class
        will generate that sequence. This class encorporates several strategies for
        sequence generation, and allows the user to decide exactly which method is used.

    This is part of an implementation of the Strategy Design Pattern. The Strategy
        design pattern is a way of organizing code that allows you to swap out
        different approaches to solving a problem without changing the underlying
        code. In this case, the end user can switch between the greedy and beam
        search generator methods with a simple parameter change. Furthermore, if new
        methods are added in the future, it does not increase the difficulty of the end
        user experience - they are just added as another strategy option. They don't need
        to worry about which classes to write or update to perform the task they want,
        as that is all taken care of by the Context class. The Strategy class
        enforces a blueprint for specific methodologies (greedy, beam) so that they
        can be accessed by the context class in a standardized way.
    """
    def __init__(self,
                 method: str = "greedy",
                 maximum_sequence_length: int = 1000,
                 **kwargs,
                 ) -> None:
        """
        Args:
            method (str): A string indicating the type of generator to be used.
                Options are currently "greedy" search and "beam" search.
            maximum_sequence_length (int): The maximum number of tokens that can
                be in the generated output sequence. Note that, if the target
                input value already has non-start-token values, the generator
                will only add the difference between what already exists and the
                maximum_sequence_length.
        Attributes:
            _strategy (GeneratorStrategy): A child of the GeneratorStrategy
                abstract class.
            maximum_sequence_length (int): See description above.
        """
        if method == "greedy":
            self._strategy = GreedyGenerator(**kwargs)
        if method == "beam":
            self._strategy = BeamGenerator(**kwargs)
        self.maximum_sequence_length = maximum_sequence_length

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            model (nn.Module, GeneratorModuleAbstractClass):
                An instance of the decoder model. Note the existence
                of the GeneratorModuleAbstractClass, which as a parent class
                ensures the existence of functionality required to be run through
                a generator (like the forward_decode function).
            end_token (int): The end token integer representation. Generally '2'.
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
            maximum_sequence_length=self.maximum_sequence_length,
            **kwargs,
        )
