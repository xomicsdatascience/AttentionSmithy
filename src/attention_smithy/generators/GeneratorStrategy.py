import torch
from torch import nn
from abc import ABC, abstractmethod

class GeneratorStrategy(ABC):
    """
    See GeneratorContext class.
    """

    @abstractmethod
    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        See GeneratorContext class.
        """
        pass
