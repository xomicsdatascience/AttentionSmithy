import torch
from torch import nn
from abc import ABC, abstractmethod

class GeneratorStrategy(ABC):
    @abstractmethod
    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        src_embedding: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ):
        pass
