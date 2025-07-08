import torch
from torch import nn
from abc import ABC, abstractmethod

class GeneratorModuleAbstractClass(ABC, nn.Module):
    """
    The use of models in the generator context/strategy require the use of a
        "forward_decode" function. Setting a model as a child of this abstract
        class will enforce setting this function.
    """
    @abstractmethod
    def forward_decode(self,
                       tgt_input: torch.Tensor,
                       **kwargs,
                       ) -> torch.Tensor:
        """
        Args:
            tgt_input (torch.Tensor): Input tensor to be decoded, of shape
                (batch_size, sequence_length, embedding_dimension)
        Returns:
            torch.Tensor: Logits for predicting next tokens, of shape
                (batch_size, sequence_length, vocab_size)
        """
        pass