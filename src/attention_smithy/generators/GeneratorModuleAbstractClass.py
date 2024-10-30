import torch
from torch import nn
from abc import ABC, abstractmethod

class GeneratorModuleAbstractClass(ABC, nn.Module):

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