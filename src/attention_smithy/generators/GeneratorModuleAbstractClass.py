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

'''
I am writing a class that is supposed to generate a sequence from a decoder pytorch model. It should take in an initial series of tokens of size (batch_size, tgt_sequence_length). It will be a batch_size of 1, and generally this initial series will be torch.Tensor([[start_token]]). It will run this input through a model, get the logits of that output, select the best token to append in a greedy fashion. It will then loop through, generating a sequence token by token, until an end token is reached or a maximum limit is reached.
'''