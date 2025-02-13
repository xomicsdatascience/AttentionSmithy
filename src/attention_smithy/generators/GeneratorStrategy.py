import torch
from torch import nn
from abc import ABC, abstractmethod
from collections import defaultdict

class GeneratorStrategy(ABC):
    """
    See GeneratorContext class.
    """
    def __init__(self,
                 no_repeat_ngram_size:int = 0,
                 **kwargs,
                 ) -> None:
        """
        Args:
            no_repeat_ngram_size (int): The "n" of ngrams. Generator algorithms can sometimes
                be caught in endlessly repeating loops, generating the same sentence or phrase
                perpetually. This parameter is part of an algorithm designed to identify and
                ban repeat segments. It determines the length of tokens that the
                repeat-identifying algorithm will focus on. If set to 1, every individual token
                must be unique. If set to 2, any 2-token combination must be unique. And
                so on. The functions that use this parameter are defined below and used in
                all generator strategy child classes.
        """

        self.no_repeat_ngram_size = no_repeat_ngram_size

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

    def _apply_ngram_repeating_restraints(self, outputs, log_probabilities):
        if self.no_repeat_ngram_size == 0:
            return
        batch_size = outputs.shape[0]
        for i in range(batch_size):
            sample = outputs[i]
            if len(sample) < self.no_repeat_ngram_size:
                continue
            n_minus_1_sequence_to_next_token_dictionary = self._make_n_minus_1_sequence_to_next_token_dictionary(sample, self.no_repeat_ngram_size)
            n_minus_1_sequence = tuple([int(x) for x in sample[-self.no_repeat_ngram_size+1:]])
            avoided_values = n_minus_1_sequence_to_next_token_dictionary[n_minus_1_sequence]
            for value in avoided_values:
                log_probabilities[i, value] = float('-inf')

    def _make_n_minus_1_sequence_to_next_token_dictionary(self, tensor, n):
        n_minus_1_sequence_to_next_token_dictionary = defaultdict(set)
        for i in range(n, len(tensor)):
            key = tuple([int(x) for x in tensor[i-n+1:i]])
            n_minus_1_sequence_to_next_token_dictionary[key].add(int(tensor[i]))
        return n_minus_1_sequence_to_next_token_dictionary

