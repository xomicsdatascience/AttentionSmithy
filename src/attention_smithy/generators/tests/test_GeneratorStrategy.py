import pytest
import torch
from torch import nn
from collections import defaultdict
from abc import ABC, abstractmethod
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class MockGenerator(GeneratorStrategy):
    """A mock generator class to test n-gram restrictions."""

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ) -> torch.Tensor:
        """Minimal implementation for testing purposes."""
        return tgt_input


@pytest.fixture
def mock_generator():
    return MockGenerator()


def test__GeneratorStrategy__ngram_size_0_allows_repeats(mock_generator):
    """Ensure that when no_repeat_ngram_size=0, repetitions are not blocked."""
    mock_generator.no_repeat_ngram_size = 0

    outputs = torch.tensor([[0, 1, 1, 2, 2, 0]])
    log_probabilities = torch.zeros((1, 5))

    mock_generator._apply_ngram_repeating_restraints(outputs, log_probabilities)

    assert not torch.isinf(log_probabilities).any()


def test__GeneratorStrategy__ngram_size_1_blocks_repeats(mock_generator):
    """Ensure that when no_repeat_ngram_size=1, repeated tokens are blocked."""
    mock_generator.no_repeat_ngram_size = 1

    outputs = torch.tensor([[0, 1, 2, 3]])
    log_probabilities = torch.zeros((1, 5))

    mock_generator._apply_ngram_repeating_restraints(outputs, log_probabilities)

    expected_blocked_tokens = {0, 1, 2, 3}

    for token in expected_blocked_tokens:
        assert log_probabilities[0, token] == float('-inf')


def test__GeneratorStrategy__ngram_size_2_blocks_bigrams(mock_generator):
    """Ensure that when no_repeat_ngram_size=2, repeated bigrams are blocked."""
    mock_generator.no_repeat_ngram_size = 2

    outputs = torch.tensor([[0, 2, 2, 1, 2]])  # Bigram (2,1) and (2,2) should be blocked
    log_probabilities = torch.zeros((1, 5))

    mock_generator._apply_ngram_repeating_restraints(outputs, log_probabilities)

    expected_blocked_tokens = {1, 2}  # 2 (from (2,2)) and 1 (from (2,1))

    for token in expected_blocked_tokens:
        assert log_probabilities[0, token] == float('-inf')