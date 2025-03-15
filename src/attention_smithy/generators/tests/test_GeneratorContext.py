import pytest
import torch
from torch import nn
from attention_smithy.generators import GeneratorContext


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test__GeneratorContext__raises_error_for_unknown_strategy():
    """
    Test that GeneratorContext raises a ValueError when initialized with
    an unknown generation strategy.
    """
    with pytest.raises(ValueError) as excinfo:
        generator = GeneratorContext(method="unknown_strategy")

    assert "Unknown generation method: 'unknown_strategy'" in str(excinfo.value)
    assert "Available methods are 'greedy' and 'beam'" in str(excinfo.value)


def test__GeneratorContext__accepts_valid_strategies():
    """
    Test that GeneratorContext accepts the valid strategies without errors.
    """
    try:
        greedy_generator = GeneratorContext(method="greedy")
        beam_generator = GeneratorContext(method="beam")
    except ValueError:
        pytest.fail("GeneratorContext raised ValueError for valid strategy")