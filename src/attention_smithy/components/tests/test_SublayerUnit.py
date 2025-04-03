import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F
from attention_smithy.components import SublayerUnit

class DummyPassthroughSublayer(nn.Module):
    def forward(self, x):
        return x.clone()

@pytest.fixture
def dummy_sublayer():
    return DummyPassthroughSublayer()

@pytest.fixture
def embedding_dimension():
    return 10

@pytest.fixture
def dropout():
    return 0.0

@pytest.fixture
def random_input(embedding_dimension):
    batch_size = 32
    sequence_length = 3
    return torch.rand((batch_size, sequence_length, embedding_dimension))

def fix_norm(norm: nn.LayerNorm):
    """
    Fix the LayerNorm parameters to ensure determinism.
    """
    nn.init.constant_(norm.weight, 1.0)
    nn.init.constant_(norm.bias, 0.0)

def test__SublayerUnit__postnorm_adds_output_of_sublayer_to_input(dummy_sublayer, embedding_dimension, dropout,
                                                                  random_input):
    """
    For post-normalization, the sublayer unit computes:
        output = LayerNorm(x + sublayer(x))
    Since the dummy sublayer returns its input, expected output is:
        expected = LayerNorm(x + x)
    """
    sublayer_unit = SublayerUnit(dummy_sublayer, embedding_dimension, dropout, use_prenorm=False)
    fix_norm(sublayer_unit.norm)

    expected = sublayer_unit.norm(random_input + random_input)
    output = sublayer_unit(random_input)

    assert torch.allclose(output, expected, atol=1e-4)


def test__SublayerUnit__prenorm_adds_output_of_sublayer_to_input(dummy_sublayer, embedding_dimension, dropout,
                                                                 random_input):
    """
    For pre-normalization, the sublayer unit computes:
        output = x + sublayer(LayerNorm(x))
    With the dummy sublayer, this reduces to:
        expected = x + LayerNorm(x)
    """
    sublayer_unit = SublayerUnit(dummy_sublayer, embedding_dimension, dropout, use_prenorm=True)
    fix_norm(sublayer_unit.norm)

    expected = random_input + sublayer_unit.norm(random_input)
    output = sublayer_unit(random_input)

    assert torch.allclose(output, expected, atol=1e-4)

