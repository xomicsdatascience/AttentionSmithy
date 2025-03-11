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

def test__SublayerUnit__adds_output_of_sublayer_to_input(dummy_sublayer, embedding_dimension, dropout, random_input):
    sublayer_unit = SublayerUnit(dummy_sublayer, embedding_dimension, dropout)
    output = sublayer_unit(random_input)
    layerNorm = nn.LayerNorm(embedding_dimension)
    expected_output = layerNorm(random_input + random_input)
    assert torch.allclose(output, expected_output, atol=1e-4)

