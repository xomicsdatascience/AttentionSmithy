import torch
from torch import nn
import pytest
from attention_smithy.generators import GeneratorContext, GeneratorModuleAbstractClass

@pytest.fixture
def generator_context():
    return GeneratorContext(method="greedy")

@pytest.fixture
def vocab_size():
    return 10

@pytest.fixture
def dummy_model(vocab_size):
    class DummyGreedyModel(GeneratorModuleAbstractClass):
        def __init__(self, vocab_size):
            super(DummyGreedyModel, self).__init__()
            self.token_embedding = nn.Embedding(vocab_size, vocab_size)
            self.token_embedding.weight.data = torch.tensor([
            #    0    1    2    3    4    5    6    7    8    9
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # token 0 (pad)
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # token 1 (start)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # token 2 (end)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # token 3
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # token 4
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # token 5
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # token 6
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # token 7
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # token 8
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # token 9
            ])

        def forward_decode(self, input_tokens):
            embedded_tokens = self.token_embedding(input_tokens)
            return embedded_tokens
    return DummyGreedyModel(vocab_size)

@pytest.fixture
def start_token():
    return 1

@pytest.fixture
def end_token():
    return 2

@pytest.fixture
def expected_output(start_token, end_token):
    return torch.tensor([[start_token, 4,  9,  3,  6,  5,  8,  7, end_token]])

@pytest.fixture
def tgt_input_tensor(start_token):
    return torch.tensor([[start_token]])

def test__GreedyGenerator(generator_context, dummy_model, tgt_input_tensor, end_token, expected_output):
    with torch.no_grad():
        output = generator_context.generate_sequence(dummy_model, end_token, tgt_input_tensor)
        assert torch.allclose(output, expected_output)