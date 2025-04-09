import torch
import pytest
from attention_smithy.numeric_embeddings import ContinuousValueEmbedding

@pytest.fixture
def model():
    return ContinuousValueEmbedding(embedding_dimension=64)

def test__ContinuousValueEmbedding__output_shape_matches_expected(model):
    numeric_values = torch.randn(8, 10)  # batch_size=8, seq_len=10
    output = model(numeric_values)
    assert output.shape == (8, 10, 64)

def test__ContinuousValueEmbedding__runs_on_different_input_shapes(model):
    for bsz, seq_len in [(1, 1), (4, 7), (16, 32)]:
        numeric_values = torch.randn(bsz, seq_len)
        output = model(numeric_values)
        assert output.shape == (bsz, seq_len, 64)

def test__ContinuousValueEmbedding__has_expected_number_of_parameters():
    model = ContinuousValueEmbedding(embedding_dimension=64)
    params = list(model.parameters())
    assert len(params) == 3
    assert model.net[2].bias is None

def test__ContinuousValueEmbedding__create_embedding_matches_forward(model):
    numeric_values = torch.randn(3, 5)
    output_from_forward = model(numeric_values)
    output_from_create = model.create_positional_or_custom_embedding(numeric_values)
    assert torch.allclose(output_from_forward, output_from_create)

def test__ContinuousValueEmbedding__gradient_flow(model):
    numeric_values = torch.randn(2, 4, requires_grad=True)
    output = model(numeric_values)
    loss = output.sum()
    loss.backward()
    assert numeric_values.grad is not None
    assert all(p.grad is not None for p in model.parameters())