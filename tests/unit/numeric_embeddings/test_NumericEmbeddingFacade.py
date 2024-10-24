import pytest
import torch
from attention_smithy.numeric_embeddings.NumericEmbeddingFacade import _NoAddEmbedding, _PassthroughEmbedding

@pytest.fixture
def input_tensor():
    batch_size = 32
    sequence_length = 20
    embedding_dimension = 50
    input_tensor = torch.rand((batch_size, sequence_length, embedding_dimension))
    return input_tensor

def test__NumericEmbeddingFacade__no_add_types_return_nothing(input_tensor):
    noAddEmbedding = _NoAddEmbedding()
    output_tensor = input_tensor + noAddEmbedding(input_tensor)
    assert torch.allclose(input_tensor, output_tensor)

def test__NumericEmbeddingFacade__passthrough_returns_tensor_with_no_change(input_tensor):
    input_tensor_copy = input_tensor.clone()
    passthroughEmbedding = _PassthroughEmbedding()
    output_tensor = passthroughEmbedding(input_tensor)
    assert torch.allclose(input_tensor_copy, output_tensor)



