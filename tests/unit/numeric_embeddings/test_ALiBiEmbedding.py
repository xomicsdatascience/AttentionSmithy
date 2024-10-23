import pytest
import re
import torch
from attention_smithy.numeric_embeddings import ALiBiEmbedding

def test__ALiBiEmbedding__general_test():
    num_heads = 2
    embedding = ALiBiEmbedding(num_heads=num_heads)
    output = embedding(query_length=5, kv_length=5)

    expected_output = torch.tensor([
        [
            [ 0.0, -0.5, -1.0, -1.5, -2.0],
            [-0.5,  0.0, -0.5, -1.0, -1.5],
            [-1.0, -0.5,  0.0, -0.5, -1.0],
            [-1.5, -1.0, -0.5,  0.0, -0.5],
            [-2.0, -1.5, -1.0, -0.5,  0.0],
        ],
        [
            [ 0.00, -0.25, -0.50, -0.75, -1.00],
            [-0.25,  0.00, -0.25, -0.50, -0.75],
            [-0.50, -0.25,  0.00, -0.25, -0.50],
            [-0.75, -0.50, -0.25,  0.00, -0.25],
            [-1.00, -0.75, -0.50, -0.25,  0.00],
        ],
    ])
    assert torch.allclose(output, expected_output)

def test__ALiBiEmbedding__slopeDegree4():
    num_heads = 2
    embedding = ALiBiEmbedding(num_heads=num_heads, slope_degree=4)
    output = embedding(query_length=5, kv_length=5)
    expected_output = torch.tensor([
        [
            [ 0.00, -0.25, -0.50, -0.75, -1.00],
            [-0.25,  0.00, -0.25, -0.50, -0.75],
            [-0.50, -0.25,  0.00, -0.25, -0.50],
            [-0.75, -0.50, -0.25,  0.00, -0.25],
            [-1.00, -0.75, -0.50, -0.25,  0.00],
        ],
        [
            [ 0.0000, -0.0625, -0.1250, -0.1875, -0.2500],
            [-0.0625,  0.0000, -0.0625, -0.1250, -0.1875],
            [-0.1250, -0.0625,  0.0000, -0.0625, -0.1250],
            [-0.1875, -0.1250, -0.0625,  0.0000, -0.0625],
            [-0.2500, -0.1875, -0.1250, -0.0625,  0.0000],
        ],
    ])
    assert torch.allclose(output, expected_output)

def test__ALiBiEmbedding__query_and_kv_lengths_differ_throws_value_error():
    query_length = 4
    kv_length = 5
    embedding = ALiBiEmbedding(num_heads=2, slope_degree=4)
    errorOutput = f"ALiBi Embedding failed. Query and Key sequence length must be identical, as in self-attention. Query length: {query_length}, Key length: {kv_length}"
    with pytest.raises(ValueError, match=re.escape(errorOutput)):
        embedding(query_length=query_length, kv_length=kv_length)


