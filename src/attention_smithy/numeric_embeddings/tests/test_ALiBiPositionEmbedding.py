import pytest
import re
import torch
from attention_smithy.numeric_embeddings import ALiBiPositionEmbedding
import warnings

def test__ALiBiPositionEmbedding__general_test():
    num_heads = 2
    query_length = 5
    kv_length = 5
    embedding = ALiBiPositionEmbedding(number_of_heads=num_heads)
    query = torch.randn(1, num_heads, query_length, 1)
    key = torch.randn(1, num_heads, kv_length, 1)
    output = embedding(query=query, key=key)
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

def test__ALiBiPositionEmbedding__slopeDegree4():
    num_heads = 2
    query_length = 5
    kv_length = 5
    embedding = ALiBiPositionEmbedding(number_of_heads=num_heads, slope_degree=4)
    query = torch.randn(1, num_heads, query_length, 1)
    key = torch.randn(1, num_heads, kv_length, 1)
    output = embedding(query=query, key=key)

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

def test__ALiBiPositionEmbedding__query_and_kv_lengths_differ__no_cross_attention_allowed_throws_warning():
    num_heads = 2
    query_length = 4
    kv_length = 5
    embedding = ALiBiPositionEmbedding(number_of_heads=num_heads, slope_degree=4)
    warningOutput = "ALiBi position encoding used, but not enabled for cross attention by default. Set `allow_cross_attention=True` in initiliazation of ALiBiPositionEmbedding if you want to change this behavior (not recommended)."
    with pytest.warns(UserWarning, match=re.escape(warningOutput)):
        query = torch.randn(1, num_heads, query_length, 1)
        key = torch.randn(1, num_heads, kv_length, 1)
        embedding(query=query, key=key)


def test__ALiBiPositionEmbedding__query_and_kv_lengths_differ__cross_attention_allowed_does_not_throw_warning():
    num_heads = 2
    query_length = 4
    kv_length = 5
    embedding = ALiBiPositionEmbedding(number_of_heads=num_heads, slope_degree=4, allow_cross_attention=True)
    with warnings.catch_warnings(record=True) as w:
        query = torch.randn(1, num_heads, query_length, 1)
        key = torch.randn(1, num_heads, kv_length, 1)
        embedding(query=query, key=key)
        assert len(w) == 0
