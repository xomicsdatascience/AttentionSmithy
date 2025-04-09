import torch
from attention_smithy.numeric_embeddings import ALiBiCustomEmbedding

def test__ALiBiCustomEmbedding__general_test():
    embedding = ALiBiCustomEmbedding(number_of_heads=1)
    query_values = torch.tensor([
        [1.0, 3.0, 4.0, 2.0, 0.0]
    ])
    key_values = torch.tensor([
        [2.0, 0.0, 4.0, 1.0, 3.0]
    ])
    expected_output = torch.tensor([
        [-0.5, -0.5, -1.5,  0.0, -1.0],
        [-0.5, -1.5, -0.5, -1.0,  0.0],
        [-1.0, -2.0,  0.0, -1.5, -0.5],
        [ 0.0, -1.0, -1.0, -0.5, -0.5],
        [-1.0,  0.0, -2.0, -0.5, -1.5],
    ])[None, None, :, :]
    output = embedding(query_values, key_values, None)
    assert torch.allclose(output, expected_output)

def test__ALiBiCustomEmbedding__numHeads2():
    embedding = ALiBiCustomEmbedding(number_of_heads=2)
    query_values = torch.tensor([
        [1.0, 3.0, 4.0, 2.0, 0.0]
    ])
    key_values = torch.tensor([
        [2.0, 0.0, 4.0, 1.0, 3.0]
    ])
    expected_output = torch.tensor([
        [
            [-0.5, -0.5, -1.5,  0.0, -1.0],
            [-0.5, -1.5, -0.5, -1.0,  0.0],
            [-1.0, -2.0,  0.0, -1.5, -0.5],
            [ 0.0, -1.0, -1.0, -0.5, -0.5],
            [-1.0,  0.0, -2.0, -0.5, -1.5],
        ],
        [
            [-0.25, -0.25, -0.75,  0.00, -0.50],
            [-0.25, -0.75, -0.25, -0.50,  0.00],
            [-0.50, -1.00,  0.00, -0.75, -0.25],
            [ 0.00, -0.50, -0.50, -0.25, -0.25],
            [-0.50,  0.00, -1.00, -0.25, -0.75],
        ],
    ])[None, :, :, :]
    output = embedding(query_values, key_values, None)
    assert torch.allclose(output, expected_output)

def test__ALiBiCustomEmbedding__numHeads2__slopeDegree4():
    embedding = ALiBiCustomEmbedding(number_of_heads=2, slope_degree=4)
    query_values = torch.tensor([
        [1.0, 3.0, 4.0, 2.0, 0.0]
    ])
    key_values = torch.tensor([
        [2.0, 0.0, 4.0, 1.0, 3.0]
    ])
    expected_output = torch.tensor([
        [
            [-0.25, -0.25, -0.75,  0.00, -0.50],
            [-0.25, -0.75, -0.25, -0.50,  0.00],
            [-0.50, -1.00,  0.00, -0.75, -0.25],
            [ 0.00, -0.50, -0.50, -0.25, -0.25],
            [-0.50,  0.00, -1.00, -0.25, -0.75],
        ],
        [
            [-0.0625, -0.0625, -0.1875,  0.0000, -0.1250],
            [-0.0625, -0.1875, -0.0625, -0.1250,  0.0000],
            [-0.1250, -0.2500,  0.0000, -0.1875, -0.0625],
            [ 0.0000, -0.1250, -0.1250, -0.0625, -0.0625],
            [-0.1250,  0.0000, -0.2500, -0.0625, -0.1875],
        ],
    ])[None, :, :, :]
    output = embedding(query_values, key_values, None)
    assert torch.allclose(output, expected_output)


def test__ALiBiCustomEmbedding__batchSize2__numHeads2():
    embedding = ALiBiCustomEmbedding(number_of_heads=2)
    query_values = torch.tensor([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 3.0, 4.0, 2.0, 0.0],
    ])
    key_values = torch.tensor([
        [4.0, 3.0, 2.0, 1.0, 0.0],
        [2.0, 0.0, 4.0, 1.0, 3.0],
    ])
    expected_output = torch.tensor([
        [
            [
                [-2.0, -1.5, -1.0, -0.5,  0.0],
                [-1.5, -1.0, -0.5,  0.0, -0.5],
                [-1.0, -0.5,  0.0, -0.5, -1.0],
                [-0.5,  0.0, -0.5, -1.0, -1.5],
                [ 0.0, -0.5, -1.0, -1.5, -2.0],
            ],
            [
                [-1.00, -0.75, -0.50, -0.25,  0.00],
                [-0.75, -0.50, -0.25, -0.00, -0.25],
                [-0.50, -0.25,  0.00, -0.25, -0.50],
                [-0.25,  0.00, -0.25, -0.50, -0.75],
                [ 0.00, -0.25, -0.50, -0.75, -1.00],
            ],
        ],
        [
            [
                [-0.5, -0.5, -1.5,  0.0, -1.0],
                [-0.5, -1.5, -0.5, -1.0,  0.0],
                [-1.0, -2.0,  0.0, -1.5, -0.5],
                [ 0.0, -1.0, -1.0, -0.5, -0.5],
                [-1.0,  0.0, -2.0, -0.5, -1.5],
            ],
            [
                [-0.25, -0.25, -0.75,  0.00, -0.50],
                [-0.25, -0.75, -0.25, -0.50,  0.00],
                [-0.50, -1.00,  0.00, -0.75, -0.25],
                [ 0.00, -0.50, -0.50, -0.25, -0.25],
                [-0.50,  0.00, -1.00, -0.25, -0.75],
            ],
        ],
    ])
    output = embedding(query_values, key_values, None)
    assert torch.allclose(output, expected_output)

def test__ALiBiCustomEmbedding__ignore_value_linear_bias():
    embedding = ALiBiCustomEmbedding(number_of_heads=1)
    query_values = torch.tensor([
        [1.0, 3.0, 4.0, 2.0, 0.0]
    ])
    key_values = torch.tensor([
        [2.0, 0.0, 4.0, 1.0, 3.0]
    ])
    value_to_ignore = 0.0

    expected_output = torch.tensor([
        [-0.5, 0.0, -1.5,  0.0, -1.0],
        [-0.5, 0.0, -0.5, -1.0,  0.0],
        [-1.0, 0.0,  0.0, -1.5, -0.5],
        [ 0.0, 0.0, -1.0, -0.5, -0.5],
        [ 0.0, 0.0,  0.0,  0.0,  0.0],
    ])[None, None, :, :]

    output = embedding(query_values, key_values, value_to_ignore)

    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
