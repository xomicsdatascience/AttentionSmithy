import torch
from attention_smithy.numeric_embeddings import SinusoidalEmbedding


def test__SinusoidalEmbedding__positional_encoding_has_expected_shape():
    embedding_dimension_size = 128
    max_len = 100
    embedding = SinusoidalEmbedding(embedding_dimension=embedding_dimension_size, max_len=max_len)
    assert embedding.positional_encoding.shape == (max_len, embedding_dimension_size)


def test__SinusoidalEmbedding__position_values_encode_correctly():
    embedding = SinusoidalEmbedding(embedding_dimension=10, max_len=4)
    x = torch.rand(2, 4, 10)
    encoding = embedding(x)
    expected_encoding = torch.tensor(
        [
            [
                0.0000e00,
                1.0000e00,
                0.0000e00,
                1.0000e00,
                0.0000e00,
                1.0000e00,
                0.0000e00,
                1.0000e00,
                0.0000e00,
                1.0000e00,
            ],
            [
                8.4147e-01,
                5.4030e-01,
                1.5783e-01,
                9.8747e-01,
                2.5116e-02,
                9.9968e-01,
                3.9811e-03,
                9.9999e-01,
                6.3096e-04,
                1.0000e00,
            ],
            [
                9.0930e-01,
                -4.1615e-01,
                3.1170e-01,
                9.5018e-01,
                5.0217e-02,
                9.9874e-01,
                7.9621e-03,
                9.9997e-01,
                1.2619e-03,
                1.0000e00,
            ],
            [
                1.4112e-01,
                -9.8999e-01,
                4.5775e-01,
                8.8908e-01,
                7.5285e-02,
                9.9716e-01,
                1.1943e-02,
                9.9993e-01,
                1.8929e-03,
                1.0000e00,
            ],
        ]
    )
    assert torch.allclose(encoding, expected_encoding, atol=1e-4)


def test__SinusoidalEmbedding__float_values_encode_correctly():
    embedding = SinusoidalEmbedding(embedding_dimension=10, max_len=2)
    x = torch.rand(2, 2, 10)
    custom_values = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )

    encoding = embedding(x=x, custom_values=custom_values)
    expected_encoding = torch.tensor(
        [
            [
                [
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                    0.0000e00,
                    1.0000e00,
                ],
                [
                    8.4147e-01,
                    5.4030e-01,
                    1.5783e-01,
                    9.8747e-01,
                    2.5116e-02,
                    9.9968e-01,
                    3.9811e-03,
                    9.9999e-01,
                    6.3096e-04,
                    1.0000e00,
                ],
            ],
            [
                [
                    9.0930e-01,
                    -4.1615e-01,
                    3.1170e-01,
                    9.5018e-01,
                    5.0217e-02,
                    9.9874e-01,
                    7.9621e-03,
                    9.9997e-01,
                    1.2619e-03,
                    1.0000e00,
                ],
                [
                    1.4112e-01,
                    -9.8999e-01,
                    4.5775e-01,
                    8.8908e-01,
                    7.5285e-02,
                    9.9716e-01,
                    1.1943e-02,
                    9.9993e-01,
                    1.8929e-03,
                    1.0000e00,
                ],
            ],
        ]
    )
    assert torch.allclose(encoding, expected_encoding, atol=1e-4)
