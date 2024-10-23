import torch
from attention_smithy.numeric_embeddings import SinusoidalEmbedding

def test__SinusoidalEmbedding__positional_encoding_has_expected_shape():
    embedding_dimension_size = 128
    max_len = 100
    embedding = SinusoidalEmbedding(d_model=embedding_dimension_size, max_len=max_len)
    assert embedding.positional_encoding.shape == (max_len, embedding_dimension_size)

def test__SinusoidalEmbedding__position_values_encode_correctly():
    embedding = SinusoidalEmbedding(d_model=10, max_len=2)
    expected_encoding = torch.tensor([
        [0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00,
         0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00],
        [8.4147e-01, 5.4030e-01, 1.5783e-01, 9.8747e-01, 2.5116e-02, 9.9968e-01,
         3.9811e-03, 9.9999e-01, 6.3096e-04, 1.0000e+00]])
    assert torch.allclose(embedding.positional_encoding, expected_encoding, atol=1e-4)
