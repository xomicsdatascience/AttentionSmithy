import torch
from attention_smithy.components import (
    MultiheadAttention,
    FeedForwardNetwork,
    SublayerUnit,
    EncoderLayer,
    Encoder
)
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingFacade

def test__Encoder():
    batch_size = 32
    sequence_length = 200
    embedding_dimension = 120
    number_of_heads = 10
    feed_forward_dimension = 240
    activation_function_string = 'gelu'
    standard_attention_method = StandardAttentionMethod()
    standard_multihead_attention = MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method)
    feed_forward_network = FeedForwardNetwork(embedding_dimension, feed_forward_dimension, activation_function_string)
    dropout = 0.0
    number_of_layers = 2
    encoder_layer = EncoderLayer(embedding_dimension, standard_multihead_attention, feed_forward_network, dropout)
    encoder = Encoder(encoder_layer, number_of_layers)

    input_tensor = torch.rand((batch_size, sequence_length, embedding_dimension))
    numeric_embedding_facade = NumericEmbeddingFacade()
    output = encoder(input_tensor, src_padding_mask=None, numeric_embedding_facade=numeric_embedding_facade)
    assert input_tensor.shape == output.shape