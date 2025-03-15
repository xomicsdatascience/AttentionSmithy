import torch
from torch import nn
from attention_smithy.components import (
    MultiheadAttention,
    FeedForwardNetwork,
    SublayerUnit,
    EncoderLayer,
    Encoder
)
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingManager

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
    assert not torch.equal(encoder.layers[0].self_attention_sublayer.sublayer_module.query_weights.weight, encoder.layers[1].self_attention_sublayer.sublayer_module.query_weights.weight)
    assert not torch.equal(encoder.layers[0].self_attention_sublayer.sublayer_module.key_weights.weight, encoder.layers[1].self_attention_sublayer.sublayer_module.key_weights.weight)
    assert not torch.equal(encoder.layers[0].self_attention_sublayer.sublayer_module.value_weights.weight, encoder.layers[1].self_attention_sublayer.sublayer_module.value_weights.weight)
    assert not torch.equal(encoder.layers[0].self_attention_sublayer.sublayer_module.out_weights.weight, encoder.layers[1].self_attention_sublayer.sublayer_module.out_weights.weight)

    input_tensor = torch.rand((batch_size, sequence_length, embedding_dimension))
    numeric_embedding_manager = NumericEmbeddingManager([])
    output = encoder(input_tensor, src_padding_mask=None, numeric_embedding_manager=numeric_embedding_manager)
    assert input_tensor.shape == output.shape

class SimpleEncoderLayer(nn.Module):
    def __init__(self, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.linear = nn.Linear(embedding_dimension, embedding_dimension)


def test_freeze_layers():
    embedding_dimension = 10
    layer = SimpleEncoderLayer(embedding_dimension)
    number_of_layers = 3
    encoder = Encoder(layer, number_of_layers)

    encoder.freeze_layers(2)

    for idx, module in enumerate(encoder.layers):
        for param in module.parameters():
            if idx < 2:
                assert not param.requires_grad, f"Layer {idx} should be frozen"
            else:
                assert param.requires_grad, f"Layer {idx} should not be frozen"

def test_freeze_all_layers():
    embedding_dimension = 10
    layer = SimpleEncoderLayer(embedding_dimension)
    number_of_layers = 3
    encoder = Encoder(layer, number_of_layers)
    encoder.freeze_layers(3)

    for module in encoder.layers:
        for param in module.parameters():
            assert not param.requires_grad, "All layers should be frozen"

def test_freeze_no_layers():
    embedding_dimension = 10
    layer = SimpleEncoderLayer(embedding_dimension)
    number_of_layers = 3
    encoder = Encoder(layer, number_of_layers)
    encoder.freeze_layers(0)

    for module in encoder.layers:
        for param in module.parameters():
            assert param.requires_grad, "No layers should be frozen"

def test_freeze_more_layers_than_exist():
    embedding_dimension = 10
    layer = SimpleEncoderLayer(embedding_dimension)
    number_of_layers = 3
    encoder = Encoder(layer, number_of_layers)
    encoder.freeze_layers(5)
    for module in encoder.layers:
        for param in module.parameters():
            assert not param.requires_grad, "All layers should be frozen"

