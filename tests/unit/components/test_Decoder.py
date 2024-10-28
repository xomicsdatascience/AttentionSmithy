import torch
from attention_smithy.components import (
    MultiheadAttention,
    FeedForwardNetwork,
    SublayerUnit,
    DecoderLayer,
    Decoder
)
from attention_smithy.attention import StandardAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingFacade

def test__Decoder():
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
    decoder_layer = DecoderLayer(embedding_dimension, standard_multihead_attention, standard_multihead_attention, feed_forward_network, dropout)
    decoder = Decoder(decoder_layer, number_of_layers)
    assert not torch.equal(decoder.layers[0].self_attention_sublayer.sublayer_module.query_weights.weight, decoder.layers[1].self_attention_sublayer.sublayer_module.query_weights.weight)
    assert not torch.equal(decoder.layers[0].self_attention_sublayer.sublayer_module.key_weights.weight, decoder.layers[1].self_attention_sublayer.sublayer_module.key_weights.weight)
    assert not torch.equal(decoder.layers[0].self_attention_sublayer.sublayer_module.value_weights.weight, decoder.layers[1].self_attention_sublayer.sublayer_module.value_weights.weight)
    assert not torch.equal(decoder.layers[0].self_attention_sublayer.sublayer_module.out_weights.weight, decoder.layers[1].self_attention_sublayer.sublayer_module.out_weights.weight)



    input_tensor = torch.rand((batch_size, sequence_length, embedding_dimension))
    numeric_embedding_facade = NumericEmbeddingFacade()
    output = decoder(input_tensor, input_tensor, tgt_padding_mask=None, src_padding_mask=None, numeric_embedding_facade=numeric_embedding_facade)

    assert input_tensor.shape == output.shape
