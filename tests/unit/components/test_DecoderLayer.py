import pytest
import torch
from attention_smithy.components import DecoderLayer, MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod, BigBirdAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingFacade
import copy

@pytest.fixture
def embedding_dimension():
    return 120

@pytest.fixture
def feed_forward_dimension():
    return 200

@pytest.fixture
def batch_size():
    return 32

@pytest.fixture
def num_blocks_query():
    return 5

@pytest.fixture
def num_blocks_kv():
    return 5

@pytest.fixture
def block_size():
    return 4

@pytest.fixture
def query_tensor(batch_size, embedding_dimension, num_blocks_query, block_size):
    sequence_length = num_blocks_query * block_size
    return torch.rand((batch_size, sequence_length, embedding_dimension))

@pytest.fixture
def kv_tensor(batch_size, embedding_dimension, num_blocks_kv, block_size):
    sequence_length = num_blocks_kv * block_size
    return torch.rand((batch_size, sequence_length, embedding_dimension))

@pytest.fixture
def dropout():
    return 0.0

@pytest.fixture
def numeric_embedding_facade():
    return NumericEmbeddingFacade()

@pytest.fixture
def feed_forward_network(embedding_dimension, feed_forward_dimension):
    activation_function_string = 'gelu'
    return FeedForwardNetwork(embedding_dimension, feed_forward_dimension, activation_function_string)

@pytest.fixture
def number_of_heads():
    return 10

@pytest.fixture
def standard_multihead_attention(embedding_dimension, number_of_heads):
    standard_attention_method = StandardAttentionMethod()
    return MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method)

@pytest.fixture
def standard_decoder_layer(embedding_dimension, standard_multihead_attention, feed_forward_network, dropout):
    return DecoderLayer(embedding_dimension, standard_multihead_attention, copy.deepcopy(standard_multihead_attention), feed_forward_network, dropout)

@pytest.fixture
def big_bird_multihead_attention(block_size, embedding_dimension, number_of_heads):
    big_bird_attention = BigBirdAttentionMethod(block_size, block_size, local_window_extension_length=0)
    return MultiheadAttention(embedding_dimension, number_of_heads, big_bird_attention)

@pytest.fixture
def big_bird_decoder_layer(embedding_dimension, big_bird_multihead_attention, feed_forward_network, dropout):
    return DecoderLayer(embedding_dimension, big_bird_multihead_attention, copy.deepcopy(big_bird_multihead_attention), feed_forward_network, dropout)

@pytest.fixture
def global_tokens_query(batch_size, num_blocks_query, block_size):
    return torch.zeros((batch_size, num_blocks_query*block_size))

@pytest.fixture
def global_tokens_kv(batch_size, num_blocks_kv, block_size):
    return torch.zeros((batch_size, num_blocks_kv*block_size))

def test__EncoderLayer__works_with_standard_self_attention(query_tensor, kv_tensor, numeric_embedding_facade, standard_decoder_layer):
    output = standard_decoder_layer(tgt=query_tensor, src=kv_tensor, tgt_padding_mask=None, src_padding_mask=None, numeric_embedding_facade=numeric_embedding_facade)
    assert output.shape == query_tensor.shape

def test__DecoderLayer__works_with_big_bird_self_attention(query_tensor, kv_tensor, numeric_embedding_facade, big_bird_decoder_layer, global_tokens_query, global_tokens_kv):
    output = big_bird_decoder_layer(
        tgt=query_tensor,
        src=kv_tensor,
        tgt_padding_mask=None,
        src_padding_mask=None,
        numeric_embedding_facade=numeric_embedding_facade,
        global_tokens_query=global_tokens_query,
        global_tokens_kv=global_tokens_kv,
    )
    assert output.shape == query_tensor.shape