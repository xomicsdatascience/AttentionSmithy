import pytest
import torch
from attention_smithy.components import DecoderLayer, MultiheadAttention, FeedForwardNetwork
from attention_smithy.attention import StandardAttentionMethod, BigBirdAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingManager
import copy
import re
import warnings

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
def numeric_embedding_manager():
    return NumericEmbeddingManager([])

@pytest.fixture
def feed_forward_network(embedding_dimension, feed_forward_dimension):
    activation_function_string = 'gelu'
    return FeedForwardNetwork(embedding_dimension, feed_forward_dimension, activation_function_string)

@pytest.fixture
def number_of_heads():
    return 10

@pytest.fixture
def is_causal_masking_warning_error():
    return "CAUTION: your decoder layer self attention method has `is_causal_masking` is set to False. This would render most decoder strategies ineffective."

def test__DecoderLayer__works_with_standard_self_attention(query_tensor, kv_tensor, numeric_embedding_manager, embedding_dimension, number_of_heads, feed_forward_network, dropout, is_causal_masking_warning_error):
    standard_attention_method__with_causal_masking = StandardAttentionMethod(is_causal_masking=True)
    self_attention = MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method__with_causal_masking)
    standard_attention_method__without_causal_masking = StandardAttentionMethod()
    cross_attention = MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method__without_causal_masking)
    standard_decoder_layer = DecoderLayer(embedding_dimension, self_attention, cross_attention, feed_forward_network, dropout)
    output = standard_decoder_layer(tgt=query_tensor, src=kv_tensor, tgt_padding_mask=None, src_padding_mask=None, numeric_embedding_manager=numeric_embedding_manager)
    assert output.shape == query_tensor.shape

def test__DecoderLayer__throws_warning_error_when_standard_self_attention_method_has_no_causal_masking(embedding_dimension, number_of_heads, feed_forward_network, dropout, is_causal_masking_warning_error):
    standard_attention_method__with_causal_masking = StandardAttentionMethod()
    self_attention = MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method__with_causal_masking)
    standard_attention_method__without_causal_masking = StandardAttentionMethod()
    cross_attention = MultiheadAttention(embedding_dimension, number_of_heads, standard_attention_method__without_causal_masking)
    with pytest.raises(RuntimeWarning, match=re.escape(is_causal_masking_warning_error)):
        standard_decoder_layer = DecoderLayer(embedding_dimension, self_attention, cross_attention, feed_forward_network, dropout)

@pytest.fixture
def global_tokens_query(batch_size, num_blocks_query, block_size):
    return torch.zeros((batch_size, num_blocks_query*block_size))

@pytest.fixture
def global_tokens_kv(batch_size, num_blocks_kv, block_size):
    return torch.zeros((batch_size, num_blocks_kv*block_size))

@pytest.mark.skip(reason="Big Bird not implemented")
def test__DecoderLayer__works_with_big_bird_self_attention(query_tensor, kv_tensor, numeric_embedding_manager, global_tokens_query, global_tokens_kv, block_size, embedding_dimension, number_of_heads, feed_forward_network, dropout):
    big_bird_attention_method__with_causal_masking = BigBirdAttentionMethod(block_size, block_size, local_window_extension_length=0, is_causal_masking=True)
    self_attention = MultiheadAttention(embedding_dimension, number_of_heads, big_bird_attention_method__with_causal_masking)
    big_bird_attention_method__without_causal_masking = BigBirdAttentionMethod(block_size, block_size, local_window_extension_length=0)
    cross_attention = MultiheadAttention(embedding_dimension, number_of_heads, big_bird_attention_method__without_causal_masking)
    big_bird_decoder_layer = DecoderLayer(embedding_dimension, self_attention, cross_attention, feed_forward_network, dropout)

    output = big_bird_decoder_layer(
        tgt=query_tensor,
        src=kv_tensor,
        tgt_padding_mask=None,
        src_padding_mask=None,
        numeric_embedding_manager=numeric_embedding_manager,
        global_tokens_query=global_tokens_query,
        global_tokens_kv=global_tokens_kv,
    )
    assert output.shape == query_tensor.shape

@pytest.mark.skip(reason="Big Bird not implemented")
def test__DecoderLayer__throws_warning_error_when_big_bird_self_attention_method_has_no_causal_masking(block_size, embedding_dimension, number_of_heads, feed_forward_network, dropout, is_causal_masking_warning_error):
    big_bird_attention_method__with_causal_masking = BigBirdAttentionMethod(block_size, block_size, local_window_extension_length=0)
    self_attention = MultiheadAttention(embedding_dimension, number_of_heads, big_bird_attention_method__with_causal_masking)
    big_bird_attention_method__without_causal_masking = BigBirdAttentionMethod(block_size, block_size, local_window_extension_length=0)
    cross_attention = MultiheadAttention(embedding_dimension, number_of_heads, big_bird_attention_method__without_causal_masking)
    with pytest.raises(RuntimeWarning, match=re.escape(is_causal_masking_warning_error)):
        big_bird_decoder_layer = DecoderLayer(embedding_dimension, self_attention, cross_attention, feed_forward_network, dropout)
