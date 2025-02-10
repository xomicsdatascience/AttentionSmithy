import pytest
import torch
import re
from attention_smithy.attention import StandardAttentionMethod, BigBirdAttentionMethod
from attention_smithy.components import MultiheadAttention
from attention_smithy.numeric_embeddings import NumericEmbeddingManager, RotaryPositionEmbedding

@pytest.fixture
def attention_method():
    return StandardAttentionMethod()

@pytest.fixture
def numeric_embedding_manager():
    return NumericEmbeddingManager()

@pytest.fixture
def embedding_dimension():
    return 1200

@pytest.fixture
def number_of_heads():
    return 10

@pytest.fixture
def batch_size():
    return 32

@pytest.fixture
def multihead_attention(embedding_dimension, number_of_heads, attention_method):
    return MultiheadAttention(embedding_dimension, number_of_heads, attention_method)

def test__MultiheadAttention__head_dimension_calculated_correctly(multihead_attention):
    head_dimension = 120
    assert multihead_attention.head_dimension == head_dimension

def test__MultiheadAttention__throws_error_when_embedding_dimension_not_divisible_by_number_of_heads(embedding_dimension, number_of_heads, attention_method):
    incorrect_embedding_dimension = embedding_dimension - 1
    errorOutput = f"embedding_dimension must be divisible by number_of_heads. embedding_dimension: {incorrect_embedding_dimension}, number_of_heads: {number_of_heads}"
    with pytest.raises(ValueError, match=re.escape(errorOutput)):
        multihead_attention = MultiheadAttention(incorrect_embedding_dimension, number_of_heads, attention_method)

def test__MultiheadAttention__forward_pass_output_has_correct_shape_and_back_propogates(batch_size, number_of_heads, embedding_dimension, multihead_attention, numeric_embedding_manager):
    query_and_kv_sequence_length = 15
    input_query = torch.rand((batch_size, query_and_kv_sequence_length, embedding_dimension))
    input_key = input_query.clone()
    input_value = input_query.clone()

    expected_output_shape = input_query.shape
    expected_attention_probabilities_shape = torch.Size([batch_size, number_of_heads, query_and_kv_sequence_length, query_and_kv_sequence_length])
    output, attention_probabilities = multihead_attention(input_query, input_key, input_value, padding_and_loss_attention_mask=None, numeric_embedding_manager=numeric_embedding_manager)
    assert output.shape == expected_output_shape
    assert attention_probabilities.shape == expected_attention_probabilities_shape

def test__MultiheadAttention__throw_error_if_embedding_dimension_does_not_match(batch_size, number_of_heads, embedding_dimension, multihead_attention, numeric_embedding_manager):
    query_and_kv_sequence_length = 15
    incorrect_embedding_dimension = embedding_dimension - 1
    input_query = torch.rand((batch_size, query_and_kv_sequence_length, incorrect_embedding_dimension))
    input_key = input_query.clone()
    input_value = input_query.clone()
    errorOutput = f"Embedding dimension of input tensors does not match the embedding dimension established in the MultiheadAttention initialization. Tensor: {incorrect_embedding_dimension}, attention: {embedding_dimension}"
    with pytest.raises(ValueError, match=re.escape(errorOutput)):
        output, attention_probabilities = multihead_attention(input_query, input_key, input_value, numeric_embedding_manager)

def test__MultiheadAttention__multihead_attention_still_works_when_query_and_key_value_sequence_lengths_are_different(
        batch_size, number_of_heads, embedding_dimension, multihead_attention, numeric_embedding_manager):
    query_sequence_length = 15
    kv_sequence_length = 20
    input_query = torch.rand((batch_size, query_sequence_length, embedding_dimension))
    input_key = torch.rand((batch_size, kv_sequence_length, embedding_dimension))
    input_value = input_key.clone()

    expected_output_shape = input_query.shape
    expected_attention_probabilities_shape = torch.Size([batch_size, number_of_heads, query_sequence_length, kv_sequence_length])
    output, attention_probabilities = multihead_attention(input_query, input_key, input_value, padding_and_loss_attention_mask=None, numeric_embedding_manager=numeric_embedding_manager)
    assert output.shape == expected_output_shape
    assert attention_probabilities.shape == expected_attention_probabilities_shape

def test__MultiheadAttention__forward_pass_output_has_correct_shape__with_rotary_embedding(batch_size, number_of_heads, embedding_dimension, multihead_attention):
    rotary_embedding = RotaryPositionEmbedding(number_of_heads)
    numeric_embedding_manager = NumericEmbeddingManager(rotary_position=rotary_embedding)
    query_and_kv_sequence_length = 15
    input_query = torch.rand((batch_size, query_and_kv_sequence_length, embedding_dimension))
    input_key = input_query.clone()
    input_value = input_query.clone()

    expected_output_shape = input_query.shape
    expected_attention_probabilities_shape = torch.Size([batch_size, number_of_heads, query_and_kv_sequence_length, query_and_kv_sequence_length])
    output, attention_probabilities = multihead_attention(input_query, input_key, input_value, padding_and_loss_attention_mask=None, numeric_embedding_manager=numeric_embedding_manager)
    assert output.shape == expected_output_shape
    assert attention_probabilities.shape == expected_attention_probabilities_shape
