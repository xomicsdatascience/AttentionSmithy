import torch
from attention_smithy.components import FeedForwardNetwork

def test__FeedForwardNetwork__forward_pass_outputs_are_of_expected_shape():
    batch_size = 30
    query_length = 20
    embedding_dimension = 128
    feed_forward_dimension = 100
    activation_function_string = 'gelu'
    input_tensor = torch.rand((batch_size, query_length, embedding_dimension))
    feed_forward_network = FeedForwardNetwork(embedding_dimension, feed_forward_dimension, activation_function_string)
    output = feed_forward_network(input_tensor)
    assert output.shape == input_tensor.shape