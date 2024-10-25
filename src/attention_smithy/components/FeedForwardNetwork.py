from torch import nn
from attention_smithy.utils import select_activation_function_module

class FeedForwardNetwork(nn.Module):
    """
    The feed forward component of encoder and decoder layers of a transformer, as described in
        the original paper "Attention Is All You Need."
    """
    def __init__(self, embedding_dimension, feed_forward_dimension, activation_function_string, dropout=0.1):
        """
        Args:
            embedding_dimension (int): The token embedding dimension size.
            feed_forward_dimension (int): The dimension size expanded or retracted to during
                this component.
            activation_function_string (str): A string representation of the desired activation
                function. The activation function module representations can be identified in
                the utils.py file.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.weights_to_feed_forward_dimension = nn.Linear(embedding_dimension, feed_forward_dimension)
        self.weights_back_to_embedding_dimension = nn.Linear(feed_forward_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
        self.activation = select_activation_function_module(activation_function_string)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The tokenized data input, of shape
                (batch_size, sequence_length, embedding_dimension)

        Returns:
            out (torch.Tensor): The output of the feed forward network, also of shape
                (batch_size, sequence_length, embedding_dimension)
        """
        out = self.weights_to_feed_forward_dimension(x)
        out = self.activation(out)
        out = self.dropout(out)
        return self.weights_back_to_embedding_dimension(out)