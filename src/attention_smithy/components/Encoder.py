from torch import nn
from attention_smithy.utils import clone_module_consecutively

class Encoder(nn.Module):
    """
    A full encoder as described in the Attention is All You Need paper.
        An encoder consists of several encoder layers. This class facilitates
        running data through each encoder layer.

    """
    def __init__(self, layer, number_of_layers):
        """
        Args:
            layer (EncoderLayer): An instance of the EncoderLayer class.
                This layer will be duplicated across the encoder.
            number_of_layers (int): The number of times the given layer
                should be duplicated throughout the encoder.
        """
        super().__init__()
        self.layers = clone_module_consecutively(layer, number_of_layers)
        self.norm = nn.LayerNorm(layer.embedding_dimension)

    def forward(self, src, src_padding_mask, **kwargs):
        """
        See args and return value of EncoderLayer forward function
        """
        for layer in self.layers:
            src = layer(src, src_padding_mask, **kwargs)
        return self.norm(src)
