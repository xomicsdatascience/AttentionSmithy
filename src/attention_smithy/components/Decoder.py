from torch import nn
from attention_smithy.utils import clone_module_consecutively

class Decoder(nn.Module):
    """
    A full decoder as described in the Attention is All You Need paper.
        A decoder consists of several decoder layers. This class facilitates
        running data through each decoder layer.

    """
    def __init__(self, layer, number_of_layers):
        """
        Args:
            layer (DecoderLayer): An instance of the DecoderLayer class.
                This layer will be duplicated across the decoder.
            number_of_layers (int): The number of times the given layer
                should be duplicated throughout the decoder.
        """
        super().__init__()
        self.layers = clone_module_consecutively(layer, number_of_layers)
        self.norm = nn.LayerNorm(layer.embedding_dimension)

    def forward(self, tgt, src, tgt_padding_mask, src_padding_mask, **kwargs):
        """
        See args and return value of DecoderLayer forward function
        """
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_padding_mask, src_padding_mask, **kwargs)
        return self.norm(tgt)
