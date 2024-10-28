import torch
from torch import nn
from typing import Union
from attention_smithy.components import MultiheadAttention, FeedForwardNetwork

class SublayerUnit(nn.Module):
    """
    A subset of functionality in encoders and decoders as described in the paper
        "Attention Is All You Need." Referencing Figure 1 of the paper, you see
        there are 5 "Add & Norm" boxes in the figure. In each of these cases,
        the tokenized input is passed through a sublayer (multihead attention,
        feedforward network), then added back to the tokenized input and normalized
        via Layer Normalization. The sublayer changes, but the adding and normalizing
        process remains identical in all 5 instances. This class captures that
        repeated process.
    """
    def __init__(self,
                 sublayer_module: Union[MultiheadAttention, FeedForwardNetwork],
                 embedding_dimension: int,
                 dropout: float
                 ) -> None:
        """
        Args:
            sublayer_module (nn.Module child): A sublayer process in a transformer-based
                encoder or decoder. This could be self-attention, cross-attention
                (both processes completed through the MultiheadAttenion class) or feed-
                forward network (the FeedForwardNetwork class). Input and output shape
                should be identical.
            embedding_dimension (int): The token embedding dimension size.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.sublayer_module = sublayer_module
        self.norm = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                **kwargs
                ):
        """
        Args:
            x (torch.Tensor): The tokenized input data of shape
                (batch_size, sequence_length, embedding_dimension).
            NOTE: Some sublayers require additional inputs, so **kwargs is included. They will
                need to be specified in the encoder process.

        Returns:
            out (torch.Tensor): The output tensor, of shape
                (batch_size, sequence_length, embedding_dimension).
        """
        out = self.sublayer_module(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out = self.dropout(out)
        out = x + out
        return self.norm(out)
