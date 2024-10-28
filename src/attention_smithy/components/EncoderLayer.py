import torch
from torch import nn
from attention_smithy.components import SublayerUnit, MultiheadAttention, FeedForwardNetwork

class EncoderLayer(nn.Module):
    """
    A single encoder layer as described in the Attention is All You Need paper.
        Most encoder models generally have multiple layers. The composition of
        an encoder layer is a single self attention sublayer followed by a feed
        forward network sublayer.
    """
    def __init__(self,
                 embedding_dimension: int,
                 self_attention: MultiheadAttention,
                 feed_forward: FeedForwardNetwork,
                 dropout: float,
                 ):
        """
        Args:
            embedding_dimension (int): The token embedding dimension size.
            self_attention (MultiheadAttention): An instance of the
                MultiheadAttention class, a component class of AttentionSmithy.
            feed_forward (FeedForwardNetwork): An instance of the
                FeedForwardNetwork class, a component class of AttentionSmithy.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
        """
        super().__init__()
        self.self_attention_sublayer = SublayerUnit(self_attention, embedding_dimension, dropout)
        self.feed_forward_sublayer = SublayerUnit(feed_forward, embedding_dimension, dropout)
        self.embedding_dimension = embedding_dimension

    def forward(self,
                src: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs
                ):
        """
        Args:
            src (torch.Tensor): The tokenized input, of shape
                (batch_size, sequence_length, embedding_dimension). The "src"
                name is used in encoder/decoder contexts to represent "source" data
                as opposed to "target" data, as in translating English (source) to
                French (target). Because this is an encoder, there is only "src" data.
            src_padding_mask (torch.tensor): The padding attention mask, of shape
                (batch_size, sequence_length).
            kwargs: Customized components downstream may require additional inputs.
                This argument effectively packages them together for use downstream.
                For example, the BigBirdAttentionMethod requires additional masking
                variables that are not explicitly required by MultiheadAttention or
                StandardAttentionMethod, so they are not included as broadly required
                parameters.
        Returns:
            torch.Tensor: An output tensor of shape
                (batch_size, sequence_length, embedding_dimension).
        """
        src = self.self_attention_sublayer(
            src,
            input_key=src,
            input_value=src,
            padding_and_loss_attention_mask=src_padding_mask,
            **kwargs,
        )
        return self.feed_forward_sublayer(src)
