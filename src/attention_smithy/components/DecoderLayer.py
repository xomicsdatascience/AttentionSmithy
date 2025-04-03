import torch
from torch import nn
from typing import Union
from attention_smithy.components import SublayerUnit, MultiheadAttention, FeedForwardNetwork
from attention_smithy.components.sublayer_helpers import wrap_sublayer
class DecoderLayer(nn.Module):
    """
    A single decoder layer as described in the Attention is All You Need paper.
        Most decoder models generally have multiple layers. The composition of
        a decoder layer is a single self attention sublayer, followed by a
        single cross attenion sublayer, followed by a feed forward network
        sublayer.
    """
    def __init__(
        self,
        embedding_dimension: int,
        self_attention: Union[MultiheadAttention, SublayerUnit],
        cross_attention: Union[MultiheadAttention, SublayerUnit],
        feed_forward: Union[FeedForwardNetwork, SublayerUnit],
        dropout: float,
    ) -> None:
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
        self.self_attention_sublayer = wrap_sublayer(self_attention, embedding_dimension, dropout)
        self.cross_attention_sublayer = wrap_sublayer(cross_attention, embedding_dimension, dropout)
        self.feed_forward_sublayer = wrap_sublayer(feed_forward, embedding_dimension, dropout)
        self.embedding_dimension = embedding_dimension
        if not self.self_attention_sublayer.sublayer_module.attention_method.is_causal_masking:
            raise RuntimeWarning(
                "CAUTION: your decoder layer self attention method has `is_causal_masking` set to False. "
                "This would render most decoder strategies ineffective."
            )


    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): The tokenized input, of shape
                (batch_size, tgt_sequence_length, embedding_dimension). The "src"
                name is used in encoder/decoder contexts to represent "source" data
                as opposed to "target" data, as in translating English (source) to
                French (target). The decoder focuses on decoding the "target" data.
            src (torch.Tensor): The source data that informs the results. In the
                translation example above, "src" represents an encoded English
                sentence to be translated into the "target" French language. Of shape
                (batch_size, src_sequence_length, embedding_dimension).
            tgt_padding_mask (torch.tensor): The padding attention mask for tgt, of
                shape (batch_size, tgt_sequence_length).
            src_padding_mask (torch.tensor): The padding attention mask for src, of
                shape (batch_size, src_sequence_length).
            kwargs: Customized components downstream may require additional inputs.
                This argument effectively packages them together for use downstream.
                For example, the BigBirdAttentionMethod requires additional masking
                variables that are not explicitly required by MultiheadAttention or
                StandardAttentionMethod, so they are not included as broadly required
                parameters.
        Returns:
            torch.Tensor: An output tensor of shape
                (batch_size, tgt_sequence_length, embedding_dimension).
        """
        tgt = self.self_attention_sublayer(
            tgt,
            input_key=tgt,
            input_value=tgt,
            padding_and_loss_attention_mask=tgt_padding_mask,
            **kwargs,
        )
        tgt = self.cross_attention_sublayer(
            tgt,
            input_key=src,
            input_value=src,
            padding_and_loss_attention_mask=src_padding_mask,
            **kwargs,
        )
        return self.feed_forward_sublayer(tgt)
