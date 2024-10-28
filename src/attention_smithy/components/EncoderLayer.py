from torch import nn
from attention_smithy.components import SublayerUnit

class EncoderLayer(nn.Module):
    """
    A single encoder layer as described in the Attention is All You Need paper.
        Most encoder models generally have multiple layers. The composition of
        an encoder layer is a single self attention sublayer followed by a feed
        forward network sublayer. Inputs are fed through both and the encoded
        outputs are returned.
    """
    def __init__(self, embedding_dimension, self_attention, feed_forward, dropout):
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

    def forward(
        self, query, src, numeric_embedding_facade, **kwargs
    ):
        """
        Args:
            query (torch.Tensor): The tokenized input meant to be analyzed as the
                "query" matrix described in the original paper, of shape
                (batch_size, query_sequence_length, embedding_dimension). It
                is also called the "target" or "tgt" matrix.
            src (torch.Tensor): The tokenized input meant to be analyzed as the
                "key" and "value" matrix described in the original paper, of shape
                (batch_size, kv_sequence_length, embedding_dimension). The "src"
                name is used in encoder/decoder contexts to represent "source" data
                as opposed to "target" data, as in translating English (source) to
                French (target).
            numeric_embedding_facade (NumericEmbeddingFacade): Facade class that contains
                all numeric embedding methods (including position).
            kwargs: Customized components downstream may require additional inputs.
                This argument effectively packages them together for use downstream.
                For example, the BigBirdAttentionMethod requires additional masking
                variables that are not explicitly required by MultiheadAttention or
                StandardAttentionMethod, so they are not included as broadly required
                parameters.
        Returns:
            torch.Tensor: An output tensor of shape
                (batch_size, query_sequence_length, embedding_dimension).
        """
        query = self.self_attention_sublayer(
            query,
            input_key=src,
            input_value=src,
            numeric_embedding_facade=numeric_embedding_facade,
            **kwargs,
        )
        return self.feed_forward_sublayer(query)
