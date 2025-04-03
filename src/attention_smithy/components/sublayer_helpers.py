from typing import Union
from attention_smithy.components import SublayerUnit, MultiheadAttention, FeedForwardNetwork

def wrap_sublayer(
    sublayer: Union[SublayerUnit, MultiheadAttention, FeedForwardNetwork],
    embedding_dimension: int,
    dropout: float,
) -> SublayerUnit:
    """
    If `sublayer` is not already a fully constructed SublayerUnit,
    wrap it into one. Otherwise, return it as is.
    """
    if isinstance(sublayer, SublayerUnit):
        return sublayer
    return SublayerUnit(sublayer, embedding_dimension, dropout)
