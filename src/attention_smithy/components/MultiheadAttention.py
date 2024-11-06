import torch
import torch.nn as nn
from typing import Union
from attention_smithy.attention import BigBirdAttentionMethod, StandardAttentionMethod
from attention_smithy.numeric_embeddings import NumericEmbeddingFacade

class MultiheadAttention(nn.Module):
    """
    A pytorch module that performs multihead attention as described in the "Attention
        Is All You Need" paper. Specifically, it multiplies the necessary tensors
        by their respective weights and reshapes them for the attention method. It
        also takes the outputs of the attention mechanism, reshapes them appropriately,
        and multiplies them by the output weight matrix before returning them.

    NOTE: The specific attention calculations of the original paper are found in the
        "StandardAttentionMethod" class. It is separated to allow for alternate attention
        mechanisms, such as Big Bird Attention, while keeping some of the foundational
        preprocessing steps (weight matrix multiplication, reshaping into heads etc.) to
        remain constant.
    """

    def __init__(self,
                 embedding_dimension: int,
                 number_of_heads: int,
                 attention_method: Union[BigBirdAttentionMethod, StandardAttentionMethod],
                 ) -> None:
        """
        Args:
            embedding_dimension (int): The token embedding dimension size.
                NOTE: embedding_dimension size MUST be divisible by number_of_heads.
            number_of_heads (int): The number of heads to split the embedding dimension
                into.
            attention_method: A class that performs attention scoring. Examples include
                the standard method as described in the original paper as well as big
                bird attention, which is a
        """

        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.attention_method = attention_method
        self.head_dimension = embedding_dimension // number_of_heads
        if self.head_dimension * number_of_heads != embedding_dimension:
            raise ValueError(f"embedding_dimension must be divisible by number_of_heads. embedding_dimension: {embedding_dimension}, number_of_heads: {number_of_heads}")

        self.query_weights = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_weights = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_weights = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_weights = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self,
                input_query: torch.Tensor,
                input_key: torch.Tensor,
                input_value: torch.Tensor,
                numeric_embedding_facade: NumericEmbeddingFacade,
                **kwargs
                ) -> torch.Tensor:
        """
        Args:
            input_query (torch.Tensor): The tokenized input meant to be analyzed as the
                "query" matrix described in the original paper, of shape
                (batch_size, query_sequence_length, embedding_dimension)
            input_key (torch.Tensor): The tokenized input meant to be analyzed as the
                "key" matrix described in the original paper, of shape
                (batch_size, kv_sequence_length, embedding_dimension)
            input_value (torch.Tensor): The tokenized input meant to be analyzed as the
                "value" matrix described in the original paper, of shape
                (batch_size, kv_sequence_length, embedding_dimension)
            numeric_embedding_facade (NumericEmbeddingFacade): Facade class that contains
                all numeric embedding methods (including position). Required to enable
                rotary embedding.
        Returns:
            attention_outputs (torch.Tensor): The output tensor, of shape
                (batch_size, query_length, embedding_dimension).
            attention_probablities (torch.Tensor): The attention probablity matrix calculated during the
                attention method. Returned with the output for external analysis reasons. Of shape
                (batch_size, number_of_heads, query_length, kv_length).
        """

        key, query, value = self._prepare_matrices_for_attention(input_key, input_query, input_value)
        query, key = numeric_embedding_facade.apply_rotation_to_query_and_key_matrices(query, key)
        attention_output, attention_probabilities = self.attention_method(
            query, key, value, numeric_embedding_facade, **kwargs
        )
        attention_output = self._reshape_attention_output_to_match_query_shape(attention_output, input_query)
        output = self.out_weights(attention_output)
        return output, attention_probabilities

    def _prepare_matrices_for_attention(self, input_key, input_query, input_value):
        batch_size, query_sequence_length, embedding_dimension = input_query.shape
        if embedding_dimension != self.embedding_dimension:
            raise ValueError(f"Embedding dimension of input tensors does not match the embedding dimension established in the MultiheadAttention initialization. Tensor: {embedding_dimension}, attention: {self.embedding_dimension}")
        _, kv_sequence_length, _ = input_key.shape
        query = self.query_weights(input_query)
        key = self.key_weights(input_key)
        value = self.value_weights(input_value)
        query = query.view(batch_size, query_sequence_length, self.number_of_heads, self.head_dimension).transpose(
            1, 2
        )
        key = key.view(batch_size, kv_sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, kv_sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
        return key, query, value

    def _reshape_attention_output_to_match_query_shape(self, attention_output, input_query):
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(input_query.shape)
        )
        return attention_output



