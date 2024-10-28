import torch
import torch.nn as nn
from torch.nn import functional as F
from attention_smithy.utils import create_causal_mask


class StandardAttentionMethod(nn.Module):
    """
    A PyTorch module implementing the attention mechanism described in the "Attention Is All You Need" paper.
        The query, key and value matrices have already been multiplied by their respective weights before passing
        through this module.
    """

    def __init__(self, dropout: float = 0.0, is_causal_masking: bool = False):
        """
        Initializes the standard attention module.

        Args:
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_causal_masking (bool, optional): When set to true, tokens can only be attended to by past tokens.
                This is most often seen decoder models like GPT. Defaults to false.
        """

        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.is_causal_masking = is_causal_masking

    def forward(
        self,
        q,
        k,
        v,
        numeric_embedding_facade,
        padding_and_loss_attention_mask=None,
        **kwargs,
    ):
        """
        Forward pass of the standard attention module.

        Args:
            q (torch.Tensor): The query tensor embedding, of shape (batch_size, num_heads, query_length, head_dim)
                Note that the embedding dimension can be calculated by multiplying (num_heads * head_dim)
            k (torch.Tensor): The key tensor embedding, of shape (batch_size, num_heads, kv_length, head_dim)
            v (torch.Tensor): The value tensor embedding, of shape (batch_size, num_heads, kv_length, head_dim)
            numeric_embedding_facade (NumericEmbeddingFacade): Facade class that contains
                all numeric embedding methods (including position). Required to enable
                alibi embedding.
            padding_and_loss_attention_mask (torch.Tensor): A boolean mask corresponding to padding tokens in the query
                and key inputs, of shape (batch_size, 1, query_length, kv_length).
                NOTE: tokens are often masked to pad samples to match the largest sample in a batch to keep
                them the same size. However, the loss function may also require masking tokens, as in BERT.
                That masking is included in this mask tensor.
        Returns:
            attention_outputs (torch.Tensor): The output tensor, of shape
                (batch_size, query_length, embedding_dimension).
            attention_probablities (torch.Tensor): The attention probablity matrix calculated during the
                attention method. Returned with the output for external analysis reasons. Of shape
                (batch_size, number_of_heads, query_length, kv_length).
        """
        attention_scores = self._calculate_query_by_key_attention_scores(q, k)
        attention_scores += numeric_embedding_facade.calculate_alibi_attention_score_distances(q, k, **kwargs)
        attention_scores = self._apply_masking_to_attention_scores(
            attention_scores, padding_and_loss_attention_mask
        )
        attention_probabilities = self._reduce_attention_scores_to_probabilities(attention_scores)
        attention_outputs = torch.matmul(attention_probabilities, v)
        return attention_outputs, attention_probabilities

    def _calculate_query_by_key_attention_scores(self, q, k):
        head_dimension = q.shape[-1]
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores /= torch.sqrt(torch.tensor(head_dimension, dtype=torch.float32))
        return attention_scores

    def _apply_masking_to_attention_scores(
        self, attention_scores, padding_and_loss_attention_mask
    ):
        attention_scores = self._apply_diagonal_causal_masking_if_relevant(attention_scores)
        attention_scores = self._apply_padding_and_loss_attention_masking_if_relevant(
            attention_scores, padding_and_loss_attention_mask
        )
        return attention_scores

    def _apply_diagonal_causal_masking_if_relevant(self, attention_scores):
        if self.is_causal_masking:
            causal_mask = create_causal_mask(attention_scores.size(-1))
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))
        return attention_scores

    def _apply_padding_and_loss_attention_masking_if_relevant(
        self, attention_scores, padding_and_loss_attention_mask
    ):
        if padding_and_loss_attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                padding_and_loss_attention_mask == 0, float("-inf")
            )
        return attention_scores

    def _reduce_attention_scores_to_probabilities(self, attention_scores):
        attention_probabilities = F.softmax(attention_scores, dim=-1)
        attention_probabilities = self.dropout_layer(attention_probabilities)
        return attention_probabilities
