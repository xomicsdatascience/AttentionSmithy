import torch
from torch import nn

class LearnedPositionEmbedding(nn.Module):
    """
    A pytorch module for learned positional embeddings.
    """
    def __init__(self,
                 max_sequence_length: int,
                 embedding_dimension: int,
                 ) -> None:
        """
        Args:
            max_sequence_length (int): The maximum sequence length for any input.
            embedding_dimension (int): The token embedding dimension size.
        Attributes:
            embedding (nn.Embedding): a module to pass position or other sequences
                to for embedding.
        """

        super(LearnedPositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_sequence_length, embedding_dimension)


    def forward(self,
                sequence_length: int,
                ) -> torch.Tensor:
        """
        Args:
            sequence_length (torch.Tensor): The sequence length of a given batch.
        Returns:
            torch.Tensor: An output tensor of shape
                (batch_size, sequence_length, embedding_dimension)
        """
        position_ids = torch.arange(sequence_length, device=self.embedding.weight.device)
        return self.embedding(position_ids)
