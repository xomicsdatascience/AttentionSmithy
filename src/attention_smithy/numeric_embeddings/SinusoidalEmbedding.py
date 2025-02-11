import torch

class SinusoidalEmbedding:
    """
    A class that encodes numeric values as tokens. It relies on sinusoidal equations described in the original
        "Attention Is All You Need" paper.
    WHEN APPLIED: Generally added to token embeddings at the beginning of the model.
    NOTE: These numeric encodings typically encode position (0, 1, 2 etc.). However, it could be used to
        encode custom numeric values as well. Thus there are two child classes in this file depending
        on the desired use case.
    """

    def __init__(self,
                 embedding_dimension: int
                 ) -> None:
        """
        Args:
            embedding_dimension (int): The expected dimension size for each token embedding.
        Attributes:
            exponent (torch.Tensor): The exponent values used in the sinusoidal equation. In
                graphs, this is the "2i/d_model" portion that 10,000 is raised to. For an
                example embedding dimension of 10, this would result in the tensor
                [0, 2, 4, 6, 8] / 10, or [0.0, 0.2, 0.4, 0.6, 0.8].
        """
        self.exponent = torch.arange(0, embedding_dimension, 2) / embedding_dimension

    def _find_sin_and_cos_embeddings_of_given_values(self, values):
        output_values = values / (10_000**self.exponent.to(values.device))
        return torch.sin(output_values), torch.cos(output_values)

class SinusoidalPositionEmbedding(SinusoidalEmbedding):
    """
    Child class of SinusoidalEmbedding that specifically encodes sequence position as a sinusoidal embedding.
    """

    def __init__(self,
                 embedding_dimension: int,
                 max_len: int = 5_000
                 ) -> None:
        """
        Args:
            embedding_dimension (int): See parent class.
            max_len(int, optional): The maximum expected length of any sequence. Defaults to
                5_000.
        """
        super().__init__(embedding_dimension)
        self._initialize_all_possible_positional_encodings(embedding_dimension, max_len)

    def _initialize_all_possible_positional_encodings(self, embedding_dimension, max_len):
        self.positional_encoding = torch.zeros(max_len, embedding_dimension)
        self.positional_encoding.requires_grad = False
        positions = torch.arange(0, max_len).unsqueeze(1)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(positions)
        self.positional_encoding[:, 0::2] = sin
        self.positional_encoding[:, 1::2] = cos

    def __call__(self,
                 x: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The embedded input tensor, of shape (batch_size, seqeunce_length, embedding_dimension)
]        Returns:
            torch.Tensor: The output tensor represents custom value encoding, of shape
                    (sequence_length, embedding_dimension). Adding is broadcasted across samples in a batch.
        """

        _, sequence_length, _ = x.shape
        return self.positional_encoding[:sequence_length, :].to(x.device)

class SinusoidalCustomEmbedding(SinusoidalEmbedding):
    """
    Child class of SinusoidalEmbedding that specifically encodes custom numeric input as a sinusoidal embedding.
    """

    def __call__(self,
                 x: torch.Tensor,
                 custom_values: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The embedded input tensor, of shape (batch_size, seqeunce_length, embedding_dimension)
            custom_values (torch.Tensor): The provided custom values to be encoded, of shape
            (batch_size, sequence_length).
        Returns:
            torch.Tensor: The output tensor represents custom value encoding, of shape
                    (batch_size, sequence_length, embedding_dimension).
        """

        custom_encoding = torch.zeros_like(x)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(
            custom_values.unsqueeze(2)
        )
        custom_encoding[:, :, 0::2] = sin
        custom_encoding[:, :, 1::2] = cos
        return custom_encoding.to(x.device)
