import torch


class SinusoidalEmbedding:
    """
    A class that encodes numeric values as tokens. It relies on sinusoidal equations described in the original
        "Attention Is All You Need" paper.
    WHEN APPLIED: Generally added to token embeddings at the beginning of the model.
    NOTE: These numeric encodings typically encode position (0, 1, 2 etc.). However, the class can encode
        customized values as well.
    """

    def __init__(self, embedding_dimension, max_len=5_000):
        """
        Initializes the embedding class.

        Args:
            embedding_dimension (int): The expected dimension size for each token embedding.
            max_len(int, optional): The maximum expected length of any sequence. Defaults to
                5_000.
        """

        self.exponent = torch.arange(0, embedding_dimension, 2) / embedding_dimension
        self._initialize_all_possible_positional_encodings(embedding_dimension, max_len)



    def __call__(self, x, custom_values=None):
        """
        Args:
            x (torch.Tensor): The embedded input tensor, of shape (batch_size, seqeunce_length, embedding_dimension)
            custom_values (torch.Tensor, optional): If given, the provided custom values will replace
                the position index matrix. Of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: The output tensor. Two possible shapes as output, both of which addable to
              encoded tokens:
                - no custom values provided: Represents positional encoding, of shape
                    (sequence_length, embedding_dimension). Adding is broadcasted across a batch.
                - custom values provided: represents custom value encoding, of shape
                    (batch_size, sequence_length, embedding_dimension).
        """

        if custom_values == None:
            _, sequence_length, _ = x.shape
            return self.positional_encoding[:sequence_length, :].to(x.device)
        else:
            custom_value_encoding = self._encode_custom_values(x, custom_values)
            return custom_value_encoding

    def _initialize_all_possible_positional_encodings(self, embedding_dimension, max_len):
        self.positional_encoding = torch.zeros(max_len, embedding_dimension)
        self.positional_encoding.requires_grad = False
        positions = torch.arange(0, max_len).unsqueeze(1)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(positions)
        self.positional_encoding[:, 0::2] = sin
        self.positional_encoding[:, 1::2] = cos

    def _encode_custom_values(self, x, custom_values):
        custom_encoding = torch.zeros_like(x)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(
            custom_values.unsqueeze(2)
        )
        custom_encoding[:, :, 0::2] = sin
        custom_encoding[:, :, 1::2] = cos
        return custom_encoding

    def _find_sin_and_cos_embeddings_of_given_values(self, initial_values):
        values = initial_values / (10_000**self.exponent)
        return torch.sin(values), torch.cos(values)
