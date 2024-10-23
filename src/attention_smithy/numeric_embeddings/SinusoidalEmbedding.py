import torch


class SinusoidalEmbedding:
    def __init__(self, d_model, max_len=5000):
        self.exponent = torch.arange(0, d_model, 2) / d_model
        self._initialize_all_possible_positional_encodings(d_model, max_len)

    def __call__(self, x, custom_values=None):
        if custom_values == None:
            _, sequence_length, _ = x.shape
            return self.positional_encoding[:sequence_length, :].to(x.device)
        else:
            custom_value_encoding = self._encode_custom_values(x, custom_values)
            return custom_value_encoding

    def _initialize_all_possible_positional_encodings(self, d_model, max_len):
        self.positional_encoding = torch.zeros(max_len, d_model)
        self.positional_encoding.requires_grad = False
        positions = torch.arange(0, max_len).unsqueeze(1)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(positions)
        self.positional_encoding[:, 0::2] = sin
        self.positional_encoding[:, 1::2] = cos

    def _encode_custom_values(self, x, custom_values):
        sinusoidal_embedding = torch.zeros_like(x)
        sin, cos = self._find_sin_and_cos_embeddings_of_given_values(
            custom_values.unsqueeze(2)
        )
        sinusoidal_embedding[:, :, 0::2] = sin
        sinusoidal_embedding[:, :, 1::2] = cos
        return sinusoidal_embedding

    def _find_sin_and_cos_embeddings_of_given_values(self, initial_values):
        values = initial_values / (10_000**self.exponent)
        return torch.sin(values), torch.cos(values)
