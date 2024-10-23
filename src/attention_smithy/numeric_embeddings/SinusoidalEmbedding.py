import torch

class SinusoidalEmbedding:
    def __init__(self, d_model, max_len=5000):
        self.positional_encoding = torch.zeros(max_len, d_model)
        self.positional_encoding.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)
        self.positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def __call__(self, x):
        batch_size, seq_len, d_model = x.size()
        return self.encoding[:seq_len, :].to(x.device)
