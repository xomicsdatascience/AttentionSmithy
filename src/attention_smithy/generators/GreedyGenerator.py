import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class GreedyGenerator(GeneratorStrategy):
    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        src_embedding: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ):
        current_sequence_length = tgt_input.size(1)
        output = tgt_input.clone()
        for step in range(current_sequence_length-1, maximum_sequence_length):
            all_logits = model.forward_decode(output)
            next_token_logits = all_logits[:, step, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)
            output = torch.cat([output, next_token_ids.unsqueeze(-1)], dim=1)
            if output[0][-1] == end_token:
                return output
        return output

