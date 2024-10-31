import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class GreedyGenerator(GeneratorStrategy):
    """
    A greedy generator strategy class.

    "Greedy" is a computer science term effectively meaning "takes the immediate best option
        every time." In this case, a sequence of tokens is generated one by one, token by token,
        always picking the "best" next token. This is not always optimal - it takes an
        immediate, short-term approach to creating sequences. Other optimizations, like
        the beam search, try to create multiple possible threads and pick the best ones
        over time. Not so here - it is a simple, "just give me the best next token"
        approach.
    """
    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        See GeneratorContext class.
        """
        current_sequence_length = tgt_input.size(1)
        output = tgt_input.clone()
        for step in range(current_sequence_length-1, maximum_sequence_length):
            output = self._add_best_next_token_to_output(model, output, step, **kwargs)
            if output[0][-1] == end_token:
                return output
        return output

    def _add_best_next_token_to_output(self, model, output, step, **kwargs):
        all_logits = model.forward_decode(output, **kwargs)
        next_token_logits = all_logits[:, step, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        output = torch.cat([output, next_token_id.unsqueeze(-1)], dim=1)
        return output

