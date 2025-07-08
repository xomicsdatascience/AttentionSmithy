import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy


class GreedyGenerator(GeneratorStrategy):
    """
    A greedy generator strategy class that supports batch processing.

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
        unfinished_sequences_mask = torch.ones(output.shape[0], dtype=torch.bool, device=output.device)

        for step in range(current_sequence_length - 1, maximum_sequence_length):
            output = self._add_best_next_tokens_to_output(model, output, step, **kwargs)
            self._update_unfinished_sequences_mask(output, unfinished_sequences_mask, end_token)

            if not unfinished_sequences_mask.any():
                break

        return output

    def _add_best_next_tokens_to_output(self, model, output, step, **kwargs):
        all_logits = model.forward_decode(output, **kwargs)
        if hasattr(all_logits, 'logits'):
            all_logits = all_logits.logits

        next_token_logits = all_logits[:, step, :]
        self._apply_ngram_repeating_restraints(output, next_token_logits)
        next_token_ids = torch.argmax(next_token_logits, dim=-1)

        return torch.cat([output, next_token_ids.unsqueeze(-1)], dim=1)

    def _update_unfinished_sequences_mask(self, output, unfinished_sequences_mask, end_token):
        newest_tokens = output[:, -1]
        sequences_just_finished = (newest_tokens == end_token)
        unfinished_sequences_mask[sequences_just_finished] = False