import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class BeamGenerator(GeneratorStrategy):
    def __init__(self, beam_width=3):
        super().__init__()
        self.beam_width = beam_width

    def generate_sequence(
        self,
        model: nn.Module,
        end_token: int,
        tgt_input: torch.Tensor,
        src_embedding: torch.Tensor,
        maximum_sequence_length: int,
        **kwargs,
    ):
        initial_sequence_length = tgt_input.size(1)
        initial_beam = tgt_input.clone()
        all_logits = model.forward_decode(initial_beam)
        next_token_logits = all_logits[:, initial_sequence_length-1, :]
        log_probabilities = torch.log(torch.softmax(next_token_logits, dim=-1))
        top_k_probabilities_across_beams, top_k_indices_across_beams = torch.topk(
            log_probabilities, self.beam_width, dim=-1
        )
        beams = initial_beam.repeat(self.beam_width, 1)
        beams = torch.cat((beams, top_k_indices_across_beams.transpose(0, 1)), dim=1)
        scores = top_k_probabilities_across_beams.flatten()
        print()
        for step in range(initial_sequence_length, maximum_sequence_length):
            finished_beam_mask = (beams == end_token).any(dim=1)
            finished_beams = beams[finished_beam_mask]
            finished_scores = scores[finished_beam_mask]
            unfinished_beams = beams[~finished_beam_mask]
            unfinished_scores = scores[~finished_beam_mask]
            all_logits = model.forward_decode(unfinished_beams)
            next_token_logits = all_logits[:, step, :]
            log_probabilities = torch.log(torch.softmax(next_token_logits, dim=-1))
            top_k_probabilities_across_beams, top_k_indices_across_beams = torch.topk(
                log_probabilities, self.beam_width, dim=-1
            )

            expanded_unfinished_beams = unfinished_beams.repeat_interleave(3, dim=0)
            expanded_beams = torch.cat((expanded_unfinished_beams, finished_beams), dim=0)
            beam_additions = torch.cat((top_k_indices_across_beams.flatten(), torch.tensor([end_token for i in range(len(finished_scores))])))
            expanded_beams_with_new_additions = torch.cat((expanded_beams, beam_additions.unsqueeze(1)), dim=1)

            expanded_old_scores = unfinished_scores.repeat_interleave(3)
            current_sequence_length = step + 1
            alpha = 0.6
            length_penalty = (current_sequence_length + 1) ** alpha / (current_sequence_length + 1)
            expanded_new_scores = (expanded_old_scores + top_k_probabilities_across_beams.flatten()) / length_penalty
            expanded_scores = torch.cat((expanded_new_scores, finished_scores), dim=0)

            _, top_score_indices = torch.topk(expanded_scores, k=self.beam_width)

            beams = expanded_beams_with_new_additions[top_score_indices].to(torch.int64)
            scores = expanded_scores[top_score_indices]
            reached_end_token_mask = (beams == end_token).any(dim=1)
            print('\n'*5)
            print(beams)
            print(scores)
            if torch.all(reached_end_token_mask):
                best_beam_index = torch.argmax(scores)
                best_beam = beams[best_beam_index]
                end_idx = torch.min(torch.where(best_beam == end_token)[0]) + 1
                best_beam = best_beam[:end_idx]
                return best_beam

        best_beam_index = torch.argmax(scores)
        return beams[best_beam_index]



