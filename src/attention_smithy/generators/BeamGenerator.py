import torch
from torch import nn
from attention_smithy.generators.GeneratorStrategy import GeneratorStrategy

class BeamGenerator(GeneratorStrategy):
    """
    A beam search generator strategy class.

    NOTE: 3 is set as the default beam_width, and for readibility the number 3
        is used in the below explanation. However, this is adjustable by the end
        user in the __init__ function.

    Beam search refers to a method that sustains several possible options while
        generating a sequence. Whereas a greedy algorithm always just picks the
        next best token and adds it immediately, beam search will find the top 3
        tokens and create a "beam" sequence for each. This allows the generator
        to find long-term optimal solutions that might not always be found through
        the singular "greedy" path.

    To begin, the generator will take the provided input, find the top 3 tokens,
        and initialize 3 different beam sequences, one for each token. All 3 beams
        will then be passed through the model, and the best 3 tokens for each beam
        will be calculated. This ultimately spawns 9 beams. Only the top 3 beams of
        these 9 would then be kept, and this iteration occurs until all 3 best beams
        have reached an end token.

    Scoring is done through the log probabilities of the next tokens. When a new token
        is added, the log probability of that token is added to the current
        score of that beam, then this new additive score is divided by a length penalty
        to normalize it. This normalization still prioritizes longer sequences, but
        not as dramatically as it would be without the normalization.

    A distinction is made between "finished" beams and "unfinished" beams. Finished
        beams have reached an end token, and therefore should not have any other tokens
        added to it. These are partitioned off from the unfinished beams that are still
        being added to. After the unfinished beams have been passed through the model
        and updated with new tokens and scores, they are recombined with the finished
        beams (the finished beams have an extra end token added to normalize lengths,
        with no change to their score). They are then all evaluated together, where the
        best 3 beams - finished or unfinished - are sent back to the beginning loop

    The program ends when all 3 optimal beams are finished, or if the maximum length is
        reached. In both cases, the best beam of those 3 is returned (with trailing end
        tokens removed if relevant).
    """
    def __init__(self,
                 beam_width:int = 3,
                 length_penalty_alpha:float = 0.6,
                 no_repeat_ngram_size: int = 0,
                 ) -> None:
        """
        Args:
            beam_width (int): The number of beams to prioritize on each pass of the token
                generation loop. Default is 3.
            length_penalty_alpha (float): Adjusts the length penalty used in calculating
                each score after a token is added. Default is 0.6.
        """
        super().__init__(no_repeat_ngram_size)
        self.beam_width = beam_width
        self.length_penalty_alpha = length_penalty_alpha

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
        self.end_token = end_token
        beams, scores = self._run_input_through_model_and_attach_best_next_tokens_to_separate_beams(model, tgt_input, **kwargs)
        for step in range(tgt_input.size(1), maximum_sequence_length):
            beams, scores = self._add_tokens_to_branching_beams_and_prune_low_scoring_branches(beams, model, scores, step, **kwargs)
            if torch.all(self.mask_for_beams_that_have_reached_the_end_token) or beams.shape[1] > maximum_sequence_length:
                best_beam = self._identify_best_beam_from_all_finished_beams(beams, scores)
                return best_beam
        best_beam_index = torch.argmax(scores)
        return beams[best_beam_index]

    def _add_tokens_to_branching_beams_and_prune_low_scoring_branches(self, beams, model, scores, step, **kwargs):
        finished_beams, finished_scores, unfinished_beams, unfinished_scores = self._separate_finished_beams_from_unfinished(
            beams, scores)
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(
            unfinished_beams, step, model, **kwargs)
        expanded_beams = self._add_next_tokens_to_unfinished_beams_then_combine_with_previously_finished_beams(
            finished_beams, finished_scores, top_k_indices_across_beams, unfinished_beams)
        expanded_scores = self._update_unfinished_scores_then_combine_with_finished_scores(finished_scores, step,
                                                                                           top_k_probabilities_across_beams,
                                                                                           unfinished_scores)
        beams, scores = self._prune_low_scoring_beams_to_only_keep_beam_width_amount(expanded_beams,
                                                                                   expanded_scores)
        return beams, scores

    def _run_input_through_model_and_attach_best_next_tokens_to_separate_beams(self, model, tgt_input, **kwargs):
        initial_sequence_length = tgt_input.size(1)
        initial_beam = tgt_input.clone()
        beams, scores = self._initialize_beams_and_corresponding_scores(initial_beam, initial_sequence_length, model, **kwargs)
        self.mask_for_beams_that_have_reached_the_end_token = (beams == self.end_token).any(dim=1)
        return beams, scores

    def _identify_best_beam_from_all_finished_beams(self, beams, scores):
        best_beam_index = torch.argmax(scores)
        best_beam = beams[best_beam_index]
        best_beam = self._remove_trailing_end_tokens_from_best_beam(best_beam)
        return best_beam

    def _remove_trailing_end_tokens_from_best_beam(self, best_beam):
        indices = torch.where(best_beam == self.end_token)[0]
        if indices.numel() > 0:
            end_idx = torch.min(indices) + 1
        else:
            end_idx = 1
        if end_idx == 1:
            return torch.cat((best_beam, torch.tensor([self.end_token]).to(best_beam.device)))
        return best_beam[:end_idx]

    def _prune_low_scoring_beams_to_only_keep_beam_width_amount(self, expanded_beams, expanded_scores):
        _, top_score_indices = torch.topk(expanded_scores, k=self.beam_width)
        beams = expanded_beams[top_score_indices].to(torch.int64)
        scores = expanded_scores[top_score_indices]
        self.mask_for_beams_that_have_reached_the_end_token = (beams == self.end_token).any(dim=1)
        return beams, scores

    def _update_unfinished_scores_then_combine_with_finished_scores(self, finished_scores, step,
                                                                    top_k_probabilities_across_beams,
                                                                    unfinished_scores):
        expanded_old_scores = unfinished_scores.repeat_interleave(self.beam_width)
        current_sequence_length = step + 1
        length_penalty = ((5 + step) ** self.length_penalty_alpha) / ((5 + 1) ** self.length_penalty_alpha)
        expanded_new_scores = (expanded_old_scores + top_k_probabilities_across_beams.flatten()) / length_penalty
        expanded_scores = torch.cat((expanded_new_scores, finished_scores), dim=0)
        return expanded_scores

    def _add_next_tokens_to_unfinished_beams_then_combine_with_previously_finished_beams(self, finished_beams,
                                                                               finished_scores,
                                                                               top_k_indices_across_beams,
                                                                               unfinished_beams):
        expanded_unfinished_beams = unfinished_beams.repeat_interleave(self.beam_width, dim=0)
        expanded_beams = torch.cat((expanded_unfinished_beams, finished_beams), dim=0)
        beam_additions = torch.cat(
            (top_k_indices_across_beams.flatten(), torch.tensor([self.end_token for i in range(len(finished_scores))]).to(expanded_beams.device)))
        expanded_beams_with_new_additions = torch.cat((expanded_beams, beam_additions.unsqueeze(1)), dim=1)
        return expanded_beams_with_new_additions

    def _separate_finished_beams_from_unfinished(self, beams, scores):
        finished_beams = beams[self.mask_for_beams_that_have_reached_the_end_token]
        finished_scores = scores[self.mask_for_beams_that_have_reached_the_end_token]
        unfinished_beams = beams[~self.mask_for_beams_that_have_reached_the_end_token]
        unfinished_scores = scores[~self.mask_for_beams_that_have_reached_the_end_token]
        return finished_beams, finished_scores, unfinished_beams, unfinished_scores

    def _initialize_beams_and_corresponding_scores(self, initial_beam, initial_sequence_length, model, **kwargs):
        top_k_indices_across_beams, top_k_probabilities_across_beams = self._run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(
            initial_beam, initial_sequence_length - 1, model, **kwargs)
        beams = initial_beam.repeat(self.beam_width, 1)
        beams = torch.cat((beams, top_k_indices_across_beams.transpose(0, 1)), dim=1)
        scores = top_k_probabilities_across_beams.flatten()
        return beams, scores

    def _run_unfinished_beams_through_model_and_get_highest_probability_next_tokens(self, beams, step,
                                                                         model, **kwargs):
        all_logits = model.forward_decode(beams, **kwargs)
        if hasattr(all_logits, 'logits'):
            all_logits = all_logits.logits
        next_token_logits = all_logits[:, step, :]
        log_probabilities = torch.log(torch.softmax(next_token_logits, dim=-1))
        self._apply_ngram_repeating_restraints(beams, log_probabilities)
        top_k_probabilities_across_beams, top_k_indices_across_beams = torch.topk(
            log_probabilities, self.beam_width, dim=-1
        )
        return top_k_indices_across_beams, top_k_probabilities_across_beams



