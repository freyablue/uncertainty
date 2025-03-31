import torch as t
import numpy as np

class WhiteBox:
    def __init__(self, tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args



    def top_logits(self, scores, lo=0, hi=None, top_k=5, normalize=True):
        # scores has shape (response_length, num_responses, vocab_size)
        scores = scores[lo:hi, ::] 
        if len(scores) == 0: # Handle case where no scores are found
            return (0, None)
        if normalize:
            scores = t.exp(scores) / t.sum(t.exp(scores), dim=1, keepdim=True)
        
        # Find top k scores and their indexes for the specific response
        # Assuming scores are already in logit form after normalization if applied
        top_scores, top_indexes = t.topk(scores, top_k, dim=1)
        top_scores = top_scores[0]
        top_indexes = top_indexes[0]

        
        vocab_logit_map = {}
        for i in range(len(top_scores)): 
            # Decode index to vocab here, assuming a placeholder `decode_token` method exists
            
            vocab = self.tokenizer.decode([top_indexes[i].item()]) # Simplified; likely need adjustment
            # Store the minimum of the top scores for this vocab across the positions
            vocab_logit_map[vocab] = top_scores[i].item()

        return vocab_logit_map


    def token_idxs_of_targets(self, text, llm_answers):
        # Find the first index of a target in the text. Then find the index of the corresponding token in the tokenized version of the text
        if type(llm_answers) == str:
            llm_answers = [llm_answers]

        answer_idxs = [text.rfind(answer) for answer in llm_answers if text.rfind(answer) != -1]

        token_idxs = []

        if len(answer_idxs) > 0:
            tokens = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

            for idx, answer_idx in enumerate(answer_idxs):
                # Tokenize the text and get offset mapping
                found = False
                # First, try to find the exact match
                for token_index, (start, end) in enumerate(tokens.offset_mapping):
                    if start <= answer_idx < end:
                        token_idxs.append(token_index)
                        found = True
                        break

                if not found:
                    for token_index, (start, end) in enumerate(tokens.offset_mapping):
                        if start <= answer_idx <= end:
                            token_idxs.append(token_index)
                            found = True
                            break

                if not found:
                    for token_index, (start, end) in enumerate(tokens.offset_mapping):
                        if start <= answer_idx <= end+1:
                            token_idxs.append(token_index)
                            found = True
                            break

                # If exact match is not found, try with indexed answers (e.g., "1. answer1")
                if not found:
                    numbered_answer = f" {llm_answers[idx]}"
                    numbered_answer_idx = text.find(numbered_answer)
                    if numbered_answer_idx != -1:
                        for token_index, (start, end) in enumerate(tokens.offset_mapping):
                            if start <= numbered_answer_idx <= end:
                                token_idxs.append(token_index)
                                found = True
                                break
                

                # If still not found, perform a more flexible search
                if not found:
                    for token_index, (start, end) in enumerate(tokens.offset_mapping):
                        if start <= answer_idx <= end or (start < answer_idx and answer_idx < end):

                            token_idxs.append(token_index)
                            found = True
                            break

                # If still not found, search for the closest match by partial token match
                if not found:
                    for token_index, (start, end) in enumerate(tokens.offset_mapping):
                        if (start <= answer_idx < end) or (answer_idx <= start < end) or (start < answer_idx and end > answer_idx):
                            token_idxs.append(token_index)
                            break

       
        return token_idxs

   

    def get_top_logits(self, text, llm_answers, scores, top_k=5, normalize=True):
        token_idxs = self.token_idxs_of_targets(text, llm_answers)
        top_vocab_logit_maps = []
        for token_idx in token_idxs:
            lo = token_idx
            hi = token_idx + 1
            top_vocab_logit_map = self.top_logits(scores, lo=lo, hi=hi, top_k=top_k, normalize=normalize)
            if type(top_vocab_logit_map) != dict:
                continue
            top_vocab_logit_maps.append(top_vocab_logit_map)

        return top_vocab_logit_maps


    def mean_top_logits(self, top_vocab_logit_maps):
        top_logits = [max(d.values()) for d in top_vocab_logit_maps]
        mean_top_logits = np.mean(top_logits)
        # check is nan
        if np.isnan(mean_top_logits):
            return 0
        return mean_top_logits


    def max_list_top_logits(self, top_vocab_logit_maps):
        top_logits = [max(d.values()) for d in top_vocab_logit_maps]
        return top_logits

    def margin_list_top_logits(self, top_vocab_logit_maps):
        margins = []
        for d in top_vocab_logit_maps:
            top_2 = sorted(d.values(), reverse=True)[:2]
            margin = top_2[0] - top_2[1]
            margins.append(margin)
        return margins

    def entropy_list_top_logits(self, top_vocab_logit_maps):
        entropy = []
        for d in top_vocab_logit_maps:
            top_logits = d.values()
            entropy.append(sum([p*np.log(p) for p in top_logits]))
        return entropy

    def mean_greedy_entropy(self, top_vocab_logit_maps):
        top_logits = [max(d.values()) for d in top_vocab_logit_maps]
        entropy = [-p*np.log(p) for p in top_logits]
        mean_greedy_entropy_val = np.mean(entropy)
        # check is nan
        if np.isnan(mean_greedy_entropy_val):
            return 0
        return mean_greedy_entropy_val