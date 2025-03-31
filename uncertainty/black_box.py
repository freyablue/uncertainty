import torch as t
import numpy as np
import itertools
from sentence_transformers import CrossEncoder


class BlackBox:
    def __init__(self, args):
        self.args = args
        self.crossencoder_setup = False

    def _setup(self, device="auto"):
        self.crossencoder = CrossEncoder(
            "cross-encoder/stsb-roberta-large", device=device
        )

    def generate_similarity_matrix(self, texts):
        if not self.crossencoder_setup:
            self._setup(device='cuda')
            self.crossencoder_setup = True

        # Generate all possible pairs from the list of texts
        unique_texts, inv = np.unique(texts, return_inverse=True)
        text_pairs = list(itertools.product(unique_texts, unique_texts))

        # Calculate similarity scores for all pairs
        sim_scores = self.crossencoder.predict(text_pairs, batch_size=1)

        # Reshape the flat similarity scores array into a matrix
        num_unique_texts = len(unique_texts)
        sim_scores_matrix = sim_scores.reshape(num_unique_texts, num_unique_texts)

        # Recover the full matrix from unique texts by gathering along both axes using the inverse index
        full_sim_matrix = sim_scores_matrix[inv, :][:, inv]

        return full_sim_matrix


    def mean_sim_score(self, texts):
        sim_matrix = self.generate_similarity_matrix(texts)
        # average similarity except diagonal elements
        return float(np.mean(sim_matrix[np.eye(sim_matrix.shape[0]) == 0]))


    def compare_two_responses(self, pred1, pred2):

        scores = []

        if len(pred1) == 0 or len(pred2) == 0:
            return 0

        for x1 in pred1:
            if x1 in pred2:
                scores.append(1)
            else:
                scores.append(0)

        return np.mean(scores)


    def mean_exact_score(self, texts):
        scores = []

        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i == j:
                    continue

                scores.append(self.compare_two_responses(text1, text2))

        return np.mean(scores)
