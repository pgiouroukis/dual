from typing import List
from sklearn.metrics.pairwise import euclidean_distances
from src.active_learning.active_learning_strategy_base import ActiveLearningStrategyBase
import numpy as np
import torch

class ActiveLearningStrategyCoreset(ActiveLearningStrategyBase):
    def requires_warmup(self) -> bool:
        return True
    
    def requires_embeddings(self) -> bool:
        return True
    
    def acquire_samples(self) -> List[int]:
        samples_per_iteration = self.active_learning_args["active_learning_samples_per_iteration"]

        # Calculate embeddings for labeled and unlabeled datasets
        labelled_embeddings = self.embeddings[self.dataset_handler.get_labelled_idxs()].cpu().numpy()
        unlabelled_embeddings = self.embeddings[self.dataset_handler._get_unlabelled_idxs()].cpu().numpy()

        # Initialize core-set selection with labeled points
        selected_idxs = []
        selected_embeddings = labelled_embeddings

        # Core-set selection using farthest-first traversal
        for _ in range(samples_per_iteration):
            # Calculate distances from selected embeddings to all unlabelled embeddings
            distances = euclidean_distances(unlabelled_embeddings, selected_embeddings)
            min_distances = distances.min(axis=1)

            # Select the unlabelled sample with the maximum minimum distance to the labeled set
            farthest_point_idx = min_distances.argmax()
            selected_idxs.append(self.dataset_handler._get_unlabelled_idxs()[farthest_point_idx])

            # Update selected embeddings to include the newly selected point
            selected_embeddings = np.vstack([selected_embeddings, unlabelled_embeddings[farthest_point_idx:farthest_point_idx+1]])

            # Remove the selected point from the unlabelled set
            unlabelled_embeddings = np.delete(unlabelled_embeddings, farthest_point_idx, axis=0)

        return selected_idxs
