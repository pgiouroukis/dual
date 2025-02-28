from typing import List
from src.active_learning.active_learning_strategy_base import ActiveLearningStrategyBase

class ActiveLearningStrategyRandom(ActiveLearningStrategyBase):
    def requires_warmup(self) -> bool:
        return False
    
    def requires_embeddings(self) -> bool:
        return False
    
    def acquire_samples(self) -> List[int]:
        samples_per_iteration = self.active_learning_args["active_learning_samples_per_iteration"]
        _, dataset_idxs = self.dataset_handler.sample_unlabelled(samples_per_iteration)
        return dataset_idxs
