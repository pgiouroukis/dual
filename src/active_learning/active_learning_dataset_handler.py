import random
from typing import List, Tuple
from datasets import Dataset

class ActiveLearningDatasetHandler:
    def __init__(self, dataset: Dataset, dataset_args: dict):
        self.dataset = dataset
        self.dataset_args = dataset_args
        self.labelled_idxs: List[int] = []

    def _get_unlabelled_idxs(self) -> List[int]:
        return [idx for idx in range(len(self.dataset)) if idx not in self.labelled_idxs]

    # Sample data from the dataset. The sampled data are not considered labelled. 
    # Invoke method label_samples separately to label the samples.
    def sample_unlabelled(self, num_samples: int) -> Tuple[Dataset, List[int]]:
        unlablled_idxs = self._get_unlabelled_idxs()
        random.shuffle(unlablled_idxs)
        unlablled_idxs = unlablled_idxs[:num_samples]
        sampled_dataset = self.dataset.select(unlablled_idxs)
        return sampled_dataset, unlablled_idxs

    def label_samples(self, sample_idxs: List[int]) -> None:
        assert all(item not in self.labelled_idxs for item in sample_idxs), "Some of the given samples are already labelled"
        self.labelled_idxs += sample_idxs

    def get_labelled_dataset(self) -> Dataset:
        dataset = self.dataset.select(self.labelled_idxs).add_column(name="index", column=self.labelled_idxs, new_fingerprint="manual_fingerprint")
        return dataset

    def get_labelled_count(self) -> int:
        return len(self.labelled_idxs)
    
    def get_labelled_idxs(self) -> List[int]:
        return self.labelled_idxs
