import logging
import torch
import json
from typing import List
from src.active_learning.active_learning_strategy_base import ActiveLearningStrategyBase
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ActiveLearningStrategyIDDS(ActiveLearningStrategyBase):
    def __init__(
        self,
        dataset_handler,
        model_name,
        train_args,
        dataset_args,
        active_learning_args,
        embedding_args,
        min_train_steps,
        train_validation,
        train_validation_samples,
        evaluation_dataset,
        validation_dataset,
    ):
        super().__init__(
            dataset_handler,
            model_name,
            train_args,
            dataset_args,
            active_learning_args,
            embedding_args,
            min_train_steps,
            train_validation,
            train_validation_samples,
            evaluation_dataset,
            validation_dataset,
        )
        self._compute_embeddings_similarities_sum()

    def requires_warmup(self) -> bool:
        return False

    def requires_embeddings(self) -> bool:
        return True
    
    def acquire_samples(self) -> List[int]:
        samples_per_iteration = self.active_learning_args["active_learning_samples_per_iteration"]
        
        logging.info(f"Acquiring {samples_per_iteration} samples using IDDS")
        sample_dataset_idxs = []
        sample_dataset_scores = []
        for i in range(samples_per_iteration):
            idds_scores = self._compute_idds_scores(sample_dataset_idxs)
            max_score_idx = int(idds_scores.argmax().item())
            sample_dataset_idxs.append(max_score_idx)
            sample_dataset_scores.append(idds_scores[max_score_idx].item())
            logging.info(f"Selecting the document with the highest IDDS score (index: {max_score_idx}, score: {idds_scores[max_score_idx]})")
            logging.info(f"For reference, the IDDS score of the document with the lowest score is {idds_scores.min()}")

        ActiveLearningStrategyIDDS._log_metrics_for_samples(self, sample_dataset_idxs, sample_dataset_scores, "ActiveLearningStrategyIDDS")

        return sample_dataset_idxs


    def _compute_embeddings_similarities_sum(self, batch_size = 7500) -> None:
        """
        Computes the sum of similarities between each embedding in the dataset and all other embeddings.
        We use this to compute the IDDS score for each embedding.
        The similarities are computed in batches to avoid memory issues.

        Returns:
            similarities_sum (torch.Tensor): A tensor containing the sum of similarities for each embedding.
        """
        dataset_len = self.embeddings.shape[0]
        similarities_sum = torch.zeros(dataset_len).to(self.train_args.device)
        tensor_dataset = TensorDataset(self.embeddings)
        tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        for index, (batch_embeddings,) in tqdm(enumerate(tensor_dataloader)):
            similarities = torch.mm(batch_embeddings, self.embeddings.T)
            assert(similarities.shape == (len(batch_embeddings), dataset_len))

            similarities_sum[index * batch_size : (index+1) * batch_size] = similarities.sum(dim=1)

        documents = self.dataset_handler.dataset[self.dataset_args["dataset_document_column"]]
        logging.info(f"Document with lowest similarity sum: {documents[similarities_sum.argmin().item()]}")
        logging.info(f"Document with highest similarity sum: {documents[similarities_sum.argmax().item()]}")

        self.similarities_sum = similarities_sum

    def _compute_idds_scores(self, new_labelled_idxs: List[int]) -> torch.Tensor:
        dataset_count = self.embeddings.shape[0]
        labelled_idxs = self.dataset_handler.get_labelled_idxs() + new_labelled_idxs
        labelled_count = len(labelled_idxs) 
        labelled_embeddings = self.embeddings[labelled_idxs]

        if len(labelled_embeddings) > 0:
            similarities_with_labelled = torch.mm(self.embeddings, labelled_embeddings.T)
            assert similarities_with_labelled.shape == (dataset_count, labelled_count)

            similarities_sum_with_labelled = torch.sum(similarities_with_labelled, dim=1).to(self.train_args.device)
        else:
            similarities_sum_with_labelled = torch.zeros(dataset_count).to(self.train_args.device)

        similarities_with_unlabelled_sum = self.similarities_sum - similarities_sum_with_labelled
        assert similarities_with_unlabelled_sum.shape == (dataset_count,)

        unlabelled_count = dataset_count - labelled_count

        labelled_score = similarities_sum_with_labelled / (labelled_count if labelled_count > 0 else 1)
        unlabelled_score = similarities_with_unlabelled_sum / unlabelled_count
        idds_scores = 0.66 * unlabelled_score - 0.33 * labelled_score
        assert idds_scores.shape == (dataset_count,)
        
        idds_scores[labelled_idxs] = 0

        return idds_scores

    def _log_metrics_for_samples(self, sample_dataset_idxs: List[int], sample_dataset_scores: List[float], source: str) -> None:
        sample_embeddings = self.embeddings[sample_dataset_idxs]
        sample_embeddings_variance = self._compute_embeddings_variance(sample_embeddings)
        sample_embeddings_avg_pairwise_distance = self._compute_embeddings_avg_pairwise_distance(sample_embeddings)
        sample_embeddings_avg_cosine_similarity = self._compute_embeddings_avg_cosine_similarity(sample_embeddings)
        sample_embeddings_avg_distance_to_centroid = self._compute_embeddings_avg_distance_to_centroid(sample_embeddings)
        json_metrics = json.dumps({
            "source": source,
            "variance": sample_embeddings_variance,
            "avg_pairwise_dist": sample_embeddings_avg_pairwise_distance,
            "avg_cosine_similarity": sample_embeddings_avg_cosine_similarity,
            "avg_distance_to_centroid": sample_embeddings_avg_distance_to_centroid,
        })
        with open(self.train_args.output_dir + "/idds_embeddings_stats.json", "a") as outfile:
            outfile.write(json_metrics + '\n')

        index_and_scores = [{"index": idx, "score": score} for idx, score in zip(sample_dataset_idxs, sample_dataset_scores)]
        json_scores = json.dumps({
            "source": source,
            "index_and_scores": index_and_scores,
        })
        with open(self.train_args.output_dir + "/idds_scores.json", "a") as outfile:
            outfile.write(json_scores + '\n')

    def _compute_embeddings_variance(self, embeddings: torch.Tensor) -> float:
        return embeddings.var(dim=0).mean().item()
    
    def _compute_embeddings_avg_pairwise_distance(self, embeddings: torch.Tensor) -> float:
        return torch.nn.functional.pdist(embeddings).mean().item()
    
    def _compute_embeddings_avg_cosine_similarity(self, embeddings: torch.Tensor) -> float:
        similarities = torch.mm(embeddings, embeddings.T)
        return similarities.mean().item()

    def _compute_embeddings_avg_distance_to_centroid(self, embeddings: torch.Tensor) -> float:
        centroid = embeddings.mean(dim=0)
        distances = torch.nn.functional.pairwise_distance(embeddings, centroid)
        return distances.mean().item()
