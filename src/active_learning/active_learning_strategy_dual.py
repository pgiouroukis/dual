import json
from math import log
import torch
import logging
import numpy as np
from typing import List
from src.active_learning.active_learning_strategy_base import ActiveLearningStrategyBase
from src.active_learning.active_learning_strategy_bas import ActiveLearningStrategyBAS
from src.active_learning.active_learning_strategy_idds import ActiveLearningStrategyIDDS

class ActiveLearningStrategyDUAL(ActiveLearningStrategyIDDS, ActiveLearningStrategyBAS, ActiveLearningStrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquired_samples_neighbor_idxs = []

    def requires_warmup(self) -> bool:
        return True
    
    def requires_embeddings(self) -> bool:
        return True
    
    def acquire_samples(self) -> List[int]:
        bas_num_samples_to_rank = self.active_learning_args["bas_num_samples_to_rank"]
        active_learning_samples_per_iteration = self.active_learning_args["active_learning_samples_per_iteration"]
        
        acquired_samples_idxs = []
        for i in range(active_learning_samples_per_iteration):
            logging.info(f"---Acquiring sample {i+1}---")
            if i < 0.5 * active_learning_samples_per_iteration:
                logging.info(f"Computing IDDS scores and selecting the top {bas_num_samples_to_rank} highest scoring documents")
                assume_already_labelled_idxs = list(set(acquired_samples_idxs + self.acquired_samples_neighbor_idxs))
                idds_scores = self._compute_idds_scores(assume_already_labelled_idxs)
                logging.info(f"Max IDDS score: {idds_scores.max()}")
                logging.info(f"Min IDDS score: {idds_scores.min()}")
                idds_asc_scores_idxs = idds_scores.argsort()
                idds_sample_dataset_idxs = idds_asc_scores_idxs[-bas_num_samples_to_rank:].tolist()
                logging.info(f"Computing BLEUvar scores for the selected documents")
                bleuvars = self.compute_bleuvar_scores(idds_sample_dataset_idxs)
            
                self._log_metrics_for_samples(idds_sample_dataset_idxs, idds_scores[idds_sample_dataset_idxs].tolist(), bleuvars, "ActiveLearningStrategyDUAL")
            
                bleuvars = np.array(bleuvars)
                bas_asc_scores_idxs = bleuvars.argsort()
                bas_asc_scores_idxs = [idx for idx in bas_asc_scores_idxs if bleuvars[idx] < 0.96] # threshold filter
                highest_score_idx = bas_asc_scores_idxs[-1]
                idx = idds_sample_dataset_idxs[highest_score_idx]
                logging.info(f"Selecting the document with the highest BLEUvar score (index: {idx}, score: {bleuvars[highest_score_idx]})")

                self.acquired_samples_neighbor_idxs.extend(idds_sample_dataset_idxs)
                self.acquired_samples_neighbor_idxs.remove(idx)
            else:
                _, idxs = self.dataset_handler.sample_unlabelled(1)
                idx = idxs[0]
                logging.info(f"Selecting random document (index: {idx})")

            acquired_samples_idxs.append(idx)

        return acquired_samples_idxs

    def _log_metrics_for_samples(self, sample_dataset_idxs: List[int], sample_dataset_scores: List[float], bleuvars: List[float], source: str):
        super()._log_metrics_for_samples(sample_dataset_idxs, sample_dataset_scores, source)

        # Log BLEUvar scores
        # We expect to have relatively high variance, because the BLEUVar distribution 
        # does not necessarily match the embeddings variance distribution
        logging.info(f"BLEUvar scores: {bleuvars}")
        json_scores = json.dumps({
            "source": source,
            "bleuvar_scores": bleuvars,
        })
        with open(self.train_args.output_dir + "/bleuvar_scores.json", "a") as outfile:
            outfile.write(json_scores + '\n')
