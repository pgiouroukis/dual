import logging
import transformers
import json
import numpy as np
from typing import List
from src.active_learning.active_learning_strategy_base import ActiveLearningStrategyBase
from transformers.models.pegasus.modeling_pegasus import PegasusEncoderLayer, PegasusDecoderLayer
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoderLayer
from transformers.models.t5.modeling_t5 import T5Block
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from src.utils.seq2seq_dataset import Seq2SeqDataset
from nltk.tokenize import sent_tokenize, word_tokenize


class ActiveLearningStrategyBAS(ActiveLearningStrategyBase):    
    def requires_warmup(self) -> bool:
        return True

    def requires_embeddings(self) -> bool:
        return False

    def acquire_samples(self) -> List[int]:
        bas_num_samples_to_rank = self.active_learning_args["bas_num_samples_to_rank"]
        active_learning_samples_per_iteration = self.active_learning_args["active_learning_samples_per_iteration"]

        sample_dataset, reduced_sample_dataset_idxs = self.dataset_handler.sample_unlabelled(bas_num_samples_to_rank)

        scores = self.compute_bleuvar_scores(reduced_sample_dataset_idxs)
        scores = np.array(scores)
        asc_scores_idxs = scores.argsort()

        json_scores = json.dumps(scores[asc_scores_idxs].tolist())
        with open(self.train_args.output_dir + "/bleuvar_scores.json", "a") as outfile:
            outfile.write(json_scores + '\n')
        
        # Remove samples with very high BLEUvar
        asc_scores_idxs = [idx for idx in asc_scores_idxs if scores[idx] <= 0.96] 
        assert(len(asc_scores_idxs) >= active_learning_samples_per_iteration)
        
        # Keep 30% of the lowest uncertainty samples and 70% of the highest uncertainty samples
        # asc_scores_idxs = asc_scores_idxs[: int(0.3 * active_learning_samples_per_iteration)] + asc_scores_idxs[int(-0.7 * active_learning_samples_per_iteration):]
        asc_scores_idxs = asc_scores_idxs[-10:]
        assert(len(asc_scores_idxs) == active_learning_samples_per_iteration)
        logging.info(f"Selected samples with BLEUvar: {scores[asc_scores_idxs]}")

        reduced_sample_dataset_idxs = [reduced_sample_dataset_idxs[idx] for idx in asc_scores_idxs]

        summary_column = self.dataset_args["dataset_summary_column"]
        al_dataset = self.dataset_handler.dataset
        assert(al_dataset[reduced_sample_dataset_idxs[0]][summary_column] == sample_dataset[asc_scores_idxs[0].item()][summary_column])
        assert(al_dataset[reduced_sample_dataset_idxs[active_learning_samples_per_iteration-1]][summary_column] == sample_dataset[asc_scores_idxs[active_learning_samples_per_iteration-1].item()][summary_column])

        return reduced_sample_dataset_idxs
        
    def compute_bleuvar_scores(self, sample_dataset_idxs: List[int]) -> List[float]:
        sample_dataset = self.dataset_handler.dataset.select(sample_dataset_idxs)
        assert(sample_dataset[0][self.dataset_args["dataset_summary_column"]] == self.dataset_handler.dataset[sample_dataset_idxs[0]][self.dataset_args["dataset_summary_column"]])
        tokenized_sample_dataset = self._get_tokenized_dataset(sample_dataset)
        
        # If you remove this line, all generated summaries will be mostly the same
        self.convert_model_to_bayesian()

        dataloader = DataLoader(
            Seq2SeqDataset(tokenized_sample_dataset),
            batch_size=self.train_args.per_device_train_batch_size,
            shuffle=False,
        )

        # Generate MC summaries
        generated_sums = []
        for i, batch in enumerate(tqdm(dataloader)):
            generations = self._run_mc_dropout(batch)
            generated_sums += generations

        # Calculate bleuvar
        bleuvars = []
        for generation_list in generated_sums:
            bleuvar = self._analyze_generation_bleuvar(generation_list)
            bleuvars.append(bleuvar)

        return bleuvars

    def convert_model_to_bayesian(self) -> None:
        self.model.eval()
        self.model.apply(self._apply_dropout)

    def _run_mc_dropout(self, batch: dict) -> List[List[str]]:
        input_ids = batch['input_ids'].to(self.train_args.device)
        generations = []
        transformers.logging.set_verbosity_error() # temporarily disable INFO logging, otherwise the generation config is logged at each generation
        for i_s in range(self.active_learning_args["bas_num_samples_mc_dropout"]):
            model_outputs = self.model.generate(
                input_ids,
                num_beams=self.train_args.generation_num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_sum = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ) for g in model_outputs["sequences"]
            ]
            generations.append(gen_sum)
        transformers.utils.logging.set_verbosity_info() # re-enable INFO logging
        generations = [list(x) for x in zip(*generations)]
        return generations

    def _analyze_generation_bleuvar(self, gen_list: List[str]) -> float:
        """
        Given a list of generated texts, computes 
        the pairwise BLEUvar between all text pairs. 
        """
        n = self.active_learning_args["bas_num_samples_mc_dropout"]
        bleu_var = 0.
        for j, dec_j in enumerate(gen_list):
            for k in range(j + 1, n):
                dec_k = gen_list[k]
                jk_bleu = self._pair_bleu(dec_j, dec_k)
                kj_bleu = self._pair_bleu(dec_k, dec_j)
                bleu_var += (1 - jk_bleu) ** 2
                bleu_var += (1 - kj_bleu) ** 2
        bleu_var /= n * (n - 1)

        return bleu_var

    def _apply_dropout(self, module: nn.Module) -> None: 
        """
        Changes all Encoder and Decoder layers to training mode.
        This will essentially turn dropout and layer normalization
        on for MC dropout prediction.
        """
        if type(module) in [PegasusEncoderLayer, PegasusDecoderLayer, BartEncoderLayer, BartDecoderLayer, T5Block]:
            module.train() 

    def _pair_bleu(self, text1: str, text2: str) -> float:
        """
        Compute the bleu score between two given texts.
        A smoothing function is used to avoid zero scores when
        there are no common higher order n-grams between the
        texts.
        """
        tok1 = [word_tokenize(s) for s in sent_tokenize(text1)]
        tok2 = [word_tokenize(s) for s in sent_tokenize(text2)]
        score = 0
        for c_cent in tok2:
            try:
                s = corpus_bleu([tok1], [c_cent], smoothing_function=SmoothingFunction().method1)
                assert isinstance(s, (int, float))
                score += s
            except KeyError:
                score = 0.
        try:
            score /= len(tok2)
        except ZeroDivisionError:
            score = 0.

        return score
