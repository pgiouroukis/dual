import logging
import math
import json
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
import random
import datasets
import torch.utils
from typing import List
from abc import ABC, abstractmethod
from transformers import Seq2SeqTrainer, EvalPrediction, EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from src.active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from src.utils.seq2seq_dataset import Seq2SeqDataset
from src.embeddings.load_embeddings_from_url import load_embeddings_from_url
from src.embeddings.compute_embeddings import compute_embeddings
from src.embeddings.run_unsupervised_embeddings_training import run_unsupervised_embeddings_training_tapt, run_unsupervised_embeddings_training_tsdae

class ActiveLearningStrategyBase(ABC):
    def __init__(
            self, 
            dataset_handler: ActiveLearningDatasetHandler, 
            model_name: str,
            train_args: Seq2SeqTrainingArguments,
            dataset_args: dict,
            active_learning_args: dict,
            embeddings_args: dict,
            min_train_steps: int,
            train_validation: bool,
            train_validation_samples: int,
            evaluation_dataset: datasets.Dataset,
            validation_dataset: datasets.Dataset,
        ):
        self.dataset_handler = dataset_handler
        self.model_name = model_name
        self.train_args = train_args
        self.dataset_args = dataset_args
        self.active_learning_args = active_learning_args
        self.embedding_args = embeddings_args
        self.min_train_steps = min_train_steps
        self.train_validation = train_validation
        self.train_validation_samples = train_validation_samples
        self.evaluation_dataset = evaluation_dataset
        self.validation_dataset = validation_dataset
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        eval_samples = self.dataset_args["eval_samples"]
        if eval_samples > 0:
            self.tokenized_evaluation_dataset = self._get_tokenized_dataset(self.evaluation_dataset.select(range(eval_samples)))
        else:
            self.tokenized_evaluation_dataset = self._get_tokenized_dataset(self.evaluation_dataset)
        
        if self.train_validation:
            assert(self.train_validation_samples > 0)
            self.tokenized_validation_dataset = self._get_tokenized_dataset(
                self.validation_dataset.shuffle(seed=self.train_args.seed).select(range(self.train_validation_samples))
            )
        
        self.rouge = evaluate.load("rouge")
        self.experiment_metrics = []

        self.given_num_train_epochs = self.train_args.num_train_epochs

        if not self.requires_embeddings():
            return

        # Handle embeddings-related arguments
        if self.embedding_args["embeddings_remote_url"] is not None:
            # Case 1: Load embeddings from a remote URL
            logging.info("Loading embeddings from the provided remote URL...")
            self.embeddings = load_embeddings_from_url(
                self.embedding_args["embeddings_remote_url"], 
                f"{self.dataset_args["dataset_hf_name"]}.pt",
                str(self.train_args.device)
            )
        elif self.embedding_args["embeddings_pt_tensor_path"] is not None:
            # Case 2: Load embeddings from a local file
            logging.info("Loading embeddings from the provided local file...")
            self.embeddings = torch.load(self.embedding_args["embeddings_pt_tensor_path"], map_location=self.train_args.device)
        else:
            # Case 3: Compute embeddings using a SentenceTransformer model
            if self.embedding_args["embeddings_run_unsupervised_training"]:
                # Case 3.1: Run unsupervised training on the embeddings model to adapt it to the dataset
                logging.info("Running unsupervised training on the embeddings model...")
                if self.embedding_args["embeddings_train_approach"] == "tapt":
                    embeddings_train_model = run_unsupervised_embeddings_training_tapt(
                        self.dataset_handler.dataset[self.dataset_args["dataset_document_column"]],
                        str(self.train_args.device),
                        self.embedding_args["embeddings_train_batch_size"],
                        self.embedding_args["embeddings_train_num_epochs"],
                        max_seq_length = self.dataset_args["max_source_length"],
                    )
                else:
                    embeddings_train_model = run_unsupervised_embeddings_training_tsdae(
                        self.dataset_handler.dataset[self.dataset_args["dataset_document_column"]],
                        str(self.train_args.device),
                        self.embedding_args["embeddings_train_batch_size"],
                        self.embedding_args["embeddings_train_num_epochs"],
                    )
            else:
                # Case 3.2: Use an embeddings model without unsupervised training
                logging.info("Using the embeddings model without unsupervised training...")
                embeddings_train_model = SentenceTransformer(self.embedding_args["embeddings_model"])
            
            logging.info("Computing embeddings for the dataset...")
            self.embeddings = compute_embeddings(
                embeddings_train_model,
                self.dataset_handler.dataset[self.dataset_args["dataset_document_column"]],
                self.embedding_args["embeddings_gen_batch_size"],
                str(self.train_args.device),
            )
            torch.save(self.embeddings, f"{self.dataset_args["dataset_hf_name"]}.pt")
            
            del embeddings_train_model


    def _tokenize(self, samples: datasets.Dataset) -> dict:
        documents = samples[self.dataset_args["dataset_document_column"]]
        if "t5" in self.model_name:
            logging.info("Using Flan model, adding 'summarize: ' prefix before tokenizing the documents")
            documents = [
                f"summarize: {doc}" for doc in documents
            ]

        tokenized_documents = self.tokenizer(
            documents,
            padding="max_length",
            truncation=True,
            max_length=self.dataset_args["max_source_length"],
            return_tensors="pt",
        ).to(self.train_args.device)

        tokenized_summaries = self.tokenizer(
            samples[self.dataset_args["dataset_summary_column"]],
            padding="max_length",
            truncation=True,
            max_length=self.dataset_args["max_generation_length"],
            return_tensors="pt",
        ).to(self.train_args.device)

        # Since we added padding, we set the padding tokens
        # to -100 so it is ignored by PyTorch's loss function
        tokenized_summaries_input_ids = tokenized_summaries["input_ids"]
        assert isinstance(tokenized_summaries_input_ids, torch.Tensor)
        tokenized_summaries["input_ids"] = torch.tensor([
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in tokenized_summaries_input_ids
        ], dtype=torch.int64).to(self.train_args.device)    

        tokenized_samples = {
            # Read here: https://huggingface.co/docs/transformers/model_doc/pegasus#transformers.PegasusModel.forward
            "input_ids": tokenized_documents["input_ids"],  # This key MUST be set to 'input_ids'
            "attention_mask": tokenized_documents["attention_mask"],  # This key MUST be set to 'attention_mask'
            # Read here: https://huggingface.co/docs/transformers/glossary#decoder-input-ids
            "labels": tokenized_summaries["input_ids"],     # This key MUST be set to 'labels'
        }

        return tokenized_samples        

    def _get_tokenized_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        tokenized_dataset = dataset.map(
            self._tokenize,
            batched=True,
            remove_columns=[self.dataset_args["dataset_document_column"], self.dataset_args["dataset_summary_column"]]
        )
        
        # map() returns the data in list instead of tensor, thus we need to set the format to torch
        # See more here: https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
        # and here: https://github.com/huggingface/datasets/issues/625#issuecomment-696127826
        tokenized_dataset.set_format("torch") # device will be cpu
        
        return tokenized_dataset

    def _calculate_num_train_epochs(self) -> int:
        if self.min_train_steps is not None:
            effective_batch_size = self.train_args.train_batch_size * self.train_args.gradient_accumulation_steps
            steps_per_epoch = math.ceil(self.dataset_handler.get_labelled_count() / effective_batch_size)
            min_epochs = math.ceil(self.min_train_steps / steps_per_epoch)
            return max(min_epochs, int(self.given_num_train_epochs))
        return int(self.given_num_train_epochs)


    def _compute_metrics(self, eval_preds: EvalPrediction) -> dict:
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, np.array(self.tokenizer.pad_token_id))
        preds = np.where(preds != -100, preds, np.array(self.tokenizer.pad_token_id))

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        assert(result is not None)

        return result

    @abstractmethod
    def requires_warmup(self) -> bool:
        pass

    @abstractmethod
    def requires_embeddings(self) -> bool:
        pass
        
    # This method is invoked at the beginning of each active learning iteration
    # It should return a list of indices of the samples that should be labelled
    @abstractmethod
    def acquire_samples(self) -> List[int]:
        pass

    def train_and_evaluate(self):
        self.train_args.num_train_epochs = self._calculate_num_train_epochs()
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.train_args.device)
        tokenized_labelled_dataset = self._get_tokenized_dataset(self.dataset_handler.get_labelled_dataset())
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset = Seq2SeqDataset(tokenized_labelled_dataset),
            eval_dataset = Seq2SeqDataset(self.tokenized_validation_dataset) if self.train_validation else None,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
                    if self.train_validation else None
        )
        trainer.train()

        metrics = trainer.evaluate(
            eval_dataset = Seq2SeqDataset(self.tokenized_evaluation_dataset),
            max_length = self.dataset_args["max_generation_length"],
            num_beams = self.train_args.generation_num_beams,
            metric_key_prefix = 'eval'
        )
        self.experiment_metrics.append(metrics)
        json_metrics = json.dumps(metrics)
        with open(self.train_args.output_dir + "/eval_metrics.json", "a") as outfile:
            outfile.write(json_metrics + '\n')

        self.log_sample_generations(5)

        gc.collect()
        torch.cuda.empty_cache()
        del trainer, tokenized_labelled_dataset

    def run(self):
        logging.info(f"\n\n\nRUNNING ACTIVE LEARNING STRATEGY {self.active_learning_args['active_learning_strategy']}\n\n")

        warmup_active_learning_iterations = 0
        if self.requires_warmup():
            logging.info("Strategy requires warmup, performing warmup...")
            num_warmup_samples = self.active_learning_args["active_learning_warmup_samples"]
            _, warmup_dataset_idxs = self.dataset_handler.sample_unlabelled(num_warmup_samples)
            self.dataset_handler.label_samples(warmup_dataset_idxs)
            self.dataset_handler.get_labelled_dataset().to_json(self.train_args.output_dir + "/train_data.json")
            warmup_active_learning_iterations = self.active_learning_args["active_learning_warmup_samples"] // self.active_learning_args["active_learning_samples_per_iteration"]
            
            # If warmup_active_learning_iterations is bigger than one, some AL steps are skipped and thus
            # there will be no evaluation metrics for them. We log a dummy metric for each skipped step.
            for i in range(warmup_active_learning_iterations - 1):
                with open(self.train_args.output_dir + "/eval_metrics.json", "a") as outfile:
                    outfile.write(json.dumps({"eval": 0.0, "reason": "AL iteration skipped due to warmup"}) + '\n')
                
            self.train_and_evaluate()

        
        for i in range(self.active_learning_args["active_learning_iterations"] - warmup_active_learning_iterations):
            logging.info(f"\n\n\nSTARTING ACTIVE LEARNING ITERATION {i+1}\n\n")
            samples_idxs = self.acquire_samples()
            self.dataset_handler.label_samples(samples_idxs)
            self.dataset_handler.get_labelled_dataset().to_json(self.train_args.output_dir + "/train_data.json")
            self.train_and_evaluate()
        
        del self.model, self.tokenizer
    
    def log_sample_generations(self, num_generations: int) -> None:
        random_indexes = random.sample(range(len(self.tokenized_evaluation_dataset)), num_generations)
        dataset = self.tokenized_evaluation_dataset.select(random_indexes)
        dataset.set_format("torch")
        input_ids = dataset["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        gens = self.model.generate(
            input_ids.to(self.train_args.device),
            max_length=self.dataset_args["max_generation_length"], 
            num_beams=self.train_args.generation_num_beams, 
            early_stopping=True
        )
        decoded_gens = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
        decoded_golden_summaries = self.evaluation_dataset.select(random_indexes)[self.dataset_args["dataset_summary_column"]]
        decoded_documents = self.evaluation_dataset.select(random_indexes)[self.dataset_args["dataset_document_column"]]
        
        logging.info("Logging sample generations: ")
        for i in range(num_generations):
            logging.info(f"Original document: {decoded_documents[i]}")
            logging.info(f"Golden summary: {decoded_golden_summaries[i]})")
            logging.info(f"Generated summary: {decoded_gens[i]}")
            logging.info("")
