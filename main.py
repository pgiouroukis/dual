import argparse
import torch
import logging
import nltk
from tqdm.contrib.logging import logging_redirect_tqdm
from src.active_learning.active_learning_strategy_picker import get_active_learning_class
from src.active_learning.active_learning_strategy_random import ActiveLearningStrategyRandom
from src.utils.load_data import load_dataset_splits_from_hf
from src.utils.set_seed import set_seed
from src.active_learning.active_learning_dataset_handler import ActiveLearningDatasetHandler
from src.utils.args_groups import get_train_args, get_dataset_args, get_active_learning_args, get_embeddings_args
from src.utils.logging_utils import setup_logging_to_file

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Path to save the experiment results")
    parser.add_argument("--model_name", type=str, help="Name of the model in Hugging Face")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42])

    # Active Learning arguments
    parser.add_argument("--active_learning_strategy", type=str)
    parser.add_argument("--active_learning_iterations", type=int)
    parser.add_argument("--active_learning_samples_per_iteration", type=int)
    parser.add_argument("--active_learning_warmup_samples", type=int, help="Number of samples to start the active learning process. Will be ignored for strategies that don't require warmup.")

    # Training and Evaluation arguments
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--min_train_steps", type=int, default=300)
    parser.add_argument("--train_validation", type=int, default=0, help="If 1, the model will be trained with a validation set, using early stopping")
    parser.add_argument("--train_validation_samples", type=int, default=0, help="Only used if train_validation is True")
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--weight_decay", type=float, default=0.028)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_beams", type=int, default=3)

    # Dataset arguments
    parser.add_argument("--dataset_hf_name", type=str)
    parser.add_argument("--dataset_hf_config", type=str, default="default", help="Configuration name for the HF dataset, more info here: https://huggingface.co/docs/datasets/load_hub#configurations")
    parser.add_argument("--dataset_data_dir", type=str, default=None)
    parser.add_argument("--dataset_train_split", type=str)
    parser.add_argument("--dataset_validation_split", type=str)
    parser.add_argument("--dataset_test_split", type=str)
    parser.add_argument("--dataset_document_column", type=str)
    parser.add_argument("--dataset_summary_column", type=str)
    parser.add_argument("--eval_samples", type=int, help="Number of samples to evaluate the model. If None or <=0, the whole dataset will be used.")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_generation_length", type=int, default=64)

    # Embeddings arguments
    # Case 1: Load embeddings from a remote URL
    parser.add_argument("--embeddings_remote_url", type=str)

    # Case 2: Load embeddings from a local pt tensor file
    parser.add_argument("--embeddings_pt_tensor_path", type=str)
    
    # Case 3: Compute embeddings using a SentenceTransformer model
    # Case 3.1: Run unsupervised training on the embeddings model to adapt it to the dataset
    parser.add_argument("--embeddings_run_unsupervised_training", type=bool)
    parser.add_argument("--embeddings_train_batch_size", type=int, default=16)
    parser.add_argument("--embeddings_train_num_epochs", type=int, default=1)
    parser.add_argument("--embeddings_train_approach", type=str, choices=["tapt", "tsdae"], default="tapt")
    # Case 3.2: Use the embeddings model without unsupervised training
    parser.add_argument("--embeddings_model", type=str)
    parser.add_argument("--embeddings_gen_batch_size", type=int, default=64)
    
    # Strategy-specific arguments
    parser.add_argument("--bas_num_samples_to_rank", type=int)
    parser.add_argument("--bas_num_samples_mc_dropout", type=int)

    args = parser.parse_args()
    
    args.train_validation = bool(args.train_validation)
    args.embeddings_run_unsupervised_training = bool(args.embeddings_run_unsupervised_training)

    return args

if __name__ == "__main__":
    args = parse_args()
    nltk.download('punkt')
    # args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    root_path = args.output_path
    for seed in args.seeds:
        set_seed(seed)
        args.seed = seed
        args.output_path = f"{root_path}/seed_{seed}"

        setup_logging_to_file(args.output_path)
        logging.info(f"Running experiment with seed {seed}")

        train_args = get_train_args(args)
        dataset_args = get_dataset_args(args)
        active_learning_args = get_active_learning_args(args)
        embedding_args = get_embeddings_args(args)
        
        train_dataset, validation_dataset, evaluation_dataset = load_dataset_splits_from_hf(args.dataset_hf_name, args.dataset_train_split, args.dataset_validation_split, args.dataset_test_split, args.dataset_hf_config, args.dataset_data_dir)
        active_learning_dataset_handler = ActiveLearningDatasetHandler(train_dataset, dataset_args)

        ActiveLearningClass = get_active_learning_class(args.active_learning_strategy)
        active_learning = ActiveLearningClass(
            active_learning_dataset_handler,
            args.model_name,
            train_args,
            dataset_args,
            active_learning_args,
            embedding_args,
            args.min_train_steps,
            args.train_validation,
            args.train_validation_samples,
            evaluation_dataset,
            validation_dataset
        )
        active_learning.run()

        del active_learning
        torch.cuda.empty_cache()
