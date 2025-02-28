import argparse
from typing import Tuple
from transformers import Seq2SeqTrainingArguments

def get_train_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir                  = args.output_path,
        overwrite_output_dir        = True,
        seed                        = args.seed,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size  = args.eval_batch_size,
        learning_rate               = args.learning_rate,
        num_train_epochs            = args.num_train_epochs,
        predict_with_generate       = True,
        generation_num_beams        = args.num_beams,
        generation_max_length       = args.max_generation_length,
        eval_strategy               = "epoch" if args.train_validation else "no",
        save_strategy               = "epoch" if args.train_validation else "no",
        optim                       = args.optim,
        weight_decay                = args.weight_decay if args.weight_decay else 0.0,
        warmup_ratio                = args.warmup_ratio if args.warmup_ratio else 0.0,
        gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps else 1,
        report_to                   = "none",
        metric_for_best_model       = "rouge1",
        greater_is_better           = True,
        load_best_model_at_end      = True,
        save_total_limit            = 1,
    )

def get_dataset_args(args: argparse.Namespace) -> dict:
    return {
        "dataset_hf_name": args.dataset_hf_name,
        "dataset_hf_config": args.dataset_hf_config,
        "dataset_document_column": args.dataset_document_column,
        "dataset_summary_column": args.dataset_summary_column,
        "max_source_length": args.max_source_length,
        "max_generation_length": args.max_generation_length,
        "eval_samples": args.eval_samples,
    }

def get_active_learning_args(args: argparse.Namespace) -> dict:
    return {
        "active_learning_strategy": args.active_learning_strategy,
        "active_learning_iterations": args.active_learning_iterations,
        "active_learning_samples_per_iteration": args.active_learning_samples_per_iteration,
        "active_learning_warmup_samples": args.active_learning_warmup_samples,
        "bas_num_samples_to_rank": args.bas_num_samples_to_rank,
        "bas_num_samples_mc_dropout": args.bas_num_samples_mc_dropout,
    }

def get_embeddings_args(args: argparse.Namespace) -> dict:
    return {
        "embeddings_remote_url": args.embeddings_remote_url,
        "embeddings_pt_tensor_path": args.embeddings_pt_tensor_path,
        "embeddings_run_unsupervised_training": args.embeddings_run_unsupervised_training,
        "embeddings_train_batch_size": args.embeddings_train_batch_size,
        "embeddings_train_num_epochs": args.embeddings_train_num_epochs,
        "embeddings_model": args.embeddings_model,
        "embeddings_gen_batch_size": args.embeddings_gen_batch_size,
        "embeddings_train_approach": args.embeddings_train_approach,
    }
