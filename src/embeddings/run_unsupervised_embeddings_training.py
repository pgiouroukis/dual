from typing import List
from torch.utils.data import DataLoader
from sentence_transformers import (
    models,
    datasets,
    losses,
    SentenceTransformer,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset

def run_unsupervised_embeddings_training_tapt(
    documents: List[str],
    device: str,
    batch_size:int,
    train_epochs: int = 1,
    embeddings_model_name: str = "bert-base-uncased",
    max_seq_length: int = 512,
) -> SentenceTransformer:
    """
    Perform unsupervised training on a pre-trained model using TAPT:
    https://arxiv.org/pdf/2109.06466

    Returns:
        SentenceTransformer: The trained embeddings model.
    """


    model = AutoModelForMaskedLM.from_pretrained(embeddings_model_name)
    tokenizer = AutoTokenizer.from_pretrained(embeddings_model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)
    
    dataset = Dataset.from_dict({"text": documents})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./embeddings-results",
        logging_dir="./embeddings-logs",
        overwrite_output_dir=True,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=0.0001,
        optim="adamw_torch",
        weight_decay=0.01,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        save_strategy="no",
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    model.save_pretrained("./tapt_model")
    tokenizer.save_pretrained("./tapt_model")

    word_embedding_model = models.Transformer("./tapt_model")
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")

    return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

def run_unsupervised_embeddings_training_tsdae(
    documents: List[str],
    device: str,
    batch_size:int,
    train_epochs: int = 1,
    embeddings_model_name: str = "bert-base-uncased",
) -> SentenceTransformer:
    """
    Trains an embeddings model using the following approach:
    https://sbert.net/examples/unsupervised_learning/TSDAE/README.html

    Returns:
        SentenceTransformer: The trained embeddings model.
    """
    embeddings_model = models.Transformer(embeddings_model_name)
    pooling_model = models.Pooling(
        embeddings_model.get_word_embedding_dimension(), "cls"
    )
    embeddings_train_model = SentenceTransformer(
        modules=[embeddings_model, pooling_model], device=device
    )

    embeddings_train_dataset = datasets.DenoisingAutoEncoderDataset(documents)
    train_dataloader = DataLoader(embeddings_train_dataset, batch_size=batch_size, shuffle=False)
    train_loss = losses.DenoisingAutoEncoderLoss(
        embeddings_train_model,
        decoder_name_or_path=embeddings_model_name,
        tie_encoder_decoder=True,
    )
    embeddings_train_model.to(device)
    embeddings_train_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=train_epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
    )

    return embeddings_train_model
