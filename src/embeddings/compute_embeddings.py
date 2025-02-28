import torch
import datasets
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from src.utils.seq2seq_dataset import Seq2SeqDataset

def compute_embeddings(
    embeddings_model: SentenceTransformer,
    documents: List[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    dataset = datasets.Dataset.from_dict({"text": documents})
    dataloader = DataLoader(
        Seq2SeqDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
    )
    
    embeddings = torch.empty((len(documents), 768), dtype=torch.float, device=device)
    start = 0
    for batch in tqdm(dataloader):
        batch_embeddings = embeddings_model.encode(
            batch["text"],
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        assert isinstance(batch_embeddings, torch.Tensor)
        end = start + len(batch["text"])
        embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
        start = end

    return embeddings
