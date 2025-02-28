import datasets
import torch.utils.data


class Seq2SeqDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for Hugging Face Datasets.
    This is used to avoid type conversion warnings when using Seq2SeqTrainer.
    """

    def __init__(self, dataset: datasets.Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {key: val for key, val in item.items()}
