from typing import Tuple
from datasets import load_dataset, Dataset
from datasets import get_dataset_config_names

def load_dataset_splits_from_hf(dataset_name: str, dataset_train_split: str, dataset_validation_split: str, dataset_test_split: str, dataset_config_name: str, dataset_data_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
    dataset = load_dataset(dataset_name, dataset_config_name, trust_remote_code=True, data_dir=dataset_data_dir)
    if len(dataset) == 1:
        # If the dataset is a single split, we need to split it into train, validation and test
        train_and_test_datasets = dataset[dataset_train_split].train_test_split(test_size=0.1)
        train_dataset = train_and_test_datasets[dataset_train_split]
        validation_and_test_dataset = train_and_test_datasets["test"].train_test_split(test_size=0.5)
        validation_dataset = validation_and_test_dataset['train']
        evaluation_dataset = validation_and_test_dataset['test']
    else:
        train_dataset = dataset[dataset_train_split]
        validation_dataset = dataset[dataset_validation_split]
        evaluation_dataset = dataset[dataset_test_split]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(evaluation_dataset)}")
    return train_dataset, validation_dataset, evaluation_dataset
