import os
import torch
import requests

def load_embeddings_from_url(url: str, file_name: str, device: str) -> torch.Tensor:
    """
    Load embeddings from a URL (e.g. an S3 bucket).
    The file is saved in "./_embeddings_/{file_name}" 
    and then loaded into a torch.Tensor in the specified device.
    """
    response = requests.get(url)
    embeddings_dir = "./_embeddings_"
    os.makedirs(embeddings_dir, exist_ok=True)
    with open(f"./_embeddings_/{file_name}", 'wb') as file:
        file.write(response.content)
    embeddings = torch.load(f"./_embeddings_/{file_name}", map_location=device)
    # embeddings = embeddings.to(device)

    return embeddings
