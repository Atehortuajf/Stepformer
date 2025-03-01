import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StepformerDataset(Dataset):
    def __init__(self, data_path):
        '''
        Initializes the dataset by loading the data from a JSON or pickle file.

        Args:
            data_path (str): Path to the dataset file. Supported formats: .json, .pkl, .pickle
        '''
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found!")

        _, ext = os.path.splitext(data_path)
        if ext == '.json':
            with open(data_path, 'r') as f:
                self.samples = json.load(f)
        elif ext in ['.pkl', '.pickle']:
            with open(data_path, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Please use JSON or pickle.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Convert lists of tokens to torch tensors
        return {
            "music_tokens": torch.tensor(item["music_tokens"], dtype=torch.long),
            "dance_tokens": torch.tensor(item["dance_tokens"], dtype=torch.long)
        }


def collate_fn(batch):
    '''
    Custom collate function to pad sequences in a batch.
    
    Args:
        batch (list): A list of samples, each a dict with keys 'music_tokens' and 'dance_tokens'.
    
    Returns:
        dict: A dictionary with padded 'music_tokens' and 'dance_tokens' tensors.
    '''
    music_tokens = pad_sequence([sample["music_tokens"] for sample in batch], batch_first=True, padding_value=0)
    dance_tokens = pad_sequence([sample["dance_tokens"] for sample in batch], batch_first=True, padding_value=0)
    return {"music_tokens": music_tokens, "dance_tokens": dance_tokens} 