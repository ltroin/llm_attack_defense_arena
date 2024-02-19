import torch
import os
import numpy as np
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from tqdm import tqdm

class PromptDataset(Dataset):
    def __init__(self, prompts, input_ids, mask, labels):
        self.prompts = prompts
        self.input_ids = input_ids
        self.mask = mask
        self.labels = labels
        assert len(self.prompts) == len(self.labels)
        assert len(self.prompts) == len(self.input_ids)
        assert len(self.prompts) == len(self.mask)


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.prompts[idx], self.input_ids[idx], self.mask[idx], self.labels[idx]