# fl_data.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class EurDatasetLocal(Dataset):
    """
    Loads Europarl *_data.pkl produced by preprocess_text.py, but from a user-provided root path.
    Expected files:
      <data_root>/europarl/train_data.pkl
      <data_root>/europarl/test_data.pkl
    """
    def __init__(self, data_root: str, split: str = "train"):
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        pkl_path = os.path.join(data_root, "europarl", f"{split}_data.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Could not find {pkl_path}. "
                f"Run preprocess_text.py or place the pkl files under {data_root}/europarl/."
            )
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]  # list[int] token ids

    def __len__(self):
        return len(self.data)

def collate_data(batch):
    # Same behavior as dataset.py: pad to max length in batch
    batch_size = len(batch)
    max_len = max(map(len, batch))
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)
    for i, sent in enumerate(sort_by_len):
        sents[i, :len(sent)] = sent
    return torch.from_numpy(sents)
