# fl_partition.py
import numpy as np

def partition_iid(num_samples: int, num_clients: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [s.tolist() for s in splits]

def partition_by_length_mild(dataset, num_clients: int, seed: int = 0):
    """
    Mild non-IID: sort by sentence length then interleave so each client gets a biased (but not extreme) range.
    This is a common realistic skew for text when you don't have labels/topics.

    dataset[i] -> list[int] token ids
    """
    lengths = np.array([len(dataset[i]) for i in range(len(dataset))])
    idx = np.argsort(lengths)  # short -> long

    # Interleave indices across clients to keep skew mild
    client_lists = [[] for _ in range(num_clients)]
    for j, i in enumerate(idx):
        client_lists[j % num_clients].append(int(i))
    return client_lists
