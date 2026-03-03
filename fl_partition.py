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


def partition_dirichlet_length(dataset, num_clients: int, alpha: float = 0.5, seed: int = 0):
    """
    Strong non-IID via Dirichlet distribution over sentence length buckets.

    alpha controls heterogeneity:
      alpha=0.1  -> very non-IID (each client dominated by one length range)
      alpha=0.5  -> moderately non-IID (recommended for your paper)
      alpha=5.0  -> nearly IID

    Steps:
      1. Bin sentences into n_buckets length buckets
      2. For each bucket, sample Dirichlet proportions across clients
      3. Assign sentences accordingly
    """
    rng = np.random.default_rng(seed)
    n_buckets = 10  # split length range into 10 bins

    # Get lengths and assign each sentence to a bucket
    lengths = np.array([len(dataset[i]) for i in range(len(dataset))])
    bucket_edges = np.percentile(lengths, np.linspace(0, 100, n_buckets + 1))
    bucket_ids = np.digitize(lengths, bucket_edges[1:-1])  # 0 to n_buckets-1

    client_indices = [[] for _ in range(num_clients)]

    for bucket in range(n_buckets):
        # All sentence indices in this bucket
        bucket_idx = np.where(bucket_ids == bucket)[0]
        if len(bucket_idx) == 0:
            continue
        rng.shuffle(bucket_idx)

        # Sample Dirichlet proportions for this bucket
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to integer counts
        counts = (proportions * len(bucket_idx)).astype(int)
        counts[-1] = len(bucket_idx) - counts[:-1].sum()  # fix rounding

        # Assign to clients
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(
                bucket_idx[start:start + count].tolist()
            )
            start += count

    return client_indices
