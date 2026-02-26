'''
local vocab and datasets --> partition training data across clients
--> initialize global DeepSC model
--> federated training loop (client selection, local training, aggregation)
so for each federated round:
1) select subset of clients
2) each selected client trains local DeepSC model starting from global weights with channel noise sampled once per local epoch
3) server aggregates updated client models using FedAvg (size-weighted) or FedLol (loss-weighted)
'''

import os
import json
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from models.transceiver import DeepSC
from utils import initNetParams, train_step, SNR_to_noise
from fl_data import EurDatasetLocal, collate_data
from fl_partition import partition_iid, partition_by_length_mild
from fl_eval import evaluate_bleu

# use GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed) #deterministic CPU RNG for torch 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) #deterministic numpy RNG
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True


def fedavg(global_model, client_states, client_sizes):
    """
    FedAvg: weighted average = sum of client models weighted by their dataset sizes.
    """
    new_state = copy.deepcopy(global_model.state_dict())
    total = float(sum(client_sizes))
    for k in new_state.keys():
        new_state[k] = sum(
            client_states[i][k] * (client_sizes[i] / total)
            for i in range(len(client_states))
        )
    global_model.load_state_dict(new_state)
    return global_model


def fedlol(global_model, client_states, client_losses, eps=1e-8):
    """
    FedLol: aggregate client models weighted by inverse loss (lower loss -> higher weight).
    weight_k = (1 / (loss_k + eps)) / sum_j(1 / (loss_j + eps))
    """
    inv_losses = np.array([1.0 / (L + eps) for L in client_losses], dtype=np.float64)
    weights = inv_losses / inv_losses.sum()
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        new_state[k] = sum(
            client_states[i][k] * weights[i]
            for i in range(len(client_states))
        )
    global_model.load_state_dict(new_state)
    return global_model


def client_update(global_model, client_loader, args, pad_idx, criterion):
    """
    Train a local DeepSC model starting from global weights.
    Returns (state_dict, mean_loss) for use with FedAvg or FedLol aggregation.
    """

    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=5e-4
    )

    num_batches = len(client_loader)
    use_fedlol = getattr(args, 'algorithm', 'fedlol') == 'fedlol'
    print(f"  [Client] Local training started | batches={num_batches}" + (" (FedLol)" if use_fedlol else ""), flush=True)

    batch_losses = []

    for local_ep in range(args.local_epochs):

        n_var = np.random.uniform(
            SNR_to_noise(args.snr_train_low),
            SNR_to_noise(args.snr_train_high),
            size=(1,)
        )[0]

        print(f"    Local epoch {local_ep+1}/{args.local_epochs} | n_var={n_var:.4e}", flush=True)

        for batch_idx, sents in enumerate(client_loader):
            sents = sents.to(device)

            loss = train_step(
                model=model,
                src=sents,
                trg=sents,
                n_var=n_var,
                pad=pad_idx,
                opt=optimizer,
                criterion=criterion,
                channel=args.channel
            )
            batch_losses.append(loss)

            if batch_idx % args.log_interval == 0:
                print(
                    f"      batch {batch_idx+1}/{num_batches} | loss={loss:.4f}",
                    flush=True
                )

    mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    print("  [Client] Local training finished", flush=True)

    return model.state_dict(), mean_loss


def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--log_interval", type=int, default=200,
                    help="Print training log every N batches {client-side}") #client-side logging

    # Data & vocab
    parser.add_argument("--data_root", type=str, required=True, help="Folder containing europarl/train_data.pkl etc.") #root path to Europarl data
    parser.add_argument("--vocab_file", type=str, default="europarl/vocab.json", help="Path relative to data_root or absolute") #vocab file path


    # DeepSC architecture (match original defaults unless you changed them)
    # model design parameters
    parser.add_argument("--num_layers", type=int, default=4) 
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=30)

    # Channel modeling
    parser.add_argument("--channel", type=str, default="Rayleigh", choices=["AWGN", "Rayleigh", "Rician"])
    parser.add_argument("--snr_train_low", type=float, default=0.0,
                        help="Low end of training SNR range (dB). Wide range (e.g. 0-20) so BLEU increases with SNR at eval.")
    parser.add_argument("--snr_train_high", type=float, default=20.0,
                        help="High end of training SNR range (dB).")
    parser.add_argument("--snr_eval", type=float, default=6.0)

    # Federated Learning settings
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--algorithm", type=str, default="fedlol", choices=["fedavg", "fedlol"],
                        help="Aggregation: fedavg (size-weighted) or fedlol (loss-weighted).")

    # Partitioning
    parser.add_argument("--partition", type=str, default="iid", choices=["iid", "length_mild"])
    parser.add_argument("--seed", type=int, default=0)

    # Saving
    parser.add_argument("--save_dir", type=str, default="checkpoints_fed")
    parser.add_argument("--save_every", type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)

    # ==================================================
    # Load Vocabulary
    # ==================================================
    vocab_path = args.vocab_file
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join(args.data_root, vocab_path)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Could not find vocab file: {vocab_path}")

    vocab = json.load(open(vocab_path, "rb"))
    token_to_idx = vocab["token_to_idx"]
    idx_to_token = vocab["idx_to_token"]

    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"] #for masking padding tokens 
    start_idx = token_to_idx["<START>"] #for start of sentence
    end_idx = token_to_idx["<END>"] #for end of sentence

    print(f"[Setup] Vocabulary size = {num_vocab}")

    # ==================================================
    # Load Datasets
    # ==================================================
    train_set = EurDatasetLocal(args.data_root, split="train")
    test_set = EurDatasetLocal(args.data_root, split="test")

    print(f"[Setup] Train samples = {len(train_set)}")
    print(f"[Setup] Test  samples = {len(test_set)}")

    # ==================================================
    # Partition Data Across Clients
    # ==================================================
    if args.partition == "iid":
        client_indices = partition_iid(
            len(train_set), args.num_clients, seed=args.seed
        )
    else:
        client_indices = partition_by_length_mild(
            train_set, args.num_clients, seed=args.seed
        )

    # Create DataLoaders for each client
    # each client gets its own DataLoader with its subset of data, data is shuffled for randomness
    client_loaders = []
    for cid in range(args.num_clients):
        subset = Subset(train_set, client_indices[cid])
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_data
        )
        client_loaders.append(loader)

    print(f"[Setup] {args.num_clients} federated clients initialized")

    # ==================================================
    # Initialize Global DeepSC Model
    # ==================================================
    global_model = DeepSC(
        args.num_layers,
        num_vocab, num_vocab, num_vocab, num_vocab,
        args.d_model,
        args.num_heads,
        args.dff,
        args.dropout
    ).to(device)

    initNetParams(global_model)
    criterion = nn.CrossEntropyLoss(reduction="none")

    os.makedirs(args.save_dir, exist_ok=True)

    # ==================================================
    # Federated Training Loop
    # ==================================================
    for r in range(1, args.rounds + 1):

        print("\n" + "=" * 70)
        print(f"[Server] Federated Round {r}/{args.rounds}")
        print("=" * 70)

        selected = np.random.choice(
            args.num_clients,
            size=min(args.clients_per_round, args.num_clients),
            replace=False
        )
        print(f"[Server] Selected clients: {selected.tolist()}")

        client_states = []
        client_sizes = []
        client_losses = []

        for cid in selected:
            print(f"[Server] -> Training client {cid}")

            st, mean_loss = client_update(
                global_model=global_model,
                client_loader=client_loaders[cid],
                args=args,
                pad_idx=pad_idx,
                criterion=criterion
            )

            client_states.append(st)
            client_sizes.append(len(client_loaders[cid].dataset))
            client_losses.append(mean_loss)

        alg = getattr(args, 'algorithm', 'fedlol')
        if alg == 'fedlol':
            print("[Server] Aggregating client models (FedLol, loss-weighted)")
            global_model = fedlol(global_model, client_states, client_losses)
        else:
            print("[Server] Aggregating client models (FedAvg)")
            global_model = fedavg(global_model, client_states, client_sizes)

        if r % args.save_every == 0:
            ckpt = os.path.join(
                args.save_dir,
                f"fed_deepsc_{args.channel}_round{r:03d}.pth"
            )
            torch.save(global_model.state_dict(), ckpt)
            print(f"[Server] Checkpoint saved: {ckpt}")

    # ==================================================
    # Final Model Save
    # ==================================================
    final_ckpt = os.path.join(
        args.save_dir,
        f"fed_deepsc_{args.channel}_final.pth"
    )
    torch.save(global_model.state_dict(), final_ckpt)
    print(f"\n[Done] Final model saved to: {final_ckpt}", flush=True)

if __name__ == "__main__":
    main()
