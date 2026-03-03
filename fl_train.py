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
from fl_partition import partition_iid, partition_by_length_mild, partition_dirichlet_length
from fl_eval import evaluate_bleu
from gan_modules import (
    SemanticGenerator,
    SemanticDiscriminator,
    local_train_gan,
    federated_aggregate_generator,
)

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
                channel=args.channel,
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


def client_update_with_gan(
    global_model,
    global_generator,
    global_prototypes,
    client_loader,
    args,
    pad_idx,
    criterion,
):
    """
    Client update that:
      1) runs standard local DeepSC training (same as client_update)
      2) runs an additional two-timescale GAN training in semantic space
         using the DeepSC encoder as feature extractor.

    Returns:
        - updated DeepSC state_dict
        - mean token-level loss
        - updated generator state_dict (for GAN aggregation)
        - client_snr used for GAN conditioning
        - local prototype bank tensor
    """

    model = copy.deepcopy(global_model).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=5e-4,
    )

    num_batches = len(client_loader)
    use_fedlol = getattr(args, "algorithm", "fedlol") == "fedlol"
    print(
        f"  [Client] Local training with GAN started | batches={num_batches}"
        + (" (FedLol)" if use_fedlol else ""),
        flush=True,
    )

    batch_losses = []

    for local_ep in range(args.local_epochs):
        n_var = np.random.uniform(
            SNR_to_noise(args.snr_train_low),
            SNR_to_noise(args.snr_train_high),
            size=(1,),
        )[0]

        print(
            f"    Local epoch {local_ep+1}/{args.local_epochs} | n_var={n_var:.4e}",
            flush=True,
        )

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
                channel=args.channel,
                generator=global_generator,
                global_prototypes=global_prototypes,
            )
            batch_losses.append(loss)

            if batch_idx % args.log_interval == 0:
                print(
                    f"      batch {batch_idx+1}/{num_batches} | loss={loss:.4f}",
                    flush=True,
                )

    mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
    print("  [Client] Local DeepSC training finished", flush=True)

    # ------------------------------
    # GAN training in semantic space
    # ------------------------------
    local_generator = copy.deepcopy(global_generator).to(device)
    local_discriminator = SemanticDiscriminator(embedding_dim=args.d_model).to(device)

    # use a representative SNR for conditioning (e.g. mid-point of training range)
    gan_snr = float((args.snr_train_low + args.snr_train_high) / 2.0)

    gen_state, client_snr, local_prototypes = local_train_gan(
        client_model=model,
        generator=local_generator,
        discriminator=local_discriminator,
        dataloader=client_loader,
        global_prototypes=global_prototypes,
        pad_idx=pad_idx,
        snr=gan_snr,
        n_disc_steps=args.gan_disc_steps,
        gen_lr=args.gan_gen_lr,
        disc_lr=args.gan_disc_lr,
    )

    print("  [Client] Local GAN training finished", flush=True)

    return model.state_dict(), mean_loss, gen_state, client_snr, local_prototypes


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
    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "length_mild", "dirichlet"],
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.5,
        help=(
            "Dirichlet alpha for 'dirichlet' partition: "
            "lower=more non-IID (0.1=extreme, 0.5=moderate, 5.0=near-IID)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    # Saving
    parser.add_argument("--save_dir", type=str, default="checkpoints_fed")
    parser.add_argument("--save_every", type=int, default=10)

    # GAN options
    parser.add_argument(
        "--enable_gan",
        action="store_true",
        help="If set, train a conditional GAN on top of the encoder and "
        "federate only the generator using similarity-weighted aggregation.",
    )
    parser.add_argument(
        "--gan_disc_steps",
        type=int,
        default=5,
        help="Number of discriminator steps per generator step in local GAN training.",
    )
    parser.add_argument(
        "--gan_gen_lr",
        type=float,
        default=1e-4,
        help="Generator learning rate for local GAN training.",
    )
    parser.add_argument(
        "--gan_disc_lr",
        type=float,
        default=1e-4,
        help="Discriminator learning rate for local GAN training.",
    )

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

    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    token_to_idx = vocab["token_to_idx"]
    # Build idx_to_token from token_to_idx if not in vocab (preprocess only saves token_to_idx)
    idx_to_token = vocab.get("idx_to_token")
    if idx_to_token is None:
        idx_to_token = {v: k for k, v in token_to_idx.items()}

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
    elif args.partition == "length_mild":
        client_indices = partition_by_length_mild(
            train_set, args.num_clients, seed=args.seed
        )
    else:  # dirichlet
        client_indices = partition_dirichlet_length(
            train_set,
            args.num_clients,
            alpha=args.dirichlet_alpha,
            seed=args.seed,
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

    # Optional global GAN generator (discriminator stays local on clients)
    if args.enable_gan:
        global_generator = SemanticGenerator(
            semantic_dim=args.d_model,
            channel_dim=1,
            prototype_dim=args.d_model,
            output_dim=args.d_model,
        ).to(device)
        global_prototypes = None
        print("[Setup] Conditional GAN enabled (generator will be federated, discriminator stays local).", flush=True)
    else:
        global_generator = None
        global_prototypes = None

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

        # for GAN-enabled mode, also collect generator states and prototype banks
        client_gen_states = []
        client_snrs = []
        client_proto_banks = []

        for cid in selected:
            print(f"[Server] -> Training client {cid}")

            if args.enable_gan:
                st, mean_loss, gen_st, client_snr, local_protos = client_update_with_gan(
                    global_model=global_model,
                    global_generator=global_generator,
                    global_prototypes=global_prototypes,
                    client_loader=client_loaders[cid],
                    args=args,
                    pad_idx=pad_idx,
                    criterion=criterion,
                )

                client_gen_states.append(gen_st)
                client_snrs.append(client_snr)
                client_proto_banks.append(local_protos)
            else:
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
            print("[Server] Aggregating client DeepSC models (FedLol, loss-weighted)")
            global_model = fedlol(global_model, client_states, client_losses)
        else:
            print("[Server] Aggregating client DeepSC models (FedAvg)")
            global_model = fedavg(global_model, client_states, client_sizes)

        # Aggregate generator only (discriminator is never aggregated)
        if args.enable_gan and client_gen_states:
            print("[Server] Aggregating client GAN generators (similarity-weighted)", flush=True)
            agg_gen_state = federated_aggregate_generator(
                client_generator_weights=client_gen_states,
                client_snrs=client_snrs,
                client_prototypes=client_proto_banks,
            )
            global_generator.load_state_dict(agg_gen_state)

            # Update global prototype bank as a simple concatenation of client prototypes
            all_protos = [p for p in client_proto_banks if p is not None and p.numel() > 0]
            if all_protos:
                global_prototypes = torch.cat(all_protos, dim=0).detach()
            print("[Server] Updated global generator and prototype bank.", flush=True)

        if r % args.save_every == 0:
            ckpt = os.path.join(
                args.save_dir,
                f"fed_deepsc_{args.channel}_round{r:03d}.pth"
            )
            torch.save(global_model.state_dict(), ckpt)
            print(f"[Server] Checkpoint saved: {ckpt}")

            if args.enable_gan:
                gan_ckpt = os.path.join(
                    args.save_dir,
                    f"fed_gan_generator_{args.channel}_round{r:03d}.pth",
                )
                torch.save(global_generator.state_dict(), gan_ckpt)
                print(f"[Server] GAN generator checkpoint saved: {gan_ckpt}")

    # ==================================================
    # Final Model Save
    # ==================================================
    final_ckpt = os.path.join(
        args.save_dir,
        f"fed_deepsc_{args.channel}_final.pth"
    )
    torch.save(global_model.state_dict(), final_ckpt)
    print(f"\n[Done] Final DeepSC model saved to: {final_ckpt}", flush=True)

    if args.enable_gan:
        final_gan_ckpt = os.path.join(
            args.save_dir,
            f"fed_gan_generator_{args.channel}_final.pth",
        )
        torch.save(global_generator.state_dict(), final_gan_ckpt)
        print(f"[Done] Final GAN generator saved to: {final_gan_ckpt}", flush=True)

if __name__ == "__main__":
    main()
