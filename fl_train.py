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
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    federated_aggregate_discriminator,
    LearnedPrototypeBank,
    build_prototype_bank,
    extract_sentence_embeddings,
)

# use GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# CHANGE 1: Two-stage training schedule constants
BACKBONE_PRETRAIN_ROUNDS = 30
GAN_START_ROUND = 25
PROTO_INIT_ROUND = BACKBONE_PRETRAIN_ROUNDS + 1  # 31
TOTAL_ROUNDS = 50

# CHANGE 6: Minimum GAN generator update steps per client
MIN_GAN_STEPS = 30

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
    global_prototype_bank,
    global_discriminator,
    client_loader,
    args,
    pad_idx,
    criterion,
    gan_gen_lr: float,
    gan_disc_lr: float,
    min_gan_steps: int,
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
                global_prototypes=global_prototype_bank,
                no_prototype=args.no_prototype,
                no_snr=args.no_snr,
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
    # CHANGE (temporary): if naive_gan is enabled, initialize from global aggregated discriminator
    if global_discriminator is not None:
        local_discriminator = copy.deepcopy(global_discriminator).to(device)
    else:
        local_discriminator = SemanticDiscriminator(embedding_dim=args.d_model).to(device)

    # use a representative SNR for conditioning (e.g. mid-point of training range)
    gan_snr = float((args.snr_train_low + args.snr_train_high) / 2.0)

    # local_train_gan now returns: gen_state, client_snr, local_proto_tensor, local_gen_recon_loss
    gen_state, client_snr, local_prototypes, local_gen_recon_loss, disc_state = local_train_gan(
        client_model=model,
        generator=local_generator,
        discriminator=local_discriminator,
        dataloader=client_loader,
        global_prototype_bank=global_prototype_bank,
        pad_idx=pad_idx,
        snr=gan_snr,
        n_disc_steps=args.gan_disc_steps,
        gen_lr=gan_gen_lr,
        disc_lr=gan_disc_lr,
        lambda_adv=args.gan_lambda_adv,
        lambda_div=args.gan_lambda_div,
        n_prototypes=args.gan_n_prototypes,
        min_gan_steps=min_gan_steps,
        no_prototype=args.no_prototype,
        no_snr=args.no_snr,
    )

    print("  [Client] Local GAN training finished", flush=True)

    # CHANGE 7: return local generator reconstruction loss for quality-weighted aggregation
    # CHANGE (temporary): also return discriminator state_dict for optional naive discriminator aggregation
    return (
        model.state_dict(),
        mean_loss,
        gen_state,
        client_snr,
        local_prototypes,
        local_gen_recon_loss,
        disc_state,
    )


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
    # CHANGE (temporary): discriminator aggregation baseline
    parser.add_argument(
        "--naive_gan",
        action="store_true",
        help="Aggregate discriminator like generator (naive baseline).",
    )
    # CHANGE (temporary): ablate conditioning signals
    parser.add_argument(
        "--no_prototype",
        action="store_true",
        help="Remove prototype conditioning from generator.",
    )
    parser.add_argument(
        "--no_snr",
        action="store_true",
        help="Remove SNR conditioning from generator.",
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
    parser.add_argument(
        "--gan_lambda_adv",
        type=float,
        default=0.1,
        help="Weight on adversarial (generator vs discriminator) term in local GAN loss.",
    )
    parser.add_argument(
        "--gan_lambda_div",
        type=float,
        default=0.01,
        help="Weight on cosine divergence between nearest global prototype and local prototype vec.",
    )
    parser.add_argument(
        "--gan_n_prototypes",
        type=int,
        default=10,
        help="Number of KMeans cluster centers per client when building the local prototype bank.",
    )

    args = parser.parse_args()
    if args.gan_n_prototypes < 1:
        parser.error("--gan_n_prototypes must be >= 1")
    # CHANGE 1: enforce total rounds = 50 as requested
    args.rounds = TOTAL_ROUNDS
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
        # CHANGE 1 + 8: learned prototype bank is initialized at round 31
        global_prototype_bank = None
        global_discriminator = None
        if args.naive_gan:
            global_discriminator = SemanticDiscriminator(embedding_dim=args.d_model).to(device)
            print("[Setup] Naive GAN enabled: discriminator will be aggregated.", flush=True)
        print("[Setup] Conditional GAN enabled (generator will be federated, discriminator stays local).", flush=True)
    else:
        global_generator = None
        global_prototype_bank = None
        global_discriminator = None

    os.makedirs(args.save_dir, exist_ok=True)

    # CHANGE 5: Cosine annealing LR schedule for generator ONLY during stage 2/3 (rounds >= 31)
    # We create a dummy optimizer + scheduler so we can read its current lr each round.
    gen_lr_scheduler = None
    gen_lr_schedule_optimizer = None
    gen_lr_base = 1e-5
    gen_lr_min = 1e-7
    gen_disc_lr_fixed_stage2 = 5e-5
    if args.enable_gan:
        gen_lr_schedule_optimizer = torch.optim.Adam(
            global_generator.parameters(),
            lr=gen_lr_base,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        gen_lr_scheduler = CosineAnnealingLR(
            gen_lr_schedule_optimizer,
            T_max=TOTAL_ROUNDS - BACKBONE_PRETRAIN_ROUNDS,
            eta_min=gen_lr_min,
        )

    def _init_learned_prototype_bank_from_converged_embeddings():
        """
        CHANGE 1: At round 31, initialize the learned prototype bank from
        sentence embeddings extracted from the converged backbone encoder.
        """
        global_model.eval()
        max_total_samples = min(5000, max(500, args.gan_n_prototypes * 200))
        max_batches_per_client = 3

        collected = []
        collected_count = 0
        with torch.no_grad():
            for loader in client_loaders:
                for bi, batch in enumerate(loader):
                    if collected_count >= max_total_samples:
                        break
                    emb = extract_sentence_embeddings(global_model, batch, pad_idx)  # [B, d_model]
                    collected.append(emb.detach())
                    collected_count += emb.size(0)
                    if bi + 1 >= max_batches_per_client:
                        break

        if not collected:
            raise RuntimeError("Prototype init failed: no embeddings collected.")

        all_emb = torch.cat(collected, dim=0)
        if all_emb.size(0) > max_total_samples:
            all_emb = all_emb[:max_total_samples]

        init_protos = build_prototype_bank(all_emb, n_prototypes=args.gan_n_prototypes)
        return init_protos

    # ==================================================
    # Federated Training Loop (CHANGE 1 + CHANGE 5)
    # ==================================================
    for r in range(1, TOTAL_ROUNDS + 1):

        print("\n" + "=" * 70)
        print(f"[Server] Federated Round {r}/{TOTAL_ROUNDS}")
        print("=" * 70)

        # CHANGE 1: stage indicators + two-stage training schedule
        if args.enable_gan:
            if r <= 24:
                stage_label = "Stage 1: Backbone-only (NO GAN)"
            elif r <= BACKBONE_PRETRAIN_ROUNDS:
                stage_label = "Stage 2: Full GAN training (prototype bank not initialized)"
            else:
                stage_label = "Stage 3: Learned prototype bank + cosine LR (generator)"
            print(f"[Server] {stage_label}", flush=True)

        # Initialize learned prototype bank at round 31
        if args.enable_gan and r == PROTO_INIT_ROUND:
            if global_prototype_bank is None:
                print("[Server][Stage 3] Initializing learned prototype bank from converged embeddings...", flush=True)
                init_protos = _init_learned_prototype_bank_from_converged_embeddings()
                global_prototype_bank = LearnedPrototypeBank(
                    n_prototypes=args.gan_n_prototypes,
                    d_model=args.d_model,
                ).to(device)
                global_prototype_bank.prototypes.data.copy_(init_protos)
                print("[Server][Stage 3] Prototype bank initialized.", flush=True)

        selected = np.random.choice(
            args.num_clients,
            size=min(args.clients_per_round, args.num_clients),
            replace=False,
        )
        print(f"[Server] Selected clients: {selected.tolist()}")

        client_states = []
        client_sizes = []
        client_losses = []

        # For GAN-enabled rounds: collect generator + prototype updates and generator-quality loss
        client_gen_states = []
        client_snrs = []
        client_proto_banks = []
        client_gen_recon_losses = []
        client_disc_states = []

        # Decide whether GAN runs this round and what LR to use for generator/discriminator
        run_gan_this_round = args.enable_gan and r >= GAN_START_ROUND
        if run_gan_this_round:
            if r >= PROTO_INIT_ROUND:
                current_gen_lr = gen_lr_schedule_optimizer.param_groups[0]["lr"]
                current_disc_lr = gen_disc_lr_fixed_stage2
            else:
                current_gen_lr = args.gan_gen_lr
                current_disc_lr = args.gan_disc_lr
        else:
            current_gen_lr = None
            current_disc_lr = None

        for cid in selected:
            print(f"[Server] -> Training client {cid}")

            if run_gan_this_round:
                st, mean_loss, gen_st, client_snr, local_proto_tensor, local_gen_recon_loss, disc_state = client_update_with_gan(
                    global_model=global_model,
                    global_generator=global_generator,
                    global_prototype_bank=global_prototype_bank,
                    global_discriminator=global_discriminator,
                    client_loader=client_loaders[cid],
                    args=args,
                    pad_idx=pad_idx,
                    criterion=criterion,
                    gan_gen_lr=current_gen_lr,
                    gan_disc_lr=current_disc_lr,
                    min_gan_steps=MIN_GAN_STEPS,
                )
                client_gen_states.append(gen_st)
                client_snrs.append(client_snr)
                client_proto_banks.append(local_proto_tensor)
                client_gen_recon_losses.append(local_gen_recon_loss)
                client_disc_states.append(disc_state)
            else:
                st, mean_loss = client_update(
                    global_model=global_model,
                    client_loader=client_loaders[cid],
                    args=args,
                    pad_idx=pad_idx,
                    criterion=criterion,
                )

            client_states.append(st)
            client_sizes.append(len(client_loaders[cid].dataset))
            client_losses.append(mean_loss)

        # Aggregate DeepSC backbone (FedAvg/FedLol)
        alg = getattr(args, "algorithm", "fedlol")
        if alg == "fedlol":
            print("[Server] Aggregating client DeepSC models (FedLol, loss-weighted)")
            global_model = fedlol(global_model, client_states, client_losses)
        else:
            print("[Server] Aggregating client DeepSC models (FedAvg)")
            global_model = fedavg(global_model, client_states, client_sizes)

        # Aggregate GAN generator (+ prototype bank if initialized)
        if run_gan_this_round and client_gen_states:
            print("[Server] Aggregating client GAN generators (CHANGE 7 quality-weighted)", flush=True)
            agg_gen_state, agg_proto_tensor = federated_aggregate_generator(
                client_generator_weights=client_gen_states,
                client_snrs=client_snrs,
                client_prototypes=client_proto_banks,
                client_gen_recon_losses=client_gen_recon_losses,
            )
            global_generator.load_state_dict(agg_gen_state)

            # If learned prototype bank exists, update global prototypes with same weights (CHANGE 8)
            if global_prototype_bank is not None and agg_proto_tensor is not None and agg_proto_tensor.numel() > 0:
                global_prototype_bank.prototypes.data.copy_(agg_proto_tensor.to(device))
                print("[Server] Updated global learned prototype bank.", flush=True)
            else:
                print("[Server] Prototype bank not active yet (round < 31).", flush=True)

        # CHANGE (temporary): optionally aggregate discriminators like generator
        if run_gan_this_round and args.naive_gan and client_disc_states:
            print("[Server] Aggregating client GAN discriminators (naive baseline)", flush=True)
            agg_disc_state = federated_aggregate_discriminator(
                client_discriminator_weights=client_disc_states,
                client_snrs=client_snrs,
                client_prototypes=client_proto_banks,
                client_gen_recon_losses=client_gen_recon_losses,
            )
            global_discriminator.load_state_dict(agg_disc_state)

        # CHANGE 5: cosine annealing LR update at end of each Stage 3 round
        if args.enable_gan and r >= PROTO_INIT_ROUND and gen_lr_scheduler is not None:
            gen_lr_scheduler.step()
            print(f"[Server] Stage 3 generator lr -> {gen_lr_schedule_optimizer.param_groups[0]['lr']:.3e}", flush=True)

        if r % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"fed_deepsc_{args.channel}_round{r:03d}.pth")
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
