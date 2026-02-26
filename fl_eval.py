"""
fl_eval.py

Evaluate a trained federated DeepSC model on the Europarl test split and
print the BLEU score.
"""

import os
import json
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transceiver import DeepSC
from fl_data import EurDatasetLocal, collate_data
from utils import SNR_to_noise, greedy_decode, SeqtoText, BleuScore


@torch.no_grad()
def evaluate_bleu(
    model,
    data_loader,
    token_to_idx,
    pad_idx,
    start_idx,
    end_idx,
    channel: str,
    snr_db: float,
    max_len: int = 30,
):
    model.eval()
    n_var = SNR_to_noise(snr_db)

    # SeqtoText expects a token->idx dict and internally builds idx->token
    seq_to_text = SeqtoText(token_to_idx, end_idx)

    # BLEU-1 (you can also do BLEU-4 by setting weights)
    bleu = BleuScore(1.0, 0.0, 0.0, 0.0)

    scores = []
    pbar = tqdm(data_loader, desc=f"Eval BLEU @ SNR={snr_db}dB", leave=False)

    for sents in pbar:
        sents = sents.to(next(model.parameters()).device)
        pred = greedy_decode(
            model=model,
            src=sents,
            n_var=n_var,
            max_len=max_len,
            padding_idx=pad_idx,
            start_symbol=start_idx,
            channel=channel,
        )

        # Convert tokens -> text for BLEU computation
        for gt_tokens, pred_tokens in zip(sents, pred):
            gt_sent = seq_to_text.sequence_to_text(gt_tokens.tolist())
            pd_sent = seq_to_text.sequence_to_text(pred_tokens.tolist())
            if len(gt_sent.strip()) == 0 or len(pd_sent.strip()) == 0:
                continue
            scores.append(bleu.compute_blue_score([gt_sent], [pd_sent])[0])

    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def build_model(args, vocab_size: int, device: torch.device) -> torch.nn.Module:
    """
    Construct DeepSC exactly as in fl_train.py so checkpoints are compatible.
    """
    model = DeepSC(
        args.num_layers,
        vocab_size,
        vocab_size,
        vocab_size,
        vocab_size,
        args.d_model,
        args.num_heads,
        args.dff,
        args.dropout,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate federated DeepSC with BLEU.")

    # Data & vocab (match fl_train.py)
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder containing europarl/train_data.pkl etc.",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="europarl/vocab.json",
        help="Path relative to data_root or absolute.",
    )

    # DeepSC architecture (must match fl_train.py)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=30)

    # Channel & SNR (match fl_train.py defaults)
    parser.add_argument(
        "--channel",
        type=str,
        default="Rayleigh",
        choices=["AWGN", "Rayleigh", "Rician"],
    )
    parser.add_argument(
        "--snr_eval",
        type=float,
        default="0,3,6,9,12,15,18"
        help="Single SNR (dB) at which to evaluate BLEU (used if --snr_list not set).",
    )
    parser.add_argument(
        "--snr_list",
        type=str,
        default="0,3,6,9,12,15,18",
        help=(
            "Comma-separated SNR values (dB), e.g. '0,3,6,9,12,15,18'. "
            "Default: 0,3,6,9,12,15,18. Set to empty string for single --snr_eval."
        ),
    )

    # Eval settings
    parser.add_argument("--batch_size", type=int, default=128)

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Path to model checkpoint (.pth). "
            "If empty, will try checkpoints_fed/fed_deepsc_<channel>_final.pth"
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Resolve vocab path
    # --------------------------------------------------
    vocab_path = args.vocab_file
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join(args.data_root, vocab_path)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Could not find vocab file: {vocab_path}")

    with open(vocab_path, "rb") as f:
        vocab = json.load(f)

    token_to_idx = vocab["token_to_idx"]
    # idx_to_token = vocab["idx_to_token"]  # not needed directly here

    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    print(f"[Eval] Vocabulary size = {num_vocab}", flush=True)

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------
    test_set = EurDatasetLocal(args.data_root, split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_data,
    )

    print(f"[Eval] Test samples = {len(test_set)}", flush=True)

    # --------------------------------------------------
    # Build model exactly as in fl_train.py
    # --------------------------------------------------
    model = build_model(args, num_vocab, device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    ckpt_path = args.checkpoint
    if ckpt_path == "":
        ckpt_path = os.path.join(
            "checkpoints_fed",
            f"fed_deepsc_{args.channel}_final.pth",
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Pass --checkpoint PATH explicitly or run fl_train.py first."
        )

    print(f"[Eval] Loading checkpoint: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # --------------------------------------------------
    # Run BLEU evaluation (single SNR or list)
    # --------------------------------------------------
    if args.snr_list.strip():
        snr_values = [float(s.strip()) for s in args.snr_list.split(",") if s.strip()]
        if not snr_values:
            raise ValueError("--snr_list must contain at least one value, e.g. '0,3,6,9,12,15,18'")
        print(f"[Eval] Evaluating BLEU at SNR (dB): {snr_values}", flush=True)
        results = []
        for snr_db in snr_values:
            bleu = evaluate_bleu(
                model=model,
                data_loader=test_loader,
                token_to_idx=token_to_idx,
                pad_idx=pad_idx,
                start_idx=start_idx,
                end_idx=end_idx,
                channel=args.channel,
                snr_db=snr_db,
                max_len=args.max_len,
            )
            results.append((snr_db, bleu))
            print(f"  SNR={snr_db:5.1f} dB  ->  BLEU-1 = {bleu:.4f}", flush=True)
        print("-" * 40, flush=True)
        print("Summary (channel={}):".format(args.channel), flush=True)
        for snr_db, bleu in results:
            print(f"  {snr_db:5.1f} dB : {bleu:.4f}", flush=True)
    else:
        bleu = evaluate_bleu(
            model=model,
            data_loader=test_loader,
            token_to_idx=token_to_idx,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            channel=args.channel,
            snr_db=args.snr_eval,
            max_len=args.max_len,
        )
        print(
            f"[Eval] BLEU-1 @ SNR={args.snr_eval} dB, channel={args.channel}: {bleu:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

