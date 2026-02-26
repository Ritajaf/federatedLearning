# fl_eval.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.transceiver import DeepSC
from models.mutual_info import Mine
from utils import SNR_to_noise, greedy_decode, SeqtoText, BleuScore

@torch.no_grad()
def evaluate_bleu(model, data_loader, idx_to_token, pad_idx, start_idx, end_idx, channel: str, snr_db: float, max_len: int = 30):
    model.eval()
    n_var = SNR_to_noise(snr_db)
    seq_to_text = SeqtoText(idx_to_token, end_idx)

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
            channel=channel
        )

        # Convert tokens -> text for BLEU computation
        for gt_tokens, pred_tokens in zip(sents, pred):
            gt_sent = seq_to_text.sequence_to_text(gt_tokens.tolist())
            pd_sent = seq_to_text.sequence_to_text(pred_tokens.tolist())
            if len(gt_sent.strip()) == 0 or len(pd_sent.strip()) == 0:
                continue
            scores.append(bleu.compute_blue_score([gt_sent], [pd_sent])[0])

    return float(np.mean(scores)) if len(scores) > 0 else 0.0
if __name__ == "__main__":
    import argparse
    from dataset import get_test_loader  # adjust if needed

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--snr_db", type=float, default=10)
    parser.add_argument("--channel", type=str, default="Rayleigh")
    args = parser.parse_args()

    # Load model
    model = DeepSC(...)
    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model.cuda()

    # Load data
    test_loader, idx_to_token, pad_idx, start_idx, end_idx = get_test_loader(args.data_root)

    # Evaluate
    bleu = evaluate_bleu(
        model=model,
        data_loader=test_loader,
        idx_to_token=idx_to_token,
        pad_idx=pad_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        channel=args.channel,
        snr_db=args.snr_db
    )

    print(f"\nBLEU score @ SNR={args.snr_db} dB ({args.channel}): {bleu:.4f}")
