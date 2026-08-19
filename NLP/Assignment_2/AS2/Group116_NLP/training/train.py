"""
Fine-tune / train the from-scratch Transformer encoder-decoder for headline
generation, with teacher forcing and cross-entropy loss (padding ignored).
Logs per-epoch train/val loss to loss_history.json and plots a loss curve.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:  # Supports both `python training/train.py` and `python -m training.train`.
    from .dataset import HeadlineDataset, make_collate_fn
    from .model import HeadlineTransformer
    from .vocab import PAD, build_and_save_vocab
except ImportError:  # pragma: no cover - direct script execution
    from dataset import HeadlineDataset, make_collate_fn
    from model import HeadlineTransformer
    from vocab import PAD, build_and_save_vocab


def run_epoch(model, loader, optimizer, criterion, device, pad_idx, train=True):
    model.train() if train else model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(train):
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]

            if train:
                optimizer.zero_grad()
            logits = model(src, tgt_in)  # (B, T-1, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n_tok = (tgt_out != pad_idx).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="../data/processed/subset/train.csv")
    ap.add_argument("--val_csv", default="../data/processed/subset/val.csv")
    ap.add_argument("--out_dir", default="../models/transformer_scratch")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_src_len", type=int, default=100)
    ap.add_argument("--max_tgt_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=3)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building vocabulary from training data ...")
    vocab = build_and_save_vocab(
        args.train_csv, out_dir / "vocab.json", max_size=args.vocab_size, min_freq=args.min_freq
    )

    train_ds = HeadlineDataset(args.train_csv, vocab, args.max_src_len, args.max_tgt_len)
    val_ds = HeadlineDataset(args.val_csv, vocab, args.max_src_len, args.max_tgt_len)
    collate = make_collate_fn(vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = HeadlineTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        pad_idx=vocab.stoi[PAD],
        max_len=max(args.max_src_len, args.max_tgt_len) + 10,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Write config up-front (with placeholder timing) so the checkpoint
    # directory is usable by infer.py/app.py even if training is still
    # in progress or gets interrupted; overwritten with final values below.
    config = vars(args)
    config["vocab_size_actual"] = len(vocab)
    config["n_parameters"] = n_params
    config["device"] = str(device)
    config["total_train_time_sec"] = None
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[PAD], label_smoothing=args.label_smoothing)

    history = {"train_loss": [], "val_loss": [], "epoch_time_sec": []}
    best_val = float("inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, vocab.stoi[PAD], train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, vocab.stoi[PAD], train=False)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time_sec"].append(dt)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        with open(out_dir / "loss_history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    config = vars(args)
    config["vocab_size_actual"] = len(vocab)
    config["n_parameters"] = n_params
    config["device"] = str(device)
    config["total_train_time_sec"] = total_time
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Plot loss curve
    plt.figure(figsize=(6, 4))
    epochs_range = list(range(1, args.epochs + 1))
    plt.plot(epochs_range, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs_range, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss (per token)")
    plt.title("Transformer headline generator - training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    print(f"\nDone in {total_time:.1f}s. Best val loss: {best_val:.4f}")
    print(f"Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
