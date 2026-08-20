"""
Create a smaller, fixed-size sample of the processed train/val/test splits
for actually running fine-tuning inside a CPU-only / time-limited
environment. The full processed splits (data/processed/{train,val,test}.csv)
remain available and the training script accepts --train_file pointing at
either -- so the exact same code scales to the full 88K-record training set
given a GPU (e.g. on the BITS OSHA Virtual Lab / Colab).
"""
import argparse
from pathlib import Path

import pandas as pd

PROC_DIR = Path(__file__).parent / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = PROC_DIR / "subset"
    out_dir.mkdir(exist_ok=True)

    for name, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        df = pd.read_csv(PROC_DIR / f"{name}.csv")
        df = df.sample(n=min(n, len(df)), random_state=args.seed).reset_index(drop=True)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows -> {out_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
