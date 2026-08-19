"""
Evaluate the trained headline generator on the held-out test split using
ROUGE-1 / ROUGE-2 / ROUGE-L (F1), and save a sample of predictions for
qualitative inspection (factual consistency, repetition, generic wording,
named-entity handling -- see report Task 5.3).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "training"))

from infer import HeadlineGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT_DIR / "models" / "transformer_scratch"))
    ap.add_argument("--test_csv", default=str(ROOT_DIR / "data" / "processed" / "subset" / "test.csv"))
    ap.add_argument("--out_dir", default=str(ROOT_DIR / "evaluation" / "results"))
    ap.add_argument("--n_eval", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from rouge_score import rouge_scorer

    gen = HeadlineGenerator(args.model_dir)
    df = pd.read_csv(args.test_csv).head(args.n_eval)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    samples = []

    for i, row in df.iterrows():
        article = str(row["article"])
        reference = str(row["headline"])
        pred = gen.greedy_decode(article)

        s = scorer.score(reference, pred)
        for k in scores:
            scores[k].append(s[k].fmeasure)

        if len(samples) < 25:
            samples.append({
                "article": article,
                "reference_headline": reference,
                "predicted_headline": pred,
                "rouge1_f": s["rouge1"].fmeasure,
                "rouge2_f": s["rouge2"].fmeasure,
                "rougeL_f": s["rougeL"].fmeasure,
            })

    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}
    summary["n_eval"] = len(df)

    with open(out_dir / "rouge_scores.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(samples).to_csv(out_dir / "sample_predictions.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSample predictions -> {out_dir / 'sample_predictions.csv'}")


if __name__ == "__main__":
    main()
