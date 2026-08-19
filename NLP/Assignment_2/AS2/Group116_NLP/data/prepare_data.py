"""
prepare_data.py
----------------
Data collection & preprocessing for the Automatic News Headline Generation
assignment (NLP Assignment 2, Group 116).

Dataset: "News Summary" / "Inshorts News Data" dataset
Source : https://www.kaggle.com/datasets/sunnysai12345/news-summary
         (mirrored at https://github.com/sunnysai12345/News_Summary)
Files used:
  - news_summary_more.csv  (98,401 records: headline + short article text,
                             scraped from the Inshorts app)
  - news_summary.csv       (4,396 usable records: headline + full article
                             text ("ctext") scraped from source publications
                             such as The Hindu, Indian Times, Guardian, etc.
                             Used here only to provide a handful of longer,
                             realistic sample articles for the application
                             demo / screenshots, NOT for training.)
License: No explicit open-data license file is published by the dataset
         author; the data is shared publicly on Kaggle for research /
         educational use. Used here strictly for non-commercial academic
         coursework, as permitted by the assignment brief (Instruction 7).

This script performs:
  1. Loading & merging the raw CSVs.
  2. Cleaning: strip residual HTML/boilerplate, normalise unicode/whitespace,
     drop empty / very short records.
  3. Normalisation: optional lowercasing (kept as a switch -- see note in
     README / report: the Transformer model is fine-tuned on true-case text
     because sub-word tokenizers and named-entity casing matter for
     headline quality; a lowercased+normalised copy is also produced for
     the vocabulary/statistics analysis that a classical LSTM pipeline
     would need).
  4. Adding explicit <sos>/<eos> markers to the raw-text representation
     (T5's tokenizer adds its own special tokens automatically at encode
     time, but the assignment explicitly asks for this step to be shown).
  5. Building a word-level vocabulary (for analysis / would-be LSTM
     baseline) and reporting truncation coverage at the chosen max lengths.
  6. Train / validation / test split and writing processed CSVs.
"""
import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOS, EOS = "<sos>", "<eos>"

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http\S+|www\.\S+")
BOILERPLATE_RE = re.compile(
    r"\(Read More\)|Also Read:.*|Click here.*|Subscribe.*newsletter.*",
    re.IGNORECASE,
)
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(text.split())


def load_raw() -> pd.DataFrame:
    more = pd.read_csv(RAW_DIR / "news_summary_more.csv")
    more = more.rename(columns={"headlines": "headline", "text": "article"})
    more = more[["headline", "article"]]
    more["source"] = "inshorts_short"
    return more


def load_demo_articles(n: int = 12) -> pd.DataFrame:
    """A handful of longer, real full-length articles for app demo/screenshots."""
    full = pd.read_csv(RAW_DIR / "news_summary.csv", encoding="latin-1")
    full = full.rename(columns={"headlines": "headline", "ctext": "article"})
    full = full.dropna(subset=["headline", "article"])[["headline", "article"]]
    full["source"] = "full_article_demo"
    sample = full.sample(n=min(n, len(full)), random_state=42)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_article_words", type=int, default=15)
    ap.add_argument("--max_article_words", type=int, default=500)
    ap.add_argument("--min_headline_words", type=int, default=3)
    ap.add_argument("--max_headline_words", type=int, default=20)
    ap.add_argument("--sample_n", type=int, default=0,
                     help="If >0, randomly sample this many rows *after* "
                          "cleaning, to keep training time reasonable "
                          "(assignment explicitly permits this).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading raw data ...")
    df = load_raw()
    print(f"  raw records: {len(df)}")

    print("Cleaning text ...")
    df["headline"] = df["headline"].apply(clean_text)
    df["article"] = df["article"].apply(clean_text)

    # Drop empty / duplicate / very short-or-long records
    df = df[(df["headline"].str.len() > 0) & (df["article"].str.len() > 0)]
    df = df.drop_duplicates(subset=["headline", "article"])
    df["article_wc"] = df["article"].apply(word_count)
    df["headline_wc"] = df["headline"].apply(word_count)
    before = len(df)
    df = df[
        (df["article_wc"] >= args.min_article_words)
        & (df["article_wc"] <= args.max_article_words)
        & (df["headline_wc"] >= args.min_headline_words)
        & (df["headline_wc"] <= args.max_headline_words)
    ]
    print(f"  dropped {before - len(df)} records outside length bounds; "
          f"{len(df)} remain")

    # Normalised / lowercase view (for vocab & analysis only)
    df["article_norm"] = df["article"].str.lower()
    df["headline_norm"] = df["headline"].str.lower()

    # Explicit SOS/EOS-tagged target (for a from-scratch LSTM decoder;
    # T5 fine-tuning does not need this -- its tokenizer adds </s> itself)
    df["headline_tagged"] = SOS + " " + df["headline_norm"] + " " + EOS

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.sample_n and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)
        print(f"  sampled down to {len(df)} records to keep training tractable")

    n = len(df)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"  {name}: {len(split)} rows -> {OUT_DIR / f'{name}.csv'}")

    # Build a simple word-level vocabulary for analysis / LSTM baseline
    from collections import Counter
    counter = Counter()
    for text in pd.concat([train_df["article_norm"], train_df["headline_norm"]]):
        counter.update(text.split())
    vocab = ["<pad>", "<unk>", SOS, EOS] + [w for w, _ in counter.most_common(30000)]
    with open(OUT_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f)
    print(f"  vocabulary size: {len(vocab)} -> {OUT_DIR / 'vocab.json'}")

    stats = {
        "total_after_cleaning": int(n),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "article_words_mean": float(df["article_wc"].mean()),
        "article_words_median": float(df["article_wc"].median()),
        "headline_words_mean": float(df["headline_wc"].mean()),
        "headline_words_median": float(df["headline_wc"].median()),
        "vocab_size": len(vocab),
    }
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    demo = load_demo_articles()
    demo.to_csv(OUT_DIR / "demo_full_articles.csv", index=False)
    print(f"  demo full-length articles -> {OUT_DIR / 'demo_full_articles.csv'}")


if __name__ == "__main__":
    main()
