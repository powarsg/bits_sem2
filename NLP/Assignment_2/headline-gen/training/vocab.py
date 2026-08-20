"""Simple word-level vocabulary for the from-scratch Transformer encoder-decoder."""
import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = [PAD, UNK, SOS, EOS]

TOKEN_RE = re.compile(r"[a-z0-9]+|[.,!?;:'\"()%$-]")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def encode(self, text, max_len=None, add_sos_eos=False):
        toks = tokenize(text)
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
        if add_sos_eos:
            ids = [self.stoi[SOS]] + ids + [self.stoi[EOS]]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids, strip_special=True):
        toks = []
        for i in ids:
            s = self.itos.get(int(i), UNK)
            if strip_special and s in SPECIALS:
                if s == EOS:
                    break
                continue
            toks.append(s)
        text = " ".join(toks)
        # Keep generated headlines readable: punctuation should follow the
        # preceding word, while opening punctuation should not have a space
        # after it.  Token IDs remain unchanged for model training.
        text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
        text = re.sub(r"([('\"])\s+", r"\1", text)
        return text

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stoi, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            stoi = json.load(f)
        return cls(stoi)

    @classmethod
    def build(cls, texts, max_size=8000, min_freq=2):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        stoi = {s: i for i, s in enumerate(SPECIALS)}
        for word, freq in counter.most_common():
            if freq < min_freq:
                continue
            if len(stoi) >= max_size:
                break
            if word not in stoi:
                stoi[word] = len(stoi)
        return cls(stoi)


def build_and_save_vocab(train_csv, out_path, max_size=8000, min_freq=2):
    import pandas as pd
    df = pd.read_csv(train_csv)
    texts = list(df["article"].astype(str)) + list(df["headline"].astype(str))
    vocab = Vocab.build(texts, max_size=max_size, min_freq=min_freq)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(out_path)
    print(f"Built vocab of size {len(vocab)} -> {out_path}")
    return vocab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_size", type=int, default=8000)
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()
    build_and_save_vocab(args.train_csv, args.out, args.max_size, args.min_freq)
