import pandas as pd
import torch
from torch.utils.data import Dataset

from vocab import PAD, SOS, EOS


class HeadlineDataset(Dataset):
    def __init__(self, csv_path, vocab, max_src_len=100, max_tgt_len=20):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_ids = self.vocab.encode(str(row["article"]), max_len=self.max_src_len)
        tgt_ids = self.vocab.encode(
            str(row["headline"]), max_len=self.max_tgt_len - 2, add_sos_eos=True
        )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def make_collate_fn(vocab):
    pad_id = vocab.stoi[PAD]

    def collate(batch):
        srcs, tgts = zip(*batch)
        src_len = max(len(s) for s in srcs)
        tgt_len = max(len(t) for t in tgts)
        src_pad = torch.full((len(srcs), src_len), pad_id, dtype=torch.long)
        tgt_pad = torch.full((len(tgts), tgt_len), pad_id, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src_pad[i, : len(s)] = s
            tgt_pad[i, : len(t)] = t
        return src_pad, tgt_pad

    return collate
