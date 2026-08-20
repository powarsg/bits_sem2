"""
Transformer Encoder-Decoder for abstractive headline generation, built from
scratch on top of torch.nn.Transformer (multi-head self-attention encoder,
masked multi-head self+cross-attention decoder), following Vaswani et al.,
"Attention Is All You Need" (2017).
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class HeadlineTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pad_idx=0,
        max_len=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def make_src_key_padding_mask(self, src):
        return src == self.pad_idx  # (B, S)

    def make_tgt_key_padding_mask(self, tgt):
        return tgt == self.pad_idx  # (B, T)

    @staticmethod
    def make_causal_mask(sz, device):
        # Boolean mask (True = blocked) to match the dtype of the boolean
        # key-padding masks used elsewhere and avoid PyTorch's mixed
        # float/bool mask warning.
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src, tgt):
        device = src.device
        src_emb = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        src_kpm = self.make_src_key_padding_mask(src)
        tgt_kpm = self.make_tgt_key_padding_mask(tgt)
        tgt_mask = self.make_causal_mask(tgt.size(1), device)

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.generator(out)  # (B, T, vocab_size)

    def encode(self, src):
        device = src.device
        src_emb = self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model))
        src_kpm = self.make_src_key_padding_mask(src)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_kpm)
        return memory, src_kpm

    def decode_step(self, tgt, memory, src_kpm):
        device = tgt.device
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        tgt_mask = self.make_causal_mask(tgt.size(1), device)
        tgt_kpm = self.make_tgt_key_padding_mask(tgt)
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        return self.generator(out)  # (B, T, vocab_size)
