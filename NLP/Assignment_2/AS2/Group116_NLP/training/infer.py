"""Greedy and beam-search decoding for the from-scratch Transformer headline
generator, plus a small HeadlineGenerator wrapper used by both the Flask app
and the evaluation script."""
import json
from pathlib import Path

import torch

from model import HeadlineTransformer
from vocab import EOS, PAD, SOS, Vocab


class HeadlineGenerator:
    def __init__(self, model_dir, device=None):
        model_dir = Path(model_dir)
        with open(model_dir / "train_config.json") as f:
            self.config = json.load(f)
        self.vocab = Vocab.load(model_dir / "vocab.json")
        self.device = device or torch.device("cpu")

        self.model = HeadlineTransformer(
            vocab_size=len(self.vocab),
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_encoder_layers=self.config["enc_layers"],
            num_decoder_layers=self.config["dec_layers"],
            dim_feedforward=self.config["ffn"],
            dropout=0.0,
            pad_idx=self.vocab.stoi[PAD],
            max_len=max(self.config["max_src_len"], self.config["max_tgt_len"]) + 10,
        ).to(self.device)
        # Checkpoints shipped in the code archive may be stored in fp16 to
        # keep the zip under upload size limits; cast back to fp32 for
        # numerically stable CPU inference regardless of how it was saved.
        state = torch.load(model_dir / "best_model.pt", map_location=self.device)
        state = {k: v.float() if torch.is_floating_point(v) else v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()

        self.max_src_len = self.config["max_src_len"]
        self.max_tgt_len = self.config["max_tgt_len"]

    @torch.no_grad()
    def _encode(self, article_text):
        src_ids = self.vocab.encode(article_text, max_len=self.max_src_len)
        if len(src_ids) == 0:
            src_ids = [self.vocab.stoi[PAD]]
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        memory, src_kpm = self.model.encode(src)
        return memory, src_kpm

    @staticmethod
    def _banned_next_tokens(seq, no_repeat_ngram_size=3):
        """Standard no-repeat-ngram blocking (as in HF `generate`), plus a
        direct ban on immediately repeating the previous token -- mitigates
        the stuttering repetition ("indian indian indian ...") that small,
        lightly-trained decoder LMs are prone to (see report Task 5.3)."""
        banned = set()
        if len(seq) >= 1:
            banned.add(seq[-1])  # no immediate token repeat
        n = no_repeat_ngram_size
        if len(seq) >= n - 1 and n >= 2:
            prefix = tuple(seq[-(n - 1):])
            for i in range(len(seq) - n + 1):
                if tuple(seq[i:i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
        return banned

    @torch.no_grad()
    def greedy_decode(self, article_text, max_len=None, no_repeat_ngram_size=3):
        max_len = max_len or self.max_tgt_len
        memory, src_kpm = self._encode(article_text)
        sos, eos = self.vocab.stoi[SOS], self.vocab.stoi[EOS]
        ys = torch.tensor([[sos]], dtype=torch.long, device=self.device)
        seq = [sos]
        for _ in range(max_len - 1):
            logits = self.model.decode_step(ys, memory, src_kpm)
            scores = logits[0, -1].clone()
            for tok in self._banned_next_tokens(seq, no_repeat_ngram_size):
                scores[tok] = float("-inf")
            next_id = scores.argmax().item()
            ys = torch.cat([ys, torch.tensor([[next_id]], device=self.device)], dim=1)
            seq.append(next_id)
            if next_id == eos:
                break
        return self.vocab.decode(ys[0].tolist())

    @torch.no_grad()
    def beam_search_decode(self, article_text, beam_size=4, max_len=None, top_k=3,
                            length_penalty=0.7, no_repeat_ngram_size=3):
        """Returns up to top_k candidate headlines, best first."""
        max_len = max_len or self.max_tgt_len
        memory, src_kpm = self._encode(article_text)
        sos, eos = self.vocab.stoi[SOS], self.vocab.stoi[EOS]

        # each beam: (token_id_list, cumulative_log_prob, finished)
        beams = [([sos], 0.0, False)]
        completed = []

        for _ in range(max_len - 1):
            new_beams = []
            active = [b for b in beams if not b[2]]
            if not active:
                break
            for seq, score, _ in active:
                ys = torch.tensor([seq], dtype=torch.long, device=self.device)
                logits = self.model.decode_step(ys, memory, src_kpm)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                for tok in self._banned_next_tokens(seq, no_repeat_ngram_size):
                    log_probs[tok] = float("-inf")
                k = min(beam_size, (log_probs > float("-inf")).sum().item()) or 1
                topk_logp, topk_idx = log_probs.topk(k)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    if lp == float("-inf"):
                        continue
                    new_seq = seq + [idx]
                    new_score = score + lp
                    finished = idx == eos
                    if finished:
                        completed.append((new_seq, new_score, True))
                    else:
                        new_beams.append((new_seq, new_score, False))
            # keep best `beam_size` active beams by length-normalised score
            new_beams.sort(key=lambda b: b[1] / (len(b[0]) ** length_penalty), reverse=True)
            beams = new_beams[:beam_size]
            if not beams:
                break

        all_candidates = completed + beams
        all_candidates.sort(key=lambda b: b[1] / (len(b[0]) ** length_penalty), reverse=True)

        results = []
        seen = set()
        for seq, score, _ in all_candidates:
            text = self.vocab.decode(seq)
            if text and text not in seen:
                seen.add(text)
                results.append(text)
            if len(results) >= top_k:
                break
        if not results:
            results = [self.greedy_decode(article_text, max_len)]
        return results
