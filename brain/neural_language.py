from __future__ import annotations

import contextlib
import math
import re
import zlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_PL_TRANS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _strip_pl(text: str) -> str:
    return text.translate(_PL_TRANS)


class SubwordTokenizer:
    """
    Lossless v2 byte-plus-word tokenizer with dialogue structure markers and space-prefixed words.
    - IDs 0..10  : special tokens (<PAD>, <BOS>, <EOS>, <UNK>, <SEP>, <Q>, <A>, <MEM>, <CTX>, <USER>, <BOT>)
    - IDs 11..266: UTF-8 byte fallback (100% lossless for Polish diacritics & code symbols)
    - IDs 267+   : dynamically learned Polish/code words & space-prefixed words up to max_vocab_size
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SEP_ID = 4
    Q_ID = 5
    A_ID = 6
    MEM_ID = 7
    CTX_ID = 8
    USER_ID = 9
    BOT_ID = 10
    BYTE_OFFSET = 11
    VOCAB_OFFSET = 267

    _SPECIAL_STR_TO_ID = {
        "<SEP>": SEP_ID,
        "<Q>": Q_ID,
        "<A>": A_ID,
        "<MEM>": MEM_ID,
        "<CTX>": CTX_ID,
        "<USER>": USER_ID,
        "<BOT>": BOT_ID,
    }
    _CONTROL_IDS = {PAD_ID, BOS_ID, UNK_ID, SEP_ID, Q_ID, A_ID, MEM_ID, CTX_ID, USER_ID, BOT_ID}

    _SPLIT_RE = re.compile(
        r"<MEM>|<CTX>|<USER>|<BOT>|<Q>|<A>|<SEP>| ?[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9_+#]+|\s+|.",
        re.DOTALL,
    )
    _LEARN_RE = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9_+#]{2,40}")

    def __init__(self, max_vocab_size: int = 16384) -> None:
        self.max_vocab_size = max_vocab_size
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self._next_id = self.VOCAB_OFFSET

    @property
    def vocab_used(self) -> int:
        return self._next_id

    def learn_texts(self, texts: list[str]) -> int:
        added = 0
        for txt in texts:
            if not txt:
                continue
            for tok in self._LEARN_RE.findall(txt):
                for variant in (tok, tok.lower(), f" {tok}", f" {tok.lower()}"):
                    if variant not in self.word_to_id:
                        if self._next_id >= self.max_vocab_size:
                            return added
                        tid = self._next_id
                        self.word_to_id[variant] = tid
                        self.id_to_word[tid] = variant
                        self._next_id += 1
                        added += 1
        return added

    def encode(
        self,
        text: str,
        max_len: int = 320,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        ids: list[int] = [self.BOS_ID] if add_bos else []
        limit = max_len - (1 if add_eos else 0)
        for piece in self._SPLIT_RE.findall(text):
            if len(ids) >= limit:
                break
            if piece in self._SPECIAL_STR_TO_ID:
                ids.append(self._SPECIAL_STR_TO_ID[piece])
            elif piece in self.word_to_id:
                ids.append(self.word_to_id[piece])
            elif piece.startswith(" ") and len(piece) > 1 and piece[1:] in self.word_to_id:
                ids.append(self.BYTE_OFFSET + 32)
                if len(ids) < limit:
                    ids.append(self.word_to_id[piece[1:]])
            else:
                for b in piece.encode("utf-8"):
                    if len(ids) >= limit:
                        break
                    ids.append(self.BYTE_OFFSET + b)
        if add_eos and len(ids) < max_len:
            ids.append(self.EOS_ID)
        return ids[:max_len]

    def decode(self, ids: list[int]) -> str:
        out_parts: list[str] = []
        byte_buf = bytearray()

        def flush_bytes() -> None:
            if byte_buf:
                out_parts.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            if tid in (self.Q_ID, self.A_ID, self.MEM_ID, self.CTX_ID, self.USER_ID, self.BOT_ID):
                flush_bytes()
                out_parts.append(" ")
            elif tid == self.SEP_ID:
                flush_bytes()
                out_parts.append(" | ")
            elif self.BYTE_OFFSET <= tid < self.VOCAB_OFFSET:
                byte_buf.append(tid - self.BYTE_OFFSET)
            else:
                flush_bytes()
                out_parts.append(self.id_to_word.get(tid, ""))
        flush_bytes()
        return "".join(out_parts)

    def export_state(self) -> dict[str, Any]:
        return {
            "version": 2,
            "max_vocab_size": self.max_vocab_size,
            "word_to_id": self.word_to_id,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 1)) != 2:
            # Legacy v1 tokenizer offset differs; reset cleanly and re-learn from memories
            self.word_to_id = {}
            self.id_to_word = {}
            self._next_id = self.VOCAB_OFFSET
            return
        raw_map = state.get("word_to_id", {})
        self.word_to_id = {}
        self.id_to_word = {}
        max_id = self.VOCAB_OFFSET
        for w, tid in raw_map.items():
            idx = int(tid)
            if self.VOCAB_OFFSET <= idx < self.max_vocab_size:
                self.word_to_id[str(w)] = idx
                self.id_to_word[idx] = str(w)
                if idx + 1 > max_id:
                    max_id = idx + 1
        self._next_id = max_id


def extract_ngram_features(
    texts: list[str],
    num_buckets: int = 2048,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build deterministic L2-normalized sparse n-gram + Polish stem feature vectors
    so inflectional variants share strong geometric overlap from initialization.
    """
    n = len(texts)
    mat = torch.zeros((n, num_buckets), dtype=torch.float32)
    for row, raw in enumerate(texts):
        low = (raw or "").strip().lower()
        if not low:
            continue
        stripped = _strip_pl(low)
        words = re.findall(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9+#]{2,}", low)
        words_s = re.findall(r"[a-z0-9+#]{2,}", stripped)

        for w in words:
            b = zlib.crc32(f"w:{w}".encode("utf-8")) % num_buckets
            mat[row, b] += 2.2
        for ws in words_s:
            b = zlib.crc32(f"ws:{ws}".encode("utf-8")) % num_buckets
            mat[row, b] += 1.8
            if len(ws) >= 5:
                stem = ws[: max(4, len(ws) - 2)]
                bs = zlib.crc32(f"st:{stem}".encode("utf-8")) % num_buckets
                mat[row, bs] += 1.6

        compact = re.sub(r"\s+", " ", stripped)
        for gram_len in (3, 4):
            if len(compact) >= gram_len:
                for i in range(len(compact) - gram_len + 1):
                    g = compact[i : i + gram_len]
                    bg = zlib.crc32(f"g{gram_len}:{g}".encode("utf-8")) % num_buckets
                    mat[row, bg] += 0.7

    mat = F.normalize(mat, dim=-1)
    return mat.to(device) if device is not None else mat


class CorticalSemanticEncoder(nn.Module):
    """
    Self-contained 256D GPU Semantic Encoder:
      - Fixed deterministic n-gram projection preserves zero-shot Polish stem similarity.
      - Trainable token embeddings + residual MLP + gated fusion fine-tuned contrastively
        during NREM sleep and language training with InfoNCE on GPU.
    """

    def __init__(self, vocab_size: int = 16384, ngram_buckets: int = 2048, d_model: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram_buckets = ngram_buckets
        self.d_model = d_model

        gen = torch.Generator().manual_seed(42)
        proj = torch.randn(ngram_buckets, d_model, generator=gen) / math.sqrt(d_model)
        self.register_buffer("ngram_proj", F.normalize(proj, dim=-1))

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.token_emb.weight[0].zero_()

        self.res_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.20))

    def forward(self, token_ids: torch.Tensor, ngram_feats: torch.Tensor) -> torch.Tensor:
        base = F.normalize(ngram_feats @ self.ngram_proj, dim=-1)
        mask = (token_ids != 0).unsqueeze(-1).float()
        tok_sum = (self.token_emb(token_ids) * mask).sum(dim=1)
        tok_mean = tok_sum / mask.sum(dim=1).clamp(min=1.0)
        delta = tok_mean + self.res_net(tok_mean + base)
        out = base + torch.sigmoid(self.gate) * 0.45 * delta
        return F.normalize(out, dim=-1)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        tokenizer: SubwordTokenizer,
        batch_size: int = 32,
        max_len: int = 128,
    ) -> list[list[float]]:
        if not texts:
            return []
        device = self.ngram_proj.device
        self.eval()
        results: list[list[float]] = [[] for _ in texts]
        non_empty = [(idx, t.strip()[:1000]) for idx, t in enumerate(texts) if t and t.strip()]
        for start in range(0, len(non_empty), max(1, batch_size)):
            batch = non_empty[start : start + max(1, batch_size)]
            b_texts = [t for _, t in batch]
            encoded = [tokenizer.encode(t, max_len=max_len, add_bos=True, add_eos=True) for t in b_texts]
            max_t = max(len(seq) for seq in encoded)
            tok_tensor = torch.zeros((len(encoded), max_t), dtype=torch.long, device=device)
            for r, seq in enumerate(encoded):
                tok_tensor[r, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            ngram_t = extract_ngram_features(b_texts, num_buckets=self.ngram_buckets, device=device)
            vecs = self.forward(tok_tensor, ngram_t).cpu().tolist()
            for (orig_idx, _), v in zip(batch, vecs):
                results[orig_idx] = [round(float(x), 5) for x in v]
        return results

    def train_contrastive(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: SubwordTokenizer,
        epochs: int = 8,
        lr: float = 8e-4,
    ) -> tuple[float, float]:
        if len(pairs) < 2:
            return (0.0, 0.0)
        device = self.ngram_proj.device
        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        qs = [p[0] for p in pairs]
        ans = [p[1] for p in pairs]

        def to_batch(items: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            seqs = [tokenizer.encode(t, max_len=128, add_bos=True, add_eos=True) for t in items]
            max_t = max(len(s) for s in seqs)
            tok = torch.zeros((len(seqs), max_t), dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                tok[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            ng = extract_ngram_features(items, num_buckets=self.ngram_buckets, device=device)
            return tok, ng

        q_tok, q_ng = to_batch(qs)
        a_tok, a_ng = to_batch(ans)
        labels = torch.arange(len(pairs), dtype=torch.long, device=device)

        anchor = {k: v.detach().clone() for k, v in self.named_parameters()}
        loss_start = 0.0
        loss_end = 0.0

        for ep in range(max(1, epochs)):
            q_vec = self.forward(q_tok, q_ng)
            a_vec = self.forward(a_tok, a_ng)
            logits = (q_vec @ a_vec.T) / 0.12
            loss_ce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            reg = sum((p - anchor[n]).pow(2).sum() for n, p in self.named_parameters()) * 1e-3
            loss = loss_ce + reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5)
            opt.step()

            val = float(loss.item())
            if ep == 0:
                loss_start = val
            loss_end = val

        return (round(loss_start, 4), round(loss_end, 4))

    def export_state(self) -> dict[str, Any]:
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def import_state(self, state: dict[str, Any]) -> None:
        device = self.ngram_proj.device
        cur_sd = self.state_dict()
        merged = {}
        for k, cur_v in cur_sd.items():
            if k in state and state[k].shape == cur_v.shape:
                merged[k] = state[k].to(device)
            elif k in state and state[k].ndim == 2 and cur_v.ndim == 2 and state[k].shape[1] == cur_v.shape[1]:
                buf = cur_v.clone()
                rows = min(buf.shape[0], state[k].shape[0])
                buf[:rows] = state[k][:rows].to(device)
                merged[k] = buf
            else:
                merged[k] = cur_v
        self.load_state_dict(merged)


class _CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h_in = self.ln1(x)
        qkv = self.qkv(h_in).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class CorticalTransformerLM(nn.Module):
    """
    Self-contained 6-layer Pre-Norm Causal Transformer trained from scratch in PyTorch on GPU.
    Trains on conversational sequences (<MEM>...<CTX>...<USER>...<BOT>...<EOS>) with bfloat16
    and generates natural Polish replies autoregressively.
    """

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 1536,
        max_seq_len: int = 320,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[0].zero_()

        self.blocks = nn.ModuleList(
            [_CausalTransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        self.trained_steps: int = 0
        self.last_loss: float = 10.0
        self.fluency_score: float = 0.0

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def _amp_ctx(self, device: torch.device):
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def sequence_scores(
        self,
        texts: list[str],
        tokenizer: SubwordTokenizer,
        batch_size: int = 24,
    ) -> list[float]:
        if not texts:
            return []
        device = self.tok_emb.weight.device
        self.eval()
        scores: list[float] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            seqs = [tokenizer.encode(t, max_len=self.max_seq_len, add_bos=True, add_eos=True) for t in chunk]
            max_t = max(2, max(len(s) for s in seqs))
            batch = torch.zeros((len(seqs), max_t), dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            with self._amp_ctx(device):
                logits = self.forward(batch[:, :-1])
                targets = batch[:, 1:]
                loss_per_tok = F.cross_entropy(
                    logits.float().reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    ignore_index=0,
                    reduction="none",
                ).view(len(seqs), -1)
            mask = (targets != 0).float()
            mean_loss = (loss_per_tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            for val in mean_loss.cpu().tolist():
                scores.append(round(float(-val), 4))
        return scores

    def train_sleep_replay(
        self,
        sequences: list[str],
        tokenizer: SubwordTokenizer,
        epochs: int = 12,
        lr: float = 9e-4,
        batch_size: int = 32,
    ) -> tuple[float, float]:
        clean_seqs = [s.strip() for s in sequences if s and len(s.strip()) >= 6]
        if not clean_seqs:
            return (0.0, 0.0)
        device = self.tok_emb.weight.device
        encoded = [
            tokenizer.encode(s, max_len=self.max_seq_len, add_bos=True, add_eos=True)
            for s in clean_seqs
        ]
        encoded = [s for s in encoded if len(s) >= 3]
        if not encoded:
            return (0.0, 0.0)

        self.train()
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        loss_start = 0.0
        loss_end = 0.0
        steps_done = 0

        for ep in range(max(1, epochs)):
            ep_loss = 0.0
            batches = 0
            perm = torch.randperm(len(encoded)).tolist()
            for start in range(0, len(perm), max(1, batch_size)):
                idxs = perm[start : start + max(1, batch_size)]
                chunk = [encoded[i] for i in idxs]
                max_t = max(len(s) for s in chunk)
                batch = torch.zeros((len(chunk), max_t), dtype=torch.long, device=device)
                loss_weights = torch.zeros((len(chunk), max_t - 1), dtype=torch.float32, device=device)

                for i, s in enumerate(chunk):
                    batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                    if SubwordTokenizer.BOT_ID in s:
                        bot_pos = s.index(SubwordTokenizer.BOT_ID)
                        # Targets are shifted by 1 relative to input positions
                        if bot_pos > 0:
                            loss_weights[i, :bot_pos] = 0.12
                        loss_weights[i, bot_pos : len(s) - 1] = 1.0
                    else:
                        loss_weights[i, : len(s) - 1] = 1.0

                with self._amp_ctx(device):
                    logits = self.forward(batch[:, :-1])
                    targets = batch[:, 1:]
                    raw_ce = F.cross_entropy(
                        logits.float().reshape(-1, self.vocab_size),
                        targets.reshape(-1),
                        ignore_index=0,
                        reduction="none",
                    ).view(len(chunk), -1)
                    valid_mask = (targets != 0).float() * loss_weights
                    loss = (raw_ce * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.5)
                opt.step()
                ep_loss += float(loss.item())
                batches += 1
                steps_done += 1

            avg_loss = ep_loss / max(1, batches)
            if ep == 0:
                loss_start = avg_loss
            loss_end = avg_loss

        self.trained_steps += steps_done
        self.last_loss = float(loss_end)
        self.fluency_score = round(min(98.0, max(5.0, 100.0 * math.exp(-0.32 * max(0.0, loss_end - 0.35)))), 1)
        return (round(loss_start, 4), round(loss_end, 4))

    @torch.no_grad()
    def generate_reply(
        self,
        prompt_text: str,
        tokenizer: SubwordTokenizer,
        max_new_tokens: int = 90,
        temperature: float = 0.25,
        top_p: float = 0.88,
        repetition_penalty: float = 1.18,
    ) -> dict[str, Any]:
        device = self.tok_emb.weight.device
        self.eval()
        prompt_ids = tokenizer.encode(prompt_text, max_len=self.max_seq_len, add_bos=True, add_eos=False)
        max_prompt = max(16, self.max_seq_len - max_new_tokens)
        if len(prompt_ids) > max_prompt:
            prompt_ids = [SubwordTokenizer.BOS_ID] + prompt_ids[-(max_prompt - 1) :]

        cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        gen_ids: list[int] = []
        probs_chosen: list[float] = []

        for _ in range(max_new_tokens):
            if cur.shape[1] >= self.max_seq_len:
                break
            with self._amp_ctx(device):
                logits = self.forward(cur)[:, -1, :].float().squeeze(0)

            for cid in SubwordTokenizer._CONTROL_IDS:
                logits[cid] = -1e9
            if repetition_penalty > 1.0 and gen_ids:
                for prev_id in set(gen_ids[-24:]):
                    if logits[prev_id] > 0:
                        logits[prev_id] /= repetition_penalty
                    else:
                        logits[prev_id] *= repetition_penalty

            probs = F.softmax(logits / max(0.05, temperature), dim=-1)
            if temperature <= 0.05:
                next_id = int(torch.argmax(probs).item())
            else:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                mask = (cum_probs - sorted_probs) > top_p
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-9)
                picked_pos = int(torch.multinomial(sorted_probs, 1).item())
                next_id = int(sorted_idx[picked_pos].item())

            if next_id == SubwordTokenizer.EOS_ID:
                break
            probs_chosen.append(float(probs[next_id].item()))
            gen_ids.append(next_id)
            cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

        text = tokenizer.decode(gen_ids).strip()
        mean_prob = round(sum(probs_chosen) / max(1, len(probs_chosen)), 4) if probs_chosen else 0.0
        return {"text": text, "mean_prob": mean_prob, "num_tokens": len(gen_ids)}

    def export_state(self) -> dict[str, Any]:
        sd: dict[str, Any] = {k: v.cpu() for k, v in self.state_dict().items()}
        sd["__meta__"] = {
            "trained_steps": self.trained_steps,
            "last_loss": self.last_loss,
            "fluency_score": self.fluency_score,
        }
        return sd

    def import_state(self, state: dict[str, Any]) -> None:
        meta = state.get("__meta__")
        if isinstance(meta, dict):
            self.trained_steps = int(meta.get("trained_steps", 0))
            self.last_loss = float(meta.get("last_loss", 10.0))
            self.fluency_score = float(meta.get("fluency_score", 0.0))

        device = self.tok_emb.weight.device
        cur_sd = self.state_dict()
        # Only restore weights if hidden dimensions match current architecture
        saved_tok = state.get("tok_emb.weight")
        if saved_tok is None or saved_tok.ndim != 2 or saved_tok.shape[1] != self.d_model:
            return
        merged = {}
        for k, cur_v in cur_sd.items():
            if k in state and isinstance(state[k], torch.Tensor) and state[k].shape == cur_v.shape:
                merged[k] = state[k].to(device)
            elif (
                k in state
                and isinstance(state[k], torch.Tensor)
                and state[k].ndim == 2
                and cur_v.ndim == 2
                and state[k].shape[1] == cur_v.shape[1]
            ):
                buf = cur_v.clone()
                rows = min(buf.shape[0], state[k].shape[0])
                buf[:rows] = state[k][:rows].to(device)
                merged[k] = buf
            else:
                merged[k] = cur_v
        self.load_state_dict(merged)
        self.lm_head.weight = self.tok_emb.weight
