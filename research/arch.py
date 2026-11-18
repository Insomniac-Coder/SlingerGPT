"""Model variants for the phase-2 tests (T2 speed/memory at 500M, T3 quality at 124M).

  modern=False  today's GPT-2 block: LayerNorm, GELU MLP, learned positions, biases (as gpt128.py)
  modern=True   RMSNorm, SwiGLU, RoPE, QK-norm (RMSNorm on each query and key head), no biases
  engram=(2, 6) Engram lookup memory before those blocks (DeepSeek, arXiv 2601.07372; simplified, see Engram)
Precision: set_precision(model, "bf16" | "fp8" | "nvfp4"). Optimiser: optimizers(model, "adamw" | "muon", lr).
"""
import math
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# This Windows PyTorch build has no flash attention; cuDNN's is 1.6x faster than the default memory-efficient
# kernel on this GPU with identical results (attn_bench.py). The others stay as fallbacks, in this order.
ATTENTION = [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


@dataclass
class Cfg:
    layers: int = 12
    width: int = 768
    heads: int = 12
    ffn: int = 2048            # SwiGLU hidden size (modern); the GPT-2 block uses 4 x width
    vocab: int = 50304
    context: int = 1024
    rope_base: float = 10000.0  # rotary base; set_rope_base() changes it mid-run (the 492M's 16,384 phase uses 1,000,000)
    modern: bool = True
    rope: bool = None           # the four parts of `modern`, switchable one at a time (None = follow `modern`)
    qknorm: bool = None
    swiglu: bool = None
    rms: bool = None            # RMSNorm and no biases
    engram: tuple = ()          # block indices that get an Engram memory in front of them
    engram_slots: int = 49157   # rows per hash table (a prime); 2 orders x 8 hashes x 2 layers x 49k x 64 = 100M
    engram_hashes: int = 8
    engram_dim: int = 64
    pad: int = 50256            # token that stands in for "before the window" in the lookup memory (end-of-text)
    zero_init: bool = False     # each block's output matrices start at zero, so a fresh block changes nothing (speed-run)
    skips: bool = False         # every block also sees the raw token embedding and block 0's attention values (speed-run)
    untied: bool = False        # separate input and output tables instead of one shared table
    relu2: bool = False         # feed-forward = ReLU squared at 4 x width (speed-run); replaces GELU / SwiGLU
    ffn_scale: float = 1.0      # thinner (< 1) or wider feed-forward
    hourglass: int = 0          # k > 0: all blocks but the first and last work on one position per k tokens
    gate: float = 0.0           # > 0: only about this share of the tokens goes through the blocks between the first and last
    gate_mode: str = "probe"    # who picks them. "probe": a side-reader predicts each token's loss from its state after
                                #   block 0 and the predicted-hard ones pass; every token is judged on its own, so the
                                #   rule is the same in training and in use. "learned": a router trained end to end
                                #   takes the top share of each window (Mixture-of-Depths style; it sees the whole
                                #   window, so its held-out loss is slightly flattering). "random": a random share.
    gate_cap: float = 2.0       # one window may send at most this many times its budget through the middle blocks
    fp8: bool = False           # the blocks' matrix multiplies in 8-bit (torchao Float8Linear); weights stay 32-bit
    sample_grad: float = 0.0    # > 0: each weight matrix's gradient is estimated from this share of the token rows,
                                #   picked in proportion to their contribution and reweighted (unbiased)

    def __post_init__(self):
        for part in ("rope", "qknorm", "swiglu", "rms"):
            if getattr(self, part) is None:
                setattr(self, part, self.modern)


def rope(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class _SampledLinear(torch.autograd.Function):
    """A linear layer whose weight gradient is a reweighted sample of token rows. The signal passed to earlier
    layers is exact. Rows are drawn in proportion to |input row| x |gradient row| (the size of that row's
    contribution) and divided by their odds, so the estimate is right on average."""
    @staticmethod
    def forward(ctx, x, w, b, share):
        ctx.save_for_backward(x, w)
        ctx.share, ctx.has_bias = share, b is not None
        return F.linear(x, w, b)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gx = (g @ w.to(g.dtype)).to(x.dtype)
        x2, g2 = x.reshape(-1, x.shape[-1]).to(g.dtype), g.reshape(-1, g.shape[-1])
        size = x2.float().norm(dim=1) * g2.float().norm(dim=1) + 1e-20
        m = max(1, int(ctx.share * size.numel()))
        pick = torch.multinomial(size, m, replacement=True)
        scale = (size.sum() / (m * size[pick])).to(g.dtype)
        gw = ((g2[pick] * scale[:, None]).t() @ x2[pick]).to(w.dtype)
        return gx, gw, (g2.float().sum(0).to(w.dtype) if ctx.has_bias else None), None


class SampledLinear(nn.Linear):
    def __init__(self, i, o, bias, share):
        super().__init__(i, o, bias=bias)
        self.share = share

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight, self.bias)
        return _SampledLinear.apply(x, self.weight, self.bias, self.share)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        w, self.c, self.hd = c.width, c, c.width // c.heads
        Norm, bias = (nn.RMSNorm, False) if c.rms else (nn.LayerNorm, True)
        self.n1, self.n2 = Norm(w), Norm(w)
        Linear = (lambda i, o, bias: SampledLinear(i, o, bias, c.sample_grad)) if c.sample_grad else nn.Linear
        self.qkv = Linear(w, 3 * w, bias=bias)
        self.o = Linear(w, w, bias=bias)
        if c.qknorm:
            self.qn, self.kn = nn.RMSNorm(self.hd), nn.RMSNorm(self.hd)
        self.gated = c.swiglu and not c.relu2
        self.up = Linear(w, (2 if self.gated else 1) * c.ffn, bias=bias)   # SwiGLU: gate and up side by side
        self.down = Linear(c.ffn, w, bias=bias)
        if c.skips:
            self.a, self.b = nn.Parameter(torch.zeros(())), nn.Parameter(torch.tensor(0.5))

    def forward(self, x, cos, sin, x0=None, v0=None, mask=None):
        b, t, w = x.shape
        if x0 is not None:
            x = x + self.a * x0
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, -1, self.hd).transpose(1, 3).unbind(2)   # (b, heads, t, hd)
        if v0 is not None:
            v = v + self.b * (v0 - v)
        if self.c.qknorm:
            q, k = self.qn(q), self.kn(k)
        if self.c.rope:
            q, k = rope(q, cos, sin), rope(k, cos, sin)
        if isinstance(mask, torch.Tensor):   # the same mask written out in full (the CPU, which flex cannot train on)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif mask is not None:   # a flex-attention block mask: causal and within one document only
            a = flex_attention(q, k, v, block_mask=mask)
        else:
            with sdpa_kernel(ATTENTION, set_priority=True):
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, t, w))
        h = self.up(self.n2(x))
        if self.gated:
            h = F.silu(h[..., : h.shape[-1] // 2]) * h[..., h.shape[-1] // 2:]
        else:
            h = F.relu(h).square() if self.c.relu2 else F.gelu(h)
        x = x + self.down(h)
        return (x, v) if self.c.skips else x


class Engram(nn.Module):
    """Hashed 2- and 3-gram lookup memory, gated by the hidden state (Engram, arXiv 2601.07372, simplified:
    no vocabulary normalisation, one table per order x hash). Each position looks up the rows for the token
    pair and triple ending there; a gate sigmoid(norm(h) . norm(k) / sqrt(width)) decides how much to add."""
    def __init__(self, c, seed):
        super().__init__()
        self.c, n = c, 2 * c.engram_hashes
        self.table = nn.Embedding(n * c.engram_slots, c.engram_dim)
        nn.init.normal_(self.table.weight, std=0.02)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("mult", torch.randint(1, 2**31 - 1, (n, 3), generator=g), persistent=True)
        self.register_buffer("base", torch.arange(n) * c.engram_slots, persistent=False)
        self.k = nn.Linear(n * c.engram_dim, c.width, bias=False)
        self.v = nn.Linear(n * c.engram_dim, c.width, bias=False)
        self.qn, self.kn = nn.RMSNorm(c.width), nn.RMSNorm(c.width)
        self.conv = nn.Conv1d(c.width, c.width, 4, groups=c.width, padding=3, bias=False)
        nn.init.zeros_(self.conv.weight)   # starts as plain gated memory; the short convolution is learned

    def rows(self, idx):
        p1, p2 = F.pad(idx, (1, 0), value=self.c.pad)[:, :-1], F.pad(idx, (2, 0), value=self.c.pad)[:, :-2]
        grams = torch.stack((p2, p1, idx), -1)                       # (b, t, 3): the triple ending here
        h = self.c.engram_hashes
        tri = (grams.unsqueeze(2) * self.mult[h:]).sum(-1)           # 3-gram hashes
        bi = (grams[..., 1:].unsqueeze(2) * self.mult[:h, 1:]).sum(-1)   # 2-gram hashes
        return (torch.cat((bi, tri), 2) % 2147483647) % self.c.engram_slots + self.base   # (b, t, 2h)

    def forward(self, x, idx):
        e = self.table(self.rows(idx)).flatten(2)
        k, v = self.k(e), self.v(e)
        gate = torch.sigmoid((self.qn(x) * self.kn(k)).sum(-1, keepdim=True) / math.sqrt(x.shape[-1]))
        out = gate * v
        out = out + self.conv(out.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return x + out


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c, self.recompute = c, False
        self.tokens = nn.Embedding(c.vocab, c.width)
        self.places = None if c.rope else nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.engram = nn.ModuleDict({str(i): Engram(c, seed=i) for i in c.engram})
        self.norm = (nn.RMSNorm if c.rms else nn.LayerNorm)(c.width)
        self.out = nn.Linear(c.width, c.vocab, bias=not c.rms)
        if not c.untied:
            self.out.weight = self.tokens.weight   # tied
        for name, p in self.named_parameters():   # GPT-2 starting values; residual outputs smaller
            if name.startswith("engram") or p.dim() < 2:
                if name.endswith("bias"):
                    nn.init.zeros_(p)
                continue
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.layers) if name.endswith(("o.weight", "down.weight")) else 0.02)
        if c.zero_init:
            for block in self.blocks:
                nn.init.zeros_(block.o.weight), nn.init.zeros_(block.down.weight)
        thin = c.hourglass or c.gate   # the middle blocks see fewer positions: no token-keyed memory or skips in there
        assert not thin or (not c.skips and set(c.engram) <= {0, 1, c.layers - 1} and not (c.hourglass and c.gate))
        assert not c.hourglass or c.context % c.hourglass == 0
        self.scores = self.share = None
        self.trained_padded, self.trained_steps = 0.0, 0
        if c.gate:
            self.probe = nn.Linear(c.width, 1)   # the side-reader (or, for "learned", the router)
            nn.init.zeros_(self.probe.weight) if c.gate_mode == "probe" else nn.init.normal_(self.probe.weight, std=0.02)
            nn.init.constant_(self.probe.bias, 0.0 if c.gate_mode == "probe" else 1.0)
            self.register_buffer("cut", torch.tensor(float("nan")))   # the predicted loss above which a token passes
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self.set_rope_base(c.rope_base)

    def set_rope_base(self, base):
        """(Re)builds the rotary tables for `base`, on the model's own device. Raising the base mid-run (Qwen-style ABF)
        stretches the slowest angles so that longer windows stay readable."""
        self.c.rope_base = float(base)
        hd = self.c.width // self.c.heads
        device = self.tokens.weight.device
        freq = 1.0 / float(base) ** (torch.arange(0, hd, 2, device=device if device.type != "meta" else "cpu") / hd)
        ang = torch.outer(torch.arange(self.c.context, device=freq.device), freq)
        self.cos, self.sin = ang.cos().to(device), ang.sin().to(device)

    def forward(self, idx):
        with torch.autocast(idx.device.type, dtype=torch.bfloat16):
            return self.out(self.features(idx))

    def features(self, idx, patch=1, mask=None):
        """The final normalised hidden state (BF16), before the output layer; training feeds it to fused_ce.
        patch > 1 (patch-level training, arXiv 2407.12665): every `patch` tokens are averaged into one position,
        so the result has one row per patch; the lookup memory is skipped, as it is keyed on single tokens.
        mask: a flex-attention block mask (document_mask below), so no position attends across a document boundary."""
        b, t = idx.shape
        k, last = self.c.hourglass, len(self.blocks) - 1
        assert mask is None or not (k or self.c.gate or self.c.skips or patch > 1), "the document mask needs the plain path"
        with torch.autocast(idx.device.type, dtype=torch.bfloat16):
            x = self.tokens(idx)
            if patch > 1:
                t //= patch
                x = x.view(b, t, patch, -1).mean(2)
            if self.places is not None:
                x = x + self.places(torch.arange(t, device=idx.device))
            cos, sin = self.cos[:t].to(x.dtype), self.sin[:t].to(x.dtype)
            x0 = x if self.c.skips else None
            v0 = None
            for i, block in enumerate(self.blocks):
                if str(i) in self.engram and patch == 1:
                    x = self.engram[str(i)](x, idx)
                if self.c.gate and 0 < i < last:   # the blocks between the first and last run on the chosen tokens only
                    if i == 1:
                        x = self.middle_on_chosen(x, cos, sin)
                    continue
                if k and i == 1:   # squeeze: the middle blocks see one position per k tokens (their average)
                    full, x, cos, sin = x, x.view(b, t // k, k, -1).mean(2), cos[: t // k], sin[: t // k]
                if k and i == last:   # widen again, shifted by k so that no token receives anything from its own future
                    x = full + F.pad(x.repeat_interleave(k, 1), (0, 0, k, 0))[:, :t]
                    cos, sin = self.cos[:t].to(x.dtype), self.sin[:t].to(x.dtype)
                if self.c.skips:
                    x, v = block(x, cos, sin, x0, v0)
                    v0 = v if v0 is None else v0
                elif self.recompute and self.training:   # store less for the backward pass, recompute it there
                    x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
                else:
                    x = block(x, cos, sin, mask=mask)
            return self.norm(x).to(torch.bfloat16)


def document_mask(idx, eot=0, dense=None, **kw):
    """A flex-attention block mask for windows that hold several documents: causal, and each position sees only
    its own document. A document starts after each end-of-text piece (`eot`); the end-of-text piece itself still
    belongs to the document it closes. dense (default: on the CPU): the same mask as a full (b, 1, t, t) table."""
    b, t = idx.shape
    doc = torch.cumsum((idx == eot).int(), dim=1) - (idx == eot).int()   # end-of-text closes its own document
    if dense or (dense is None and idx.device.type == "cpu"):
        causal = torch.ones(t, t, dtype=torch.bool, device=idx.device).tril()
        return (causal & (doc[:, :, None] == doc[:, None, :]))[:, None]

    def same_doc_causal(bi, h, q, kv):
        return (q >= kv) & (doc[bi, q] == doc[bi, kv])
    return create_block_mask(same_doc_causal, B=b, H=None, Q_LEN=t, KV_LEN=t, device=idx.device, **kw)


def middle_on_chosen(self, x, cos, sin):
    """Runs the blocks between the first and last on the chosen tokens only; the others keep their state.
    Chosen tokens are gathered to the front in reading order and padded to a common length. Padding sits at the
    end, where causal attention cannot see it, and is thrown away afterwards. Lengths are rounded up to a
    multiple of 16, because the attention kernel plans afresh (tens of ms) for every new length."""
    b, t, w = x.shape
    c = self.c
    want = max(1, round(c.gate * t))
    with torch.autocast("cuda", enabled=False):   # 32-bit, so that scores rarely tie
        score = self.probe((x.detach() if c.gate_mode == "probe" else x).float()).squeeze(-1)   # (b, t)
    if c.gate_mode == "learned":   # exactly the top share of each window
        order = score.detach().topk(want, dim=1).indices.sort(dim=1).values
        count, longest = torch.full((b,), want, device=x.device), want
    else:
        if c.gate_mode == "probe":
            self.scores = score
            if self.training:   # the cut follows the score that the hardest `gate` share of recent tokens exceeded
                now = torch.quantile(score.detach().flatten()[:: max(1, score.numel() // 8192)], 1 - c.gate)
                self.cut.copy_(now if torch.isnan(self.cut) else 0.9 * self.cut + 0.1 * now)
            chosen = score.detach() >= self.cut
        else:
            chosen = torch.rand(b, t, device=x.device) < c.gate
        cap = min(t, max(want, round(c.gate_cap * want)))   # a window full of hard text pays at most gate_cap x the budget: its first `cap` hard tokens pass
        count = chosen.sum(1).clamp(max=cap)
        longest = min(cap, -(-int(count.max()) // 16) * 16)
        order = chosen.float().argsort(dim=1, descending=True, stable=True)[:, :longest]   # chosen first, in reading order
    self.share, self.padded = count.float().mean() / t, longest / t
    if self.training:   # what the middle blocks really worked on over the whole run, padding included
        self.trained_padded, self.trained_steps = self.trained_padded + self.padded, self.trained_steps + 1
    if longest == 0:
        return x
    real = torch.arange(longest, device=x.device) < count[:, None]
    at = order[..., None].expand(-1, -1, w)
    start = xs = x.gather(1, at)
    cs, sn = cos[order][:, None], sin[order][:, None]   # each chosen token keeps the rotary angle of its own position
    weight = score.gather(1, order)[..., None].to(xs.dtype) if c.gate_mode == "learned" else None
    for block in self.blocks[1:-1]:
        ys = block(xs, cs, sn)
        xs = ys if weight is None else xs + weight * (ys - xs)
    return x.scatter(1, at, torch.where(real[..., None], xs, start))


def side(self, per_token_loss):
    """Teaches the side-reader to predict each token's loss, which training has just computed anyway. Kept
    small so that it does not crowd the real gradient out of the shared gradient clip."""
    if not (self.c.gate and self.c.gate_mode == "probe") or self.scores is None:
        return 0.0
    return 0.01 * F.mse_loss(self.scores, per_token_loss.float())


def usage(self):
    if self.share is None:
        return {}
    return {"share of tokens through the middle blocks": round(float(self.share), 3), "padded to": round(self.padded, 3),
            "padded share averaged over training": round(self.trained_padded / max(1, self.trained_steps), 3)}


GPT.middle_on_chosen, GPT.side, GPT.usage = middle_on_chosen, side, usage


def set_precision(model, kind, fp4_skip_last=2):
    """fp8: torchao Float8 on every block linear. nvfp4: NVFP4 on the block linears except the last
    `fp4_skip_last` blocks (NVIDIA's recipe keeps the final blocks in higher precision), FP8 on those.
    The output head, embeddings, norms, attention maths and Engram stay BF16."""
    if kind == "bf16":
        return model
    if kind in ("nvfp4", "nvfp4op"):   # nvfp4: FP4 layer outside torch.compile; nvfp4op: sealed op inside it
        from fp4_sm120 import NVFP4Linear, NVFP4LinearOp, refresh_fp4_weights
        layer = NVFP4LinearOp if kind == "nvfp4op" else NVFP4Linear
        for i, block in enumerate(model.blocks[: len(model.blocks) - fp4_skip_last]):
            for name in ("qkv", "o", "up", "down"):
                setattr(block, name, layer.from_linear(getattr(block, name)))
        refresh_fp4_weights(model)
    from torchao.float8 import convert_to_float8_training
    convert_to_float8_training(model, module_filter_fn=lambda m, name: type(m) is nn.Linear and name.startswith("blocks"))
    return model


def optimizers(model, kind, lr, weight_decay=0.1):
    """AdamW everywhere, or Muon for the block matrices (DeepSeek-V4 / Moonshot practice) with AdamW for the
    embeddings, norms and biases. Engram tables: Adam at 5x the learning rate, no weight decay (Engram paper)."""
    named = list(model.named_parameters())
    tables = [p for n, p in named if n.endswith("table.weight")]
    matrices = [p for n, p in named if p.dim() == 2 and n.startswith(("blocks", "engram")) and not n.endswith("table.weight")]
    flat = [p for n, p in named if p.dim() != 2 and n.startswith(("blocks", "engram", "norm"))] + \
           [p for n, p in named if n.startswith(("out.bias", "places", "probe"))]
    embed = [p for n, p in named if n in ("tokens.weight", "out.weight")]
    groups = [{"params": embed, "weight_decay": weight_decay}, {"params": flat, "weight_decay": 0.0}]
    if tables:
        groups.append({"params": tables, "weight_decay": 0.0, "lr": 5 * lr})
    opts = []
    if kind == "muon":
        opts.append(torch.optim.Muon(matrices, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"))
    else:
        groups.append({"params": matrices, "weight_decay": weight_decay})
    opts.append(torch.optim.AdamW([{"lr": lr, **g} for g in groups], lr=lr, betas=(0.9, 0.95), fused=True))
    assert sum(len(g["params"]) for o in opts for g in o.param_groups) == len(named), "a parameter has no optimiser"
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    return opts


CONFIGS = {
    "124m": dict(layers=12, width=768, heads=12, ffn=2048),
    "500m": dict(layers=22, width=1280, heads=20, ffn=3456),   # deep-narrow; every size a multiple of 128 (FP4)
    "proxy": dict(layers=6, width=384, heads=6, ffn=1024),     # ~30M, for many short comparison runs (lab.py)
    "10m": dict(layers=6, width=384, heads=6, ffn=1024, vocab=256, pad=0,  # ~10.7M, byte by byte (lab.py)
                engram_slots=4093, engram_hashes=4, engram_dim=32),
    "30m": dict(layers=6, width=640, heads=10, ffn=1704, vocab=256, pad=0,  # ~29.5M in the body: the 10m body made wider,
                engram_slots=4093, engram_hashes=4, engram_dim=32),         # not deeper, so 128 x 256 still fits a 12 GB card
    "1m": dict(layers=4, width=128, heads=4, ffn=344, vocab=256, pad=0,   # ~0.8M, reads text byte by byte (lab.py)
               engram_slots=1021, engram_hashes=2, engram_dim=16),
}


def build(size, device="cuda", **kw):
    c = Cfg(**{**CONFIGS[size], **kw})   # kw may override the size's own numbers (e.g. layers)
    if not c.swiglu or c.relu2:
        c.ffn = 4 * c.width
    c.ffn = int(c.ffn * c.ffn_scale)
    model = GPT(c).to(device)
    if c.fp8:   # only the block layers: the piece table, the output layer and the lookup memory stay in 16-bit
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(model, module_filter_fn=lambda m, name: name.startswith("blocks.") and isinstance(m, nn.Linear))
    return model


def check_sampled_gradient():
    """The sampled weight gradient must equal the exact one on average."""
    torch.manual_seed(0)
    exact, sampled = nn.Linear(24, 16, bias=False).cuda(), SampledLinear(24, 16, False, 0.25).cuda()
    sampled.weight.data.copy_(exact.weight.data)
    x = torch.randn(4096, 24, device="cuda") * torch.rand(4096, 1, device="cuda") ** 4   # a few rows dominate
    target = torch.randn(4096, 16, device="cuda")
    (exact(x) - target).square().sum().backward()
    total = torch.zeros_like(exact.weight)
    for _ in range(400):
        sampled.weight.grad = None
        (sampled(x) - target).square().sum().backward()
        total += sampled.weight.grad
    error = ((total / 400 - exact.weight.grad).norm() / exact.weight.grad.norm()).item()
    assert error < 0.03, error
    print(f"sampled gradient: average of 400 draws is within {error:.1%} of the exact gradient")


if __name__ == "__main__":
    check_sampled_gradient()
    for size, extra in [("124m", {}), ("124m", {"modern": False}), ("124m", {"engram": (2, 6)}), ("500m", {})]:
        m = build(size, **extra)
        n = sum(p.numel() for p in m.parameters())
        x = torch.randint(0, 50257, (2, 256), device="cuda")
        y = m(x)
        assert y.shape == (2, 256, 50304) and torch.isfinite(y).all()
        print(f"{size} {extra}: {n / 1e6:.1f}M parameters (embeddings {m.tokens.weight.numel() / 1e6:.1f}M), forward OK")
        for kind in ("adamw", "muon"):
            optimizers(m, kind, 6e-4)
        del m
