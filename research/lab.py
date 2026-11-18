"""Short training runs that compare model variants at equal tokens, on a small model.

    python -P research/lab.py lr --size 1m                 learning-rate pilot: both optimisers x rates x old/new block
    python -P research/lab.py stage1 --size 1m             every combination of the six FACTORS (64 runs)
    python -P research/lab.py stage2 --size 1m --base muon,rope      every combination of the four EXTRAS on a base
    python -P research/lab.py custom NAME --size 1m --configs JSON   any list of variants, e.g. '[{"muon":true,"gate":0.25}]'
    python -P research/lab.py show NAME@1m                 results, and what each change is worth on average

Sizes (SETUPS): "1m" = 4 layers, 128 wide, reads data/corpus/mix_v2.txt byte by byte in 256-byte windows;
"10m" = 6 layers, 384 wide, the same text; "10m-bpe1024", "10m-bpe4096", "10m-bpe16384", "10m-gpt2" = that same
model reading the same text as word pieces (data/pieces/, made by prepare_pieces.py);
"proxy" = 6 layers, 384 wide, 512 word pieces from data/tokens128 (~30M parameters, mostly vocabulary).

Every run sees the same batches in the same order and is scored on the same held-out batches, so two runs differ
only in the model. One line per finished run goes to research/logs/lab.jsonl; a restart skips finished runs.
Each line has the held-out loss at every quarter, seconds trained, and the arithmetic of one forward pass per
token (counted, so it does not depend on this machine). Speed is measured without torch.compile; on the 1M
model the stopwatch mostly measures Python overhead, so trust the counted arithmetic more than the seconds.

Besides the switches of arch.Cfg (gate, hourglass, sample_grad, ...), a variant may set:
  head="shortlist"     output layer scores the 8,192 most common word pieces plus "something rarer", and only then the
                       rare ones (adaptive softmax); word-piece sizes only; needs logs/token_rank.npy (where_time_goes.py)
  head="sampled"       while training, each position scores the right piece and a shared random sample of the others
                       (samples=4096 by default) instead of the whole vocabulary; common pieces are drawn more often
                       and the scores are corrected for that. Held-out scoring always uses the whole vocabulary.
  patch=[k, share]     the first `share` of the tokens are trained k tokens to a position (patch-level training)
  short=[ctx, share]   the first `share` of the tokens come in windows of `ctx` tokens (more of them per step)
  grow=share           train half the layers, then stack them twice at `share` of the run
  save="name"          keep the trained weights as models/<name>.pt (ledger.py reads models/small_thinker.pt)
  seed=n               a different starting point and batch order (default 0)
  compiled=true        torch.compile the model body, so the many small operations are fused; compiling happens
                       before the stopwatch starts. Not for gate, patch or grow.
"""
import argparse, gc, itertools, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import arch
from fused_ce import fused_linear_ce

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():   # a run that needs more memory than the card has free stops with an error here;
    _free, _total = torch.cuda.mem_get_info()   # without this Windows spills into system memory, and the PC can freeze
    torch.cuda.set_per_process_memory_fraction(max(0.1, (_free - 0.5 * 2**30) / _total))
ROOT = HERE.parent
LOG = HERE / "logs" / "lab.jsonl"
SETUPS = {   # window, sequences per step, held-out batches, default run length, rates tried by the pilot, lookup-memory blocks
    "1m": dict(ctx=256, batch=128, val_batches=16, tokens=32_768_000, rates=(1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2), engram=(1,)),
    "10m": dict(ctx=256, batch=128, val_batches=16, tokens=32_768_000, rates=(1e-3, 2e-3, 4e-3, 8e-3), engram=(1,)),
    "proxy": dict(ctx=512, batch=32, val_batches=32, tokens=16_384_000, rates=(6e-4, 1.2e-3, 2.4e-3), engram=(1, 3)),
}
for _name, _vocab in (("bpe1024", 1024), ("bpe4096", 4096), ("bpe16384", 16384), ("gpt2", 50304), ("phrase65k", 65536)):
    # the 10m model body reading word pieces instead of letters (the files come from prepare_pieces.py)
    SETUPS[f"10m-{_name}"] = dict(SETUPS["10m"], body="10m", data=f"data/pieces/{_name}", cfg={"vocab": _vocab})
for _name, _vocab in (("bpe4096", 4096), ("gpt2", 50304)):   # the same text on the three-times-bigger 30m body
    SETUPS[f"30m-{_name}"] = dict(SETUPS["10m"], body="30m", data=f"data/pieces/{_name}", cfg={"vocab": _vocab})
for _name, _vocab in (("bpe4096", 4096), ("bpe16384", 16384), ("gpt2", 50304), ("phrase65k", 65536)):
    # the 124M body as gpt128.py trains it: 1,024-piece windows, 16 per slice, 4 slices per weight update
    SETUPS[f"124m-{_name}"] = dict(ctx=1024, batch=16, accum=4, val_batches=16, tokens=0, rates=(), engram=(1,), body="124m",
                                   data=f"data/pieces/{_name}", cfg={"vocab": _vocab, "pad": 0, "engram_slots": 16381,
                                                                     "engram_hashes": 4, "engram_dim": 64})
    # the same, on 990 MB of the text instead of 297 MB (prepare_more_text.py), for runs long enough to learn facts
    SETUPS[f"124m-{_name}-1g"] = dict(SETUPS[f"124m-{_name}"], data=f"data/pieces1g/{_name}")
FACTORS = ("muon", "rope", "qknorm", "swiglu", "rms", "engram")
EXTRAS = ("zero_init", "skips", "untied", "relu2")
LR = {"10m": {False: 2e-3, True: 4e-3},          # placeholders until the 10m rate check; its runs pass lr explicitly
      "1m": {False: 8e-3, True: 1.6e-2},         # AdamW, Muon: best average over both blocks in the `lr` pilot (2026-11-13);
      "proxy": {False: 1.2e-3, True: 1.2e-3}}    # 1m Muon's best is the highest rate tried, proxy Muon's is not found yet
MB = 1 << 20


class Windows:
    """Random windows of ctx + 1 tokens, in pinned memory for a fast copy to the card.
    proxy: word pieces from data/tokens128/train.bin or val.bin. 1m: bytes straight from the text file, where
    every 100th megabyte is held out for scoring and never trained on."""
    def __init__(self, size, part, count, ctx, seed):
        self.count, self.ctx, self.rng = count, ctx, np.random.default_rng(seed)
        source = SETUPS[size].get("data", "data/tokens128" if size == "proxy" else "bytes")
        self.bytes = source == "bytes"
        if self.bytes:
            self.data = np.memmap(ROOT / "data/corpus/mix_v2.txt", dtype=np.uint8, mode="r")
            chunks = np.arange(len(self.data) // MB)
            self.chunks = chunks[(chunks % 100 == 99) == (part == "val")]
        else:
            self.data = np.memmap(ROOT / source / f"{part}.bin", dtype=np.uint16, mode="r")

    def next(self):
        if self.bytes:   # a whole window inside one megabyte, so training never touches a held-out megabyte
            at = self.rng.choice(self.chunks, self.count) * MB + self.rng.integers(0, MB - self.ctx - 1, self.count)
        else:
            at = self.rng.integers(0, len(self.data) - self.ctx - 1, self.count)
        chunk = torch.from_numpy(self.data[at[:, None] + np.arange(self.ctx + 1)].astype(np.int64)).pin_memory()
        chunk = chunk.cuda(non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]


def val_set(size):
    s = SETUPS[size]
    held_out = Windows(size, "val", s["batch"] // 2, s["ctx"], seed=7)
    return [held_out.next() for _ in range(s["val_batches"])]


def stacked(small, cfg, big_layers):
    """A model with twice the layers: block i starts as a copy of the small model's block i mod half."""
    big, half = arch.build(**{**cfg, "layers": big_layers}), big_layers // 2
    old = small.state_dict()
    name_in_small = lambda n: f"blocks.{int(n.split('.')[1]) % half}.{n.split('.', 2)[2]}" if n.startswith("blocks.") else n
    big.load_state_dict({n: old[name_in_small(n)] for n in big.state_dict()})
    return big


def run(val, tokens, size="proxy", lr=None, muon=False, engram=False, head="full", patch=None, short=None, grow=None,
        save=None, seed=0, compiled=False, samples=4096, stop_at=None, **parts):
    """Train one variant for `tokens` tokens; held-out loss (and seconds trained so far) at every 5% of the way.
    stop_at: check every 2% instead, and stop (saving the model, if `save`) the first time the held-out loss per
    unit is at or below this value."""
    s = SETUPS[size]
    CTX, BATCH, ACCUM = s["ctx"], s["batch"], s.get("accum", 1)   # ACCUM slices of BATCH x CTX per weight update
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    body_size = s.get("body", size)   # which arch.CONFIGS entry; differs from `size` when only the data changes
    lr = lr or LR[body_size][bool(muon)]
    layers = arch.CONFIGS[body_size]["layers"]
    assert not (grow and engram), "growing is tested without the lookup memory"
    cfg = dict(size=body_size, context=CTX, modern=False, engram=s["engram"] if engram else (), **s.get("cfg", {}), **parts)
    model = arch.build(**{**cfg, "layers": layers // 2}) if grow else arch.build(**cfg)
    small_vocab = model.c.vocab <= 8192
    optimisers = lambda m: arch.optimizers(m, "muon" if muon else "adamw", lr)
    opts, extra = optimisers(model), []
    if head == "shortlist":
        if "data" in s:   # word pieces of this run's own text: order them by how often they occur in it
            counts = np.bincount(np.memmap(ROOT / s["data"] / "train.bin", dtype=np.uint16, mode="r"), minlength=model.c.vocab)
            order = np.empty(model.c.vocab, dtype=np.int64)
            order[np.argsort(-counts, kind="stable")] = np.arange(model.c.vocab)
            rank = torch.from_numpy(order).cuda()
        else:
            rank = torch.from_numpy(np.load(HERE / "logs" / "token_rank.npy")).cuda()
        short_head = nn.AdaptiveLogSoftmaxWithLoss(model.c.width, model.c.vocab, [8192], div_value=1.0).cuda()
        extra = [torch.optim.AdamW(short_head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)]
        extra[0].param_groups[0]["base_lr"] = lr

    if head == "sampled":   # how often each piece occurs: common pieces are drawn more often, and corrected for
        seen = np.bincount(np.memmap(ROOT / s["data"] / "train.bin", dtype=np.uint16, mode="r"), minlength=model.c.vocab)
        freq = torch.from_numpy(seen.astype(np.float32) + 1.0).cuda()
        freq = freq / freq.sum()
        log_expected = torch.log(samples * freq)   # log of how many times each piece is expected among the draws

    def log_probs(m, h):   # (rows, vocab) log-probabilities and the column of each token id
        if head == "shortlist":
            return short_head.log_prob(h.float()), rank
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.log_softmax(m.out(h).float(), -1), None

    def loss_of(m, x, y, k=1):
        if k > 1:   # patch-level: one guess per patch, scored on all k tokens of the next patch
            logp, col = log_probs(m, m.features(x, patch=k)[:, :-1].flatten(0, 1))
            nxt = x.view(x.shape[0], -1, k)[:, 1:].flatten(0, 1)
            return -logp.gather(1, nxt if col is None else col[nxt]).mean()
        h = (body(x) if m.training else m.features(x)).flatten(0, 1)
        if head == "shortlist":
            per = -short_head(h.float(), rank[y.flatten()]).output
        elif head == "sampled":
            want, w = y.flatten(), m.out.weight
            draw = torch.multinomial(freq, samples, replacement=True)   # one shared set of draws for the whole batch
            with torch.autocast("cuda", dtype=torch.bfloat16):
                own = (h * w[want]).sum(-1).float()                     # score of the right piece, per position
                rest = (h @ w[draw].t()).float()                        # scores of the drawn pieces, (positions, samples)
            if m.out.bias is not None:
                own, rest = own + m.out.bias[want], rest + m.out.bias[draw]
            rest = (rest - log_expected[draw]).masked_fill(want[:, None] == draw[None, :], float("-inf"))
            per = torch.logsumexp(torch.cat((own[:, None], rest), 1), 1) - own
        elif small_vocab:   # a 256-entry output layer: the plain loss is cheaper than slicing it
            with torch.autocast("cuda", dtype=torch.bfloat16):
                per = F.cross_entropy(m.out(h).float(), y.flatten(), reduction="none")
        else:
            assert m.scores is None, "the side-reader needs each token's loss, which the sliced loss does not return"
            return fused_linear_ce(h, m.out.weight, y.flatten(), m.out.bias)
        return per.mean() + m.side(per.detach().view_as(y))

    @torch.no_grad()
    def unseen_loss(m):
        m.eval()
        total, count = 0.0, 0
        for x, y in val:
            h, want = m.features(x).flatten(0, 1), y.flatten()
            for at in range(0, len(h), 4096):   # in slices: all positions x a 50k vocabulary at once would be 8 GB
                logp, col = log_probs(m, h[at:at + 4096])
                right = want[at:at + 4096]
                total += -logp.gather(1, (right if col is None else col[right])[:, None]).sum().item()
            count += len(h)
        m.train()
        return total / count

    data = Windows(size, "train", BATCH, CTX, seed=2024 + seed)
    steps = tokens // (BATCH * CTX * ACCUM)
    checks = 50 if stop_at else 20
    warm, marks = max(1, steps // 10), {round(steps * f / checks) for f in range(1, checks + 1)}
    body = model.features
    if compiled:   # one throwaway pass at the training shape, so that compiling is not on the stopwatch
        assert not (grow or patch or model.c.gate), "compiled runs are for the plain training path"
        body = torch.compile(model.features)
        loss_of(model, *Windows(size, "train", BATCH, CTX, seed=1).next()).backward()
        for o in opts:
            o.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    curve, spent, began = [], 0.0, time.time()
    for step in range(steps):
        share = step / steps
        if grow and step == round(steps * grow):
            model = stacked(model, cfg, layers)
            opts, body = optimisers(model), model.features
        factor = min(1.0, (step + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * share)))
        for o in opts + extra:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * factor
        data.ctx = short[0] if short and share < short[1] else CTX
        data.count = BATCH * CTX // data.ctx
        for _ in range(ACCUM):
            x, y = data.next()
            loss = loss_of(model, x, y, patch[0] if patch and share < patch[1] else 1) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(itertools.chain(model.parameters(), short_head.parameters() if extra else ()), 1.0)
        for o in opts + extra:
            o.step()
            o.zero_grad(set_to_none=True)
        if step + 1 in marks:
            if not math.isfinite(loss.item()):   # .item() also waits for the card, so the timing is honest
                raise FloatingPointError(f"loss is {loss.item()} at step {step + 1}")
            spent += time.time() - began
            curve.append([(step + 1) * BATCH * CTX * ACCUM, round(unseen_loss(model), 4), round(spent, 1)])
            if stop_at and curve[-1][1] <= stop_at:
                break
            began = time.time()
    if save:
        torch.save({"cfg": cfg, "model": model.state_dict()}, ROOT / "models" / f"{save}.pt")
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:   # the arithmetic of one forward pass, in use
        model(val[0][0][:4])
    count = sum(p.numel() for p in model.parameters()) + (sum(p.numel() for p in short_head.parameters()) if extra else 0)
    return {"parameters_M": round(count / 1e6, 3), "curve": curve, "final": curve[-1][1], "seconds": round(spent, 1),
            "tokens_trained": curve[-1][0], "tokens_per_s": round(curve[-1][0] / spent), "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "kflop_per_token": round(counter.get_total_flops() / (4 * CTX) / 1e3, 1), "used": model.usage()}


def label(cfg):
    return "+".join(k if v is True else f"{k}={v}" for k, v in cfg.items() if v) or "base"


def stage(name, size, configs, tokens):
    tokens = tokens or SETUPS[size]["tokens"]
    tag = name if size == "proxy" else f"{name}@{size}"
    done = {json.loads(line)["key"] for line in LOG.open(encoding="utf-8")} if LOG.exists() else set()
    val = val_set(size)
    for i, cfg in enumerate(configs):
        key = f"{tag}:{label(cfg)}:{tokens}"
        if key in done:
            continue
        try:
            r = run(val, tokens, size=size, **cfg)
        except Exception as e:   # one broken variant must not stop the others
            r = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        r.update(key=key, stage=tag, config=cfg, tokens=tokens)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(f"[{i + 1}/{len(configs)}] {label(cfg):60} " + (f"loss {r['final']:.4f}  {r['seconds']:6.1f} s  "
              f"{r['kflop_per_token']:8.1f} kflop/token  {r['used'] or ''}" if "final" in r else r["error"]), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


def show(name):
    runs = [r for r in map(json.loads, LOG.open(encoding="utf-8")) if r["stage"] == name]
    for r in runs:
        if "error" in r:
            print(f"  FAILED {label(r['config'])}: {r['error']}")
    runs = sorted((r for r in runs if "final" in r), key=lambda r: r["final"])
    print(f"{len(runs)} runs of {name}, best first: held-out loss (at each quarter), seconds trained, forward arithmetic per token, parameters")
    for r in runs:
        quarters = [c for c in r["curve"][:-1] if round(c[0] / r["tokens"] * 20) in (5, 10, 15) or len(r["curve"]) <= 4]
        print(f"  {r['final']:.4f}  ({' '.join(f'{c[1]:.3f}' for c in quarters)})  {r.get('seconds', 0):6.1f} s  "
              f"{r.get('kflop_per_token', 0):8.1f} kflop  {r['parameters_M']:>7}M  {label(r['config'])}")
    keys = sorted({k for r in runs for k, v in r["config"].items() if isinstance(v, bool)})
    keys = [k for k in keys if len({bool(r["config"].get(k)) for r in runs}) == 2]
    mean = lambda rs, field="final": sum(r.get(field, 0) for r in rs) / len(rs)
    if keys:
        print("\naverage effect of switching each change on (loss: negative = better; seconds and arithmetic: x1 = no change):")
    for k in keys:
        on, off = [r for r in runs if r["config"].get(k)], [r for r in runs if not r["config"].get(k)]
        print(f"  {k:10} loss {mean(on) - mean(off):+.4f}   seconds x{mean(on, 'seconds') / max(mean(off, 'seconds'), 1e-9):.2f}"
              f"   arithmetic x{mean(on, 'kflop_per_token') / max(mean(off, 'kflop_per_token'), 1e-9):.2f}")
    if len(keys) > 1:
        print("\npairs that do more (or less) together than apart (negative = they help each other):")
    for a, b in itertools.combinations(keys, 2):
        cell = lambda va, vb: [r for r in runs if bool(r["config"].get(a)) == va and bool(r["config"].get(b)) == vb]
        if all(cell(va, vb) for va in (0, 1) for vb in (0, 1)):
            inter = mean(cell(1, 1)) - mean(cell(1, 0)) - mean(cell(0, 1)) + mean(cell(0, 0))
            if abs(inter) >= 0.005:
                print(f"  {a} x {b}: {inter:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["lr", "stage1", "stage2", "custom", "show"])
    p.add_argument("name", nargs="?", default="stage1")
    p.add_argument("--size", default="proxy", choices=list(SETUPS))
    p.add_argument("--tokens", type=int, default=0, help="tokens per run (default: the size's own)")
    p.add_argument("--base", default="", help="comma-separated switches every variant of the stage starts from")
    p.add_argument("--configs", default="[]", help="for custom: a JSON list of variants, or @path to a file holding one")
    a = p.parse_args()
    base = {k: True for k in a.base.split(",") if k}
    grid = lambda names: [dict(zip(names, on)) for on in itertools.product((False, True), repeat=len(names))]
    if a.what == "show":
        show(a.name)
    elif a.what == "lr":
        new = dict(rope=True, qknorm=True, swiglu=True, rms=True)
        stage("lr", a.size, [dict(muon=m, lr=lr, **block) for block in ({}, new) for m in (False, True)
                             for lr in SETUPS[a.size]["rates"]], a.tokens)
    elif a.what == "stage1":
        stage("stage1", a.size, grid(FACTORS), a.tokens)
    elif a.what == "stage2":
        stage("stage2", a.size, [{**base, **extra} for extra in grid(EXTRAS)], a.tokens)
    else:
        text = Path(a.configs[1:]).read_text(encoding="utf-8") if a.configs.startswith("@") else a.configs   # @file.json
        stage(a.name, a.size, [{**base, **cfg} for cfg in json.loads(text)], a.tokens)
