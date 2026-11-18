"""How fast the 124M model reads text, for the normal way and versions of the new way. A stopwatch, not training.

    python -P research/speed_124m.py VARIANT          one variant per process; results appended to logs/speed_124m.jsonl
    python -P research/speed_124m.py report           the table

Every variant uses the same harness as gpt128.py: 16 sequences of 1,024 pieces per slice, 4 slices added up
before each weight update (65,536 pieces), the model body compiled, real pieces from data/pieces/<list>/train.bin.
The full scoring step is done in slices (research/fused_ce.py) for the long lists, so the normal way fits too.
Speed is counted in letters per second: pieces per second x letters per piece of that list.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import arch
from fused_ce import fused_linear_ce

LOG = HERE / "logs" / "speed_124m.jsonl"
MEMORY = dict(engram=(1,), engram_slots=16381, engram_hashes=4, engram_dim=64)
NEW = dict(rope=True, qknorm=True, swiglu=True, pad=0)
VARIANTS = {   # list of pieces, optimiser, model switches, scoring, sequences x window per slice
    "normal":              ("gpt2", "adamw", {}, "full", 16, 1024),
    "new":                 ("bpe4096", "muon", {**NEW, **MEMORY}, "full", 16, 1024),
    "new-no-memory":       ("bpe4096", "muon", NEW, "full", 16, 1024),
    "new-16384":           ("bpe16384", "muon", {**NEW, **MEMORY}, "full", 16, 1024),
    "new-16384-sampled":   ("bpe16384", "muon", {**NEW, **MEMORY}, "sampled", 16, 1024),
    "new-gpt2-sampled":    ("gpt2", "muon", {**NEW, **MEMORY}, "sampled", 16, 1024),
    "new-phrase":          ("phrase65k", "muon", {**NEW, **MEMORY}, "full", 16, 1024),
    "new-phrase-sampled":  ("phrase65k", "muon", {**NEW, **MEMORY}, "sampled", 16, 1024),
    "new-window256":       ("bpe4096", "muon", {**NEW, **MEMORY}, "full", 64, 256),
    "new-fp8":             ("bpe4096", "muon", {**NEW, **MEMORY, "fp8": True}, "full", 16, 1024),
}
ACCUM, SAMPLES, WARM, TIMED = 4, 4096, 2, 12   # slices per update, sampled pieces, updates off and on the stopwatch


def measure(name):
    pieces, kind, parts, head, micro, ctx = VARIANTS[name]
    free, total = torch.cuda.mem_get_info()   # stop with an error rather than spill into system memory
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - 0.5 * 2**30) / total))
    torch.backends.cuda.matmul.allow_tf32 = True
    folder = HERE.parent / "data" / "pieces" / pieces
    manifest = json.loads((folder / "manifest.json").read_text())
    vocab = 50304 if pieces == "gpt2" else manifest["vocabulary"]
    data = np.memmap(folder / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    model = arch.build("124m", vocab=vocab, context=ctx, modern=False, **parts)
    opts = arch.optimizers(model, kind, 1e-3)
    body = torch.compile(model.features)
    if head == "sampled":
        freq = torch.from_numpy(np.bincount(data, minlength=vocab).astype(np.float32) + 1).cuda()
        freq /= freq.sum()
        log_expected = torch.log(SAMPLES * freq)

    def loss_of(x, y):
        h, want, w = body(x).flatten(0, 1), y.flatten(), model.out.weight
        if head == "full" and vocab > 8192:
            return fused_linear_ce(h, w, want, model.out.bias)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if head == "full":
                return F.cross_entropy(model.out(h).float(), want)
            draw = torch.multinomial(freq, SAMPLES, replacement=True)   # as lab.py's head="sampled"
            own = (h * w[want]).sum(-1).float()
            rest = (h @ w[draw].t()).float()
        if model.out.bias is not None:
            own, rest = own + model.out.bias[want], rest + model.out.bias[draw]
        rest = (rest - log_expected[draw]).masked_fill(want[:, None] == draw[None, :], float("-inf"))
        return (torch.logsumexp(torch.cat((own[:, None], rest), 1), 1) - own).mean()

    def update():
        for _ in range(ACCUM):
            at = rng.integers(0, len(data) - ctx - 1, micro)
            chunk = torch.from_numpy(data[at[:, None] + np.arange(ctx + 1)].astype(np.int64)).pin_memory().cuda(non_blocking=True)
            (loss_of(chunk[:, :-1], chunk[:, 1:]) / ACCUM).backward()
        torch.cuda.synchronize()
        mark = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - mark

    for _ in range(WARM):   # compiling and warming up
        update()
    began = time.perf_counter()
    optimiser = sum(update() for _ in range(TIMED))
    took = time.perf_counter() - began
    pieces_per_s = TIMED * ACCUM * micro * ctx / took
    r = {"variant": name, "list": pieces, "vocab": vocab, "letters_per_piece": manifest["letters per piece, val"],
         "weights_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "pieces_per_s": round(pieces_per_s),
         "letters_per_s": round(pieces_per_s * manifest["letters per piece, val"]), "s_per_update": round(took / TIMED, 3),
         "optimiser_share": round(optimiser / took, 3), "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    print(r)


def report():
    rows = {r["variant"]: r for r in map(json.loads, LOG.open(encoding="utf-8"))}   # the latest of each
    base = rows.get("normal", {}).get("letters_per_s")
    print(f"{'variant':20} {'list':>9} {'weights':>8} {'pieces/s':>9} {'letters/piece':>13} {'letters/s':>10} {'vs normal':>9} "
          f"{'optimiser':>9} {'peak GB':>7}")
    for name in VARIANTS:
        if name in rows:
            r = rows[name]
            print(f"{name:20} {r['vocab']:9,} {r['weights_M']:7.1f}M {r['pieces_per_s']:9,} {r['letters_per_piece']:13.2f} "
                  f"{r['letters_per_s']:10,} {r['letters_per_s'] / base if base else float('nan'):8.2f}x {r['optimiser_share']:9.1%} {r['peak_GB']:7.1f}")


if __name__ == "__main__":
    report() if sys.argv[1] == "report" else measure(sys.argv[1])
