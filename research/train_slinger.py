"""SlingerGPT 1 pico: the 124M model trained the new way on data/slinger/ (made by research/prepare_slinger.py).

    python -P research/train_slinger.py train            start, or carry on from the last checkpoint
    python -P research/train_slinger.py pause            the training saves a checkpoint and stops within seconds
    python -P research/train_slinger.py status           progress so far
    python -P research/train_slinger.py write "What is the Sun?"     an answer from the latest weights

Carrying on after a finished run (the 19-hour plan of 15 November: 4.0B more pieces, the last tenth in windows of
2,048 pieces), from the finished run's checkpoint into a run of its own (pause and resume with the same command):
    python -P research/train_slinger.py train --fp8 --name slingergpt1_pico_full --start-from slingergpt1_pico
        --updates 61035 --warm 0.02 --long --title "SlingerGPT 1 pico full"

The new way exactly as tested (research/test_124m.py, research/facts_quiz.py): 12 layers x 768, rotary positions,
QK-norm, SwiGLU, lookup memory before block 2 (16,381 x 4 x 2 tables of 64), Muon for the block matrices and AdamW
for the rest at a peak rate of 0.002, 4 slices of 16 windows of 1,024 pieces per weight update, the rate rising
over the first 10% of updates and falling along a cosine to 10%, bf16, model body compiled.
Each window comes from one source, picked by the shares below; the last fifth of the updates leans on the best
material. Safety: GPU memory is capped; the run pauses whenever the GPU reaches 87 C until it is back at 70 C,
takes a two-minute break every hour, and saves a checkpoint every 500 updates (about 10 minutes).
"""
import argparse, json, math, os, signal, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import arch

DATA, MORE, MODELS, LOGS = ROOT / "data" / "slinger", ROOT / "data" / "slinger_more", ROOT / "models", HERE / "logs"
CTX, MICRO, ACCUM, LR = 1024, 16, 4, 0.002
LONG_CTX = 2048   # --long: the last tenth reads windows of 2,048 pieces, 8 per slice (the same pieces per update)
MAIN = {"fineweb_best": 0.10, "fineweb_good": 0.22, "cosmopedia": 0.24, "smoltalk": 0.09, "soda": 0.06,
        "wikipedia": 0.11, "simplewiki": 0.04, "finemath": 0.11, "orca_math": 0.03}   # first 80% of the updates
FINAL = {"fineweb_best": 0.25, "fineweb_good": 0.0, "cosmopedia": 0.15, "smoltalk": 0.20, "soda": 0.05,
         "wikipedia": 0.09, "simplewiki": 0.06, "finemath": 0.12, "orca_math": 0.08}  # last 20%: the best material
OVERALL = {k: 0.8 * MAIN[k] + 0.2 * FINAL[k] for k in MAIN}   # also weights the score in every run, so scores compare
# Carrying on (--start-from), from data/slinger/ and data/slinger_more/ together. Shares chosen so that no source is
# read more than about 3 times over both runs (Simple Wikipedia and Orca-Math have no new text).
MAIN2 = {"fineweb_best": 0.195, "fineweb_good": 0.10, "cosmopedia": 0.23, "smoltalk": 0.10, "soda": 0.04,
         "wikipedia": 0.15, "simplewiki": 0.03, "finemath": 0.13, "orca_math": 0.025}
FINAL2 = {"fineweb_best": 0.25, "fineweb_good": 0.0, "cosmopedia": 0.15, "smoltalk": 0.22, "soda": 0.05,
          "wikipedia": 0.10, "simplewiki": 0.05, "finemath": 0.12, "orca_math": 0.06}
CFG = dict(size="124m", vocab=4096, context=CTX, modern=False, rope=True, qknorm=True, swiglu=True, engram=(1,),
           engram_slots=16381, engram_hashes=4, engram_dim=64, pad=0)
HOT, COOL, BREAK_EVERY, BREAK = 87, 70, 3000, 120
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def gpu_temp():
    try:
        return float(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True))
    except Exception:
        return 0.0


class Mix:
    """Windows of `length` + 1 pieces, each from one source picked by the shares, in pinned memory for a fast copy.
    A source with text in several folders picks a file in proportion to its size, so every piece is equally likely."""
    def __init__(self, part, seed, folders=(DATA,)):
        self.data = {k: [np.memmap(f / k / f"{part}.bin", dtype=np.uint16, mode="r") for f in folders
                         if (f / k / f"{part}.bin").exists()] for k in MAIN}
        self.size = {k: np.array([len(d) for d in files], dtype=np.float64) for k, files in self.data.items()}
        self.rng = np.random.default_rng(seed)

    def window(self, name, length=CTX):
        files = self.data[name]
        d = files[0] if len(files) == 1 else files[self.rng.choice(len(files), p=self.size[name] / self.size[name].sum())]
        at = int(self.rng.integers(0, len(d) - length - 1))
        return d[at:at + length + 1]

    def next(self, shares, count, length=CTX):
        names = list(shares)
        picks = self.rng.choice(len(names), size=count, p=np.array([shares[n] for n in names]) / sum(shares.values()))
        chunk = torch.from_numpy(np.stack([self.window(names[i], length) for i in picks]).astype(np.int64)).pin_memory()
        chunk = chunk.cuda(non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]


def held_out():
    """The same 8 held-out windows of every source each time, so scores are comparable through the run."""
    mix, out = Mix("val", seed=7), {}
    for name in MAIN:
        rows = np.stack([mix.window(name) for _ in range(8)]).astype(np.int64)
        out[name] = torch.from_numpy(rows).cuda()
    return out


def train(args):
    free, total = torch.cuda.mem_get_info()   # stop with an error rather than spill into system memory
    torch.cuda.set_per_process_memory_fraction(max(0.1, (free - 0.5 * 2**30) / total))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    per = {k: manifest["sources"][k]["val"]["letters per piece"] for k in MAIN}
    ckpt, final, log = MODELS / f"{args.name}.ckpt", MODELS / f"{args.name}.pt", LOGS / f"{args.name}.log"
    torch.manual_seed(1337)
    # --fp8: the blocks' matrix multiplies in 8-bit; a 16-bit checkpoint carries on as is. --long builds the rotary
    # table for 2,048 positions; the first 1,024 are the same numbers as before.
    cfg = {**CFG, "fp8": args.fp8, "context": LONG_CTX if args.long else CTX}
    model = arch.build(**cfg)
    opts = arch.optimizers(model, "muon", LR)
    main_mix, final_mix = (MAIN2, FINAL2) if args.start_from else (MAIN, FINAL)
    mix, step, spent, curve = Mix("train", seed=2024 if not args.start_from else 2025,
                                  folders=(DATA, MORE) if args.start_from else (DATA,)), 0, 0.0, []
    if args.start_from and not ckpt.exists():   # a new run from a finished one: its weights and optimiser state
        state = torch.load(MODELS / f"{args.start_from}.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["optimizers"]):
            o.load_state_dict(s)
        print(f"starting from {args.start_from} (its update {state['step']:,}); {args.updates:,} more updates", flush=True)
        del state
    if ckpt.exists():   # read into system memory, so no second copy of the weights sits on the card
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        for o, s in zip(opts, state["optimizers"]):
            o.load_state_dict(s)
        mix.rng.bit_generator.state = state["rng"]
        step, spent, curve = state["step"], state["seconds"], state["curve"]
        del state
        print(f"carrying on from update {step:,} of {args.updates:,}", flush=True)
    body = torch.compile(model.features)
    val = held_out()

    def write_log(line):
        with log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def save():
        state = {"model": model.state_dict(), "optimizers": [o.state_dict() for o in opts], "step": step,
                 "rng": mix.rng.bit_generator.state, "seconds": spent, "curve": curve, "cfg": cfg, "updates": args.updates}
        torch.save(state, str(ckpt) + ".tmp")
        os.replace(str(ckpt) + ".tmp", ckpt)

    @torch.no_grad()
    def score():
        model.eval()
        loss = {}
        for name, rows in val.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.out(model.features(rows[:, :-1]))
            loss[name] = F.cross_entropy(logits.float().flatten(0, 1), rows[:, 1:].flatten()).item()
        model.train()
        overall = sum(OVERALL[k] * loss[k] for k in MAIN) / sum(OVERALL[k] * per[k] for k in MAIN)
        return overall, {k: round(loss[k] / per[k], 4) for k in MAIN}

    asked = []
    previous = signal.signal(signal.SIGINT, lambda *_: asked.append(True))
    pause_file = Path(str(ckpt) + ".pause")
    warm, began, window_start, window_step, last_break = max(1, int(args.updates * args.warm)), time.time(), time.time(), step, step
    mark = None
    while step < args.updates:
        if asked or pause_file.exists() or (args.stop_after and step >= args.stop_after):
            spent += time.time() - began
            save()
            pause_file.unlink(missing_ok=True)
            signal.signal(signal.SIGINT, previous)
            write_log(f"{time.strftime('%H:%M:%S')} paused at update {step:,}; carry on with: python -P {' '.join(sys.argv)}")
            return
        if step % 25 == 0 and gpu_temp() >= HOT:   # too hot: wait, off the stopwatch
            spent += time.time() - began
            write_log(f"{time.strftime('%H:%M:%S')} GPU at {gpu_temp():.0f} C, cooling down")
            while gpu_temp() > COOL:
                time.sleep(10)
            began = time.time()
        if step % BREAK_EVERY == 0 and step != last_break:   # an hourly break, off the stopwatch
            spent += time.time() - began
            time.sleep(BREAK)
            began, last_break = time.time(), step
        share = step / args.updates
        factor = min(1.0, (step + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * share)))
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * factor
        shares = main_mix if share < 0.8 else final_mix
        ctx, micro = (LONG_CTX, MICRO * CTX // LONG_CTX) if args.long and share >= 0.9 else (CTX, MICRO)
        for _ in range(ACCUM):
            x, y = mix.next(shares, micro, ctx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.out(body(x))
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0 and mark:   # a speed reading every 100 updates (to the .out file), so a slowdown shows early
            print(f"{time.strftime('%H:%M:%S')} update {step}: {100 * MICRO * ACCUM * CTX / (time.time() - mark):,.0f} pieces/s", flush=True)
        if step % args.every == 0 or step == args.updates:
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at update {step}")
            spent += time.time() - began
            overall, parts = score()
            rate = (step - window_step) * MICRO * ACCUM * CTX / max(1e-9, time.time() - window_start)
            curve.append([step, round(overall, 4), round(spent, 1)])
            save()
            write_log(json.dumps({"time": time.strftime("%H:%M:%S"), "update": step, "of": args.updates,
                                  "hours": round(spent / 3600, 2), "wrongness_per_letter": round(overall, 4),
                                  "by_source": parts, "pieces_per_s": round(rate), "gpu_C": gpu_temp()}))
            began, window_start, window_step = time.time(), time.time(), step
        if step % 100 == 0:
            mark = time.time()   # set after any score check, so each reading counts training only
    signal.signal(signal.SIGINT, previous)
    torch.save({"name": args.title, "cfg": cfg, "model": model.state_dict(), "updates": step, "start_from": args.start_from,
                "pieces": step * MICRO * ACCUM * CTX, "training_seconds": round(spent), "curve": curve}, final)
    shutil.copy(DATA / "tokenizer.json", MODELS / f"{args.name}.tokenizer.json")
    write_log(f"{time.strftime('%H:%M:%S')} done: {step:,} updates, {spent / 3600:.2f} hours of training, saved {final.name}")


@torch.no_grad()
def write(args):
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    path = MODELS / f"{args.name}.pt"
    state = torch.load(path if path.exists() else MODELS / f"{args.name}.ckpt", map_location="cuda", weights_only=False)
    model = arch.build(**{**state["cfg"], "fp8": False})   # answering runs in 16-bit either way
    model.load_state_dict(state["model"])
    model.eval()
    ids = tok.encode(f"User: {args.prompt}\nAssistant:" if args.chat else args.prompt).ids
    start = len(ids)
    for _ in range(args.length):
        logits = model(torch.tensor([ids[-CTX:]]).cuda())[0, -1].float() / args.temperature
        top = torch.topk(logits, args.top_k)
        nxt = int(top.indices[torch.multinomial(torch.softmax(top.values, -1), 1)])
        if nxt == 0:
            break
        ids.append(nxt)
        if args.chat and "\nUser:" in tok.decode(ids[start:]):
            break
    print(tok.decode(ids[start:]).split("\nUser:")[0].strip())


def status(args):
    log = LOGS / f"{args.name}.log"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    for line in lines[-args.lines:]:
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["train", "pause", "status", "write"])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--name", default="slingergpt1_pico")
    p.add_argument("--updates", type=int, default=30_520, help="weight updates of 65,536 pieces (2.0B pieces, ~10 hours)")
    p.add_argument("--every", type=int, default=500, help="updates between score checks and checkpoints")
    p.add_argument("--stop-after", type=int, default=0, help="for testing: stop as if paused after this many updates")
    p.add_argument("--fp8", action="store_true", help="the blocks' matrix multiplies in 8-bit (torchao Float8Linear)")
    p.add_argument("--start-from", default="", help="a finished run whose checkpoint a new run starts from")
    p.add_argument("--warm", type=float, default=0.10, help="share of the updates over which the rate rises")
    p.add_argument("--long", action="store_true", help="the last tenth of the updates in windows of 2,048 pieces")
    p.add_argument("--title", default="SlingerGPT 1 pico", help="the name saved with the finished model")
    p.add_argument("--chat", action="store_true", default=True)
    p.add_argument("--raw", dest="chat", action="store_false", help="write: continue the prompt as plain text")
    p.add_argument("--length", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--lines", type=int, default=8)
    a = p.parse_args()
    if a.what == "pause":
        Path(MODELS / f"{a.name}.ckpt.pause").touch()
        print("asked the training to save and stop")
    else:
        {"train": train, "status": status, "write": write}[a.what](a)
