"""SlingerGPT 492M: the 8-GPU trainer (plan: research/492m/TRAINING_PLAN.md, section 2 and stage A.3-A.4).

    python research/train_492m.py train --gpus 8                  start, or carry on from the last save
    python research/train_492m.py train --gpus 8 --stop-after 300 the test run: speed, memory, then save and stop
    python research/train_492m.py ratecheck --gpus 8              200 updates at each of 0.0015, 0.002 and 0.003
    python research/train_492m.py pause                           save and stop within one update
    python research/train_492m.py status                          the last lines of the log

The text comes from prepare_492m.py: <data>/<source>/NNNN.bin (16-bit piece numbers, documents ended by piece 0)
and <data>/manifest.json, which lists only finished files. The shares come from research/492m/mix_492m.json.

How the windows are chosen (no reader state to save): every update reads the same number of pieces (2^20), cut
into windows whose length follows the schedule below. Window slot j of a schedule segment goes to the source whose
share interval holds frac((j + 1) x 0.618...), a low-discrepancy sequence, so every source gets its share within a
window or two over any stretch of updates. Each source is read in order, as one stream made of its finished files,
from where the earlier windows left it; a source that has been read to its end starts again (a second pass) once
its preparation is complete. All of this follows from the update number alone, so a resumed run reads exactly the
windows it would have read. GPU r takes slots r, r + G, r + 2G, ... of each update.

If a source's preparation falls behind the reading, its windows are read from a random place in what is ready
(a repeat) and the log says so: tell the owner.

Saving: every 15 minutes (the last 2 are kept), a full copy at the end of each schedule segment and at 74B, and on
SIGTERM, SIGHUP or SIGINT (Nebius sends SIGTERM 60 seconds before it stops a spot server; tmux passes SIGHUP when
it is killed). At the start of every update the processes agree, with one tiny all-reduce, whether any of them was
asked to stop; if so GPU 0 saves and all stop together.
"""
import argparse, contextlib, hashlib, json, math, os, queue, signal, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import arch
from fused_ce import fused_linear_ce

GOLDEN = 0.6180339887498949
PIECES_PER_UPDATE = 2 ** 20
TOTAL_PIECES = 80_000_000_000
# (ends at this share of the run, window, mix phase, rotary base, document mask)
SCHEDULE = [(25 / 80, 1024, "main", 10_000, False), (50 / 80, 2048, "main", 10_000, False),
            (70 / 80, 4096, "main", 10_000, False), (78 / 80, 4096, "anneal", 10_000, False),
            (1.0, 16384, "long", 1_000_000, True)]
EXTRA_COPIES = [74 / 80]   # data_mix.md: a copy at 74B as well, so the owner can compare partway through the anneal
CFG = dict(size="500m", vocab=16384, context=16384, modern=False, rope=True, qknorm=True, swiglu=True, engram=(1,),
           engram_slots=65521, engram_hashes=4, engram_dim=64, pad=0)
WEIGHTS = 492_406_016
HOT = 90
VAL_WINDOW, VAL_WINDOWS = 1024, 8


# ---------------------------------------------------------------------------------------------------------- the text

class Streams:
    """Each source as one stream of piece numbers: the finished files of manifest.json, in order, without gaps."""
    def __init__(self, data):
        self.data, self.files, self.mtime = Path(data), {}, None
        self.reload()

    def reload(self):
        path = self.data / "manifest.json"
        st = path.stat()
        mtime = (st.st_mtime_ns, st.st_size)
        if mtime == self.mtime:
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.mtime, self.manifest = mtime, manifest
        for name, info in manifest["sources"].items():
            done, have = [], self.files.get(name, ([], None, None))[0]
            for i, f in enumerate(info.get("files", [])):   # file NNNN only once 0000 to NNNN-1 are all finished
                if f["name"] != f"{i:04d}.bin" or not f.get("pieces"):
                    break
                done.append(f)
            maps = have[:]   # keep the memory maps already open
            for f in done[len(maps):]:
                maps.append(np.memmap(self.data / name / f["name"], dtype=np.uint16, mode="r"))
            ends = np.cumsum([len(m) for m in maps]) if maps else np.zeros(0, dtype=np.int64)
            self.files[name] = (maps, ends, bool(info.get("complete")) and len(done) == len(info.get("files", [])))

    def ready(self, name):
        maps, ends, _ = self.files.get(name, ([], np.zeros(0), False))
        return int(ends[-1]) if len(ends) else 0

    def complete(self, name):
        return self.files.get(name, ([], None, False))[2]

    def read(self, name, at, length):
        """Pieces [at, at + length) of the stream (at < ready); may span files; wraps at the end of the stream."""
        maps, ends, _ = self.files[name]
        out, total = [], int(ends[-1])
        at %= total
        while length > 0:
            i = int(np.searchsorted(ends, at, side="right"))
            start = at - (int(ends[i - 1]) if i else 0)
            part = maps[i][start:start + length]
            out.append(part)
            length -= len(part)
            at = (at + len(part)) % total
        return np.concatenate(out) if len(out) > 1 else out[0]

    def val(self, name):
        path = self.data / name / "val.bin"
        info = self.manifest["sources"].get(name, {}).get("val")
        if not path.exists() or not info or not info.get("pieces"):
            return None, None
        return np.memmap(path, dtype=np.uint16, mode="r"), info["letters"] / info["pieces"]


class Plan:
    """The schedule in updates, with each segment's window, shares and the slot-to-source rule."""
    def __init__(self, mix, total_pieces, per_update, windows=None, seed=0):
        self.per_update, self.updates = per_update, max(1, round(total_pieces / per_update))
        self.segments, start = [], 0
        for i, (end, window, phase, base, masked) in enumerate(SCHEDULE):
            window = windows[i] if windows else window
            stop = self.updates if i == len(SCHEDULE) - 1 else round(end * self.updates)
            shares = {k: v for k, v in mix["phases"][phase].items() if v > 0}
            names = sorted(shares)
            cum = np.cumsum([shares[n] for n in names]) / sum(shares.values())
            cum[-1] = 1.0
            assert per_update % window == 0, "every update must be a whole number of windows"
            self.segments.append(dict(start=start, stop=stop, window=window, phase=phase, base=base, masked=masked,
                                      names=names, cum=cum, slots=per_update // window))
            start = stop
        self.copies = sorted({s["stop"] for s in self.segments} | {round(e * self.updates) for e in EXTRA_COPIES})
        self.offset = (seed * GOLDEN) % 1.0
        self.sources = sorted({n for s in self.segments for n in s["names"]})
        # each source's share of everything the run reads: the weights of the overall held-out score
        weight = {n: 0.0 for n in self.sources}
        for s in self.segments:
            share = dict(mix["phases"][s["phase"]])
            norm = sum(share[n] for n in s["names"])
            for n in s["names"]:
                weight[n] += (s["stop"] - s["start"]) * share[n] / norm
        self.weight = {n: w / self.updates for n, w in weight.items()}

    def segment(self, update):
        for s in self.segments:
            if update < s["stop"]:
                return s
        return self.segments[-1]

    def sources_of(self, seg, first, last):
        """Source index of every slot in [first, last) of a segment."""
        j = np.arange(first, last, dtype=np.float64)
        return np.searchsorted(seg["cum"], np.mod((j + 1) * GOLDEN + self.offset, 1.0), side="right")

    def counts(self, seg, first, last, chunk=1 << 24):
        c = np.zeros(len(seg["names"]), dtype=np.int64)
        for a in range(first, last, chunk):
            c += np.bincount(self.sources_of(seg, a, min(last, a + chunk)), minlength=len(seg["names"]))
        return c

    def cursors(self, update):
        """Pieces of each source read before `update` (including the slots of earlier segments)."""
        at = {n: 0 for n in self.sources}
        for seg in self.segments:
            if update <= seg["start"]:
                break
            upto = min(update, seg["stop"]) - seg["start"]
            for n, c in zip(seg["names"], self.counts(seg, 0, upto * seg["slots"])):
                at[n] += int(c) * seg["window"]
        return at


class Reader:
    """Makes each update's windows for one GPU in a background thread, ahead of the training."""
    def __init__(self, plan, streams, rank, world, micro_pieces, start, seed, device):
        self.plan, self.streams, self.rank, self.world = plan, streams, rank, world
        self.micro_pieces, self.seed, self.device = micro_pieces, seed, device
        self.cursor = plan.cursors(start)
        self.update, self.behind, self.substituted = start, 0, 0
        self.queue, self.lock = queue.Queue(maxsize=2), threading.Lock()
        self.notes = []
        threading.Thread(target=self.fill, daemon=True).start()

    def fill(self):
        try:
            while True:
                self.queue.put(self.make(self.update))
                self.update += 1
        except Exception as e:   # surfaced by next()
            self.queue.put(e)

    def make(self, update):
        if update % 50 == 0:
            self.streams.reload()
        plan, seg = self.plan, self.plan.segment(update)
        first = (update - seg["start"]) * seg["slots"]
        src = plan.sources_of(seg, first, first + seg["slots"])
        window, rows = seg["window"], []
        for j, s in enumerate(src):
            name = seg["names"][s]
            at = self.cursor[name]
            self.cursor[name] += window
            if j % self.world != self.rank:
                continue
            rows.append(self.window(name, at, window, update, j, seg))
        rows = torch.from_numpy(np.stack(rows).astype(np.int64))
        micro = max(1, min(len(rows), self.micro_pieces // window))
        batches = []
        for a in range(0, len(rows), micro):
            chunk = rows[a:a + micro]
            if self.device.type == "cuda":
                chunk = chunk.pin_memory()
            batches.append(chunk)
        return update, batches

    def window(self, name, at, window, update, j, seg):
        need, streams = window + 1, self.streams
        ready = streams.ready(name)
        if ready < need:   # nothing ready for this source: a window from another source of this phase, by share
            with self.lock:
                self.substituted += 1
            rng = np.random.default_rng([self.seed, update, j, 1])
            ok = [n for n in seg["names"] if streams.ready(n) >= need]
            if not ok:
                raise RuntimeError(f"no source of phase {seg['phase']} has a window of {need} pieces ready")
            name = ok[int(rng.integers(len(ok)))]
            ready = streams.ready(name)
            return streams.read(name, int(rng.integers(0, ready - need + 1)), need)
        if at + need <= ready or streams.complete(name):
            return streams.read(name, at, need)
        with self.lock:   # preparation is behind: a repeat from a random place in what is ready
            self.behind += 1
        rng = np.random.default_rng([self.seed, update, j, 2])
        return streams.read(name, int(rng.integers(0, ready - need + 1)), need)

    def next(self):
        item = self.queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def ahead(self, update):
        """For each source of the current phase that is still being prepared: updates of reading left in what is ready."""
        seg = self.plan.segment(update)
        per = {n: seg["slots"] * seg["window"] * (seg["cum"][i] - (seg["cum"][i - 1] if i else 0))
               for i, n in enumerate(seg["names"])}
        cursor = dict(self.cursor)   # at most 2 updates ahead of `update`
        return {n: round((self.streams.ready(n) - cursor[n]) / per[n]) for n in seg["names"]
                if not self.streams.complete(n)}


# ------------------------------------------------------------------------------------------------------- the model

class Loss(nn.Module):
    """The model and its loss in one module, so DistributedDataParallel sees the whole forward pass."""
    def __init__(self, model, body, fused):
        super().__init__()
        self.model, self.body, self.fused = model, body, fused

    def forward(self, x, y, mask=None):
        h = self.body(x, mask=mask)
        if self.fused:
            return fused_linear_ce(h.flatten(0, 1), self.model.out.weight, y.flatten(), self.model.out.bias)
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            logits = self.model.out(h)
        return F.cross_entropy(logits.float().flatten(0, 1), y.flatten())


def gpu_report():
    """Temperature, slowdown reasons, power and use of every GPU (nvidia-smi), or [] without one."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,temperature.gpu,clocks_throttle_reasons.active,"
                                       "power.draw,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        i, t, why, p, u = [v.strip() for v in line.split(",")]
        rows.append({"gpu": int(i), "C": float(t), "slowdown": why, "W": float(p), "busy": float(u)})
    return rows


THERMAL = 0x20 | 0x40 | 0x8   # SW thermal, HW thermal and HW slowdown bits of clocks_throttle_reasons.active


def paths(args):
    out = Path(args.out or Path(args.data) / "runs")
    return out, out / f"{args.name}.ckpt", out / f"{args.name}.prev.ckpt", out / f"{args.name}.log"


def worker(rank, world, args):
    cuda = args.device == "cuda"
    if world > 1 or "RANK" in os.environ:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.port))
        dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world,
                                timeout=__import__("datetime").timedelta(minutes=30))
    device = torch.device(f"cuda:{rank % max(1, torch.cuda.device_count())}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        arch.ATTENTION = {"cudnn": [arch.SDPBackend.CUDNN_ATTENTION], "flash": [arch.SDPBackend.FLASH_ATTENTION],
                          "efficient": [arch.SDPBackend.EFFICIENT_ATTENTION]}[args.attention] + \
            [arch.SDPBackend.EFFICIENT_ATTENTION, arch.SDPBackend.MATH]
    else:
        torch.set_num_threads(max(1, args.threads))
    stop = []
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, lambda n, f: stop.append(n))
    try:
        if args.what == "ratecheck":
            results = {}
            for lr in [float(r) for r in args.rates.split(",")]:
                results[lr] = run(rank, world, args, device, stop, lr=lr, name=f"{args.name}_rate{lr}", fresh=True)
                if stop:
                    break
            if rank == 0:
                print(json.dumps({"rate check (held-out wrongness per letter)": results}), flush=True)
        else:
            run(rank, world, args, device, stop, lr=args.lr, name=args.name, fresh=False)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(rank, world, args, device, stop, lr, name, fresh):
    args = argparse.Namespace(**{**vars(args), "name": name})
    out, ckpt, prev, logfile = paths(args)
    lead = rank == 0
    if lead:
        out.mkdir(parents=True, exist_ok=True)
    streams = Streams(args.data)
    mix = json.loads(Path(args.mix).read_text(encoding="utf-8"))
    windows = [int(w) for w in args.windows.split(",")] if args.windows else None
    plan = Plan(mix, args.total_pieces, args.per_update, windows, seed=args.seed)
    updates = plan.updates
    vocab = int(streams.manifest["pieces in the list"])
    cfg = {**CFG, "vocab": vocab, "fp8": args.fp8}
    if args.size != "492m":   # small models for the CPU tests
        cfg = {**cfg, "size": args.size, "engram_slots": 1021, "engram_hashes": 2, "engram_dim": 16}
    cfg["context"] = max(s["window"] for s in plan.segments)
    torch.manual_seed(args.seed)   # the same starting weights on every GPU (DDP also copies GPU 0's)
    model = arch.build(device=device, **cfg)
    count = sum(p.numel() for p in model.parameters())
    if args.size == "492m":
        assert count == WEIGHTS, f"the model has {count:,} weights, not {WEIGHTS:,}"
    opts = arch.optimizers(model, "muon", lr)
    if device.type == "cpu":   # the fused AdamW and Muon of the GPU build need no change; on the CPU, plain AdamW
        for o in opts:
            if isinstance(o, torch.optim.AdamW):
                o.defaults["fused"] = False
                for g in o.param_groups:
                    g["fused"] = False
    warm = args.warm_updates or max(1, int(updates * args.warm))
    step, spent, curve = 0, 0.0, []
    if not fresh:
        state = load(ckpt, prev)
        if state is not None:
            model.load_state_dict(state["model"])
            for o, s in zip(opts, state["optimizers"]):
                o.load_state_dict(s)
            step, spent, curve = state["step"], state["seconds"], state["curve"]
            if state["plan"] != describe(plan) and not args.allow_plan_change:
                raise SystemExit("the schedule or shares changed since this run was saved (--allow-plan-change to go on)")
            del state
            if lead:
                print(f"carrying on from update {step:,} of {updates:,}", flush=True)
    model.set_rope_base(plan.segment(step)["base"])
    body = torch.compile(model.features) if args.compile else model.features
    net = Loss(model, body, args.fused_ce).to(device)
    if dist.is_initialized():
        net = nn.parallel.DistributedDataParallel(net, device_ids=[device.index] if device.type == "cuda" else None,
                                                  broadcast_buffers=False, bucket_cap_mb=args.bucket_mb,
                                                  gradient_as_bucket_view=True)
    reader = Reader(plan, streams, rank, world, args.micro_pieces, step, args.seed, device)
    assert all(s["slots"] % world == 0 for s in plan.segments), "every update's windows must split evenly over the GPUs"
    val = held_out(streams, plan, device, min(VAL_WINDOW, plan.segments[0]["window"]))
    stop_after = min(updates, step + args.stop_after) if args.stop_after else updates
    if args.updates_limit:
        stop_after = min(stop_after, args.updates_limit)

    def log(record):
        if lead:
            line = json.dumps(record) if isinstance(record, dict) else record
            with logfile.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(line, flush=True)

    def save(path, final=False):
        if not lead:
            return 0.0
        t0 = time.time()
        state = {"model": model.state_dict(), "optimizers": [o.state_dict() for o in opts], "step": step,
                 "seconds": spent + (time.time() - began), "curve": curve, "cfg": {**cfg, "rope_base": model.c.rope_base},
                 "plan": describe(plan),
                 "args": {k: v for k, v in vars(args).items()}}
        tmp = Path(str(path) + ".tmp")
        torch.save(state, tmp)
        if path == ckpt and ckpt.exists():   # keep the last 2 regular saves
            os.replace(ckpt, prev)
        os.replace(tmp, path)
        return time.time() - t0

    if lead:
        log({"time": time.strftime("%H:%M:%S"), "event": "start", "update": step, "of": updates, "weights": count,
             "gpus": world, "pieces_per_update": args.per_update, "lr": lr, "warm_updates": warm,
             "segments": [{k: s[k] for k in ("start", "stop", "window", "phase", "base", "masked")} for s in plan.segments]})
    pause = Path(str(ckpt) + ".pause")
    began = last_save = last_gpu = mark_time = time.time()
    mark_step, losses = step, []
    flag = torch.zeros(1, device=device)
    while step < stop_after:
        flag.fill_(1.0 if stop or (lead and pause.exists()) else 0.0)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0:
            seconds = save(ckpt)
            pause.unlink(missing_ok=True) if lead else None
            log({"time": time.strftime("%H:%M:%S"), "event": "stopped and saved", "update": step,
                 "save_seconds": round(seconds, 1), "signal": [signal.Signals(s).name for s in stop]})
            return None
        seg = plan.segment(step)
        if model.c.rope_base != seg["base"]:
            model.set_rope_base(seg["base"])
            log({"time": time.strftime("%H:%M:%S"), "event": f"rotary base now {seg['base']:,}", "update": step})
        share = step / updates
        factor = min(1.0, (step + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * share)))
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * factor
        got, batches = reader.next()
        assert got == step, (got, step)
        if args.trace:   # tests: a fingerprint of the windows this GPU reads at each update
            with (out / f"{name}.windows.{rank}.txt").open("a") as f:
                f.write(f"{step} {hashlib.sha1(b''.join(b.numpy().tobytes() for b in batches)).hexdigest()}\n")
        total = 0.0
        for i, rows in enumerate(batches):
            rows = rows.to(device, non_blocking=True)
            x, y = rows[:, :-1], rows[:, 1:]
            mask = arch.document_mask(x, _compile=x.is_cuda) if seg["masked"] or args.doc_mask_all else None
            sync = i == len(batches) - 1 or not isinstance(net, nn.parallel.DistributedDataParallel)
            with (contextlib.nullcontext() if sync else net.no_sync()):
                loss = net(x, y, mask) / len(batches)
                loss.backward()
            total += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        losses.append(total)
        if step % args.log_every == 0 or step == stop_after:
            mean = torch.stack(losses).mean().reshape(1).float()
            if dist.is_initialized():
                dist.all_reduce(mean)
                mean /= world
            if not math.isfinite(mean.item()):
                raise FloatingPointError(f"loss is {mean.item()} at update {step}")
            now = time.time()
            rate = (step - mark_step) * args.per_update / max(1e-9, now - mark_time)
            hours = (spent + now - began) / 3600
            left = (updates - step) * args.per_update / max(1.0, rate) / 3600
            record = {"time": time.strftime("%H:%M:%S"), "update": step, "of": updates, "done": f"{100 * step / updates:.1f}%",
                      "pieces_B": round(step * args.per_update / 1e9, 3), "window": seg["window"], "phase": seg["phase"],
                      "loss": round(mean.item(), 4), "grad_norm": round(float(norm), 3), "lr": round(lr * factor, 6),
                      "pieces_per_s": round(rate), "hours": round(hours, 3), "hours_left_at_this_speed": round(left, 2),
                      "cost_so_far_usd": round(hours * args.price_per_hour, 1)}
            ahead = reader.ahead(step)
            if ahead:
                record["updates_of_text_ready_ahead_min"] = min(ahead.values())
                short = {n: a for n, a in ahead.items() if a < args.warn_ahead}
                if short:
                    record["warning"] = f"text preparation is close to the reading for {short}: tell the owner"
            if reader.behind or reader.substituted:
                record["windows_repeated_because_preparation_is_behind"] = reader.behind
                record["windows_from_another_source"] = reader.substituted
            log(record)
            losses, mark_time, mark_step = [], time.time(), step
        if lead and time.time() - last_gpu > 300:
            report = gpu_report()
            hot = [g for g in report if g["C"] >= HOT or int(g["slowdown"], 16) & THERMAL]
            log({"time": time.strftime("%H:%M:%S"), "gpus": report, **({"warning": f"GPU heat: {hot}: tell the owner"}
                                                                           if hot else {})})
            last_gpu = time.time()
        if step % args.score_every == 0 or step == stop_after or step in plan.copies:
            overall, parts = score(model, val, plan)
            same = same_weights(model)
            curve.append([step, overall, round(spent + time.time() - began)])
            log({"time": time.strftime("%H:%M:%S"), "update": step, "held_out_wrongness_per_letter": overall,
                 "by_source": parts, "copies_identical": same})
            if not same:
                raise RuntimeError("the copies of the model on the GPUs have drifted apart")
            mark_time = time.time()   # score time is not training speed
        if step in plan.copies:
            seconds = save(out / f"{name}.{billions(step, args)}B.ckpt")
            log({"time": time.strftime("%H:%M:%S"), "event": "milestone copy", "update": step, "save_seconds": round(seconds, 1)})
            last_save = mark_time = time.time()
        elif time.time() - last_save > args.save_minutes * 60:
            seconds = save(ckpt)
            log({"time": time.strftime("%H:%M:%S"), "event": "saved", "update": step, "save_seconds": round(seconds, 1)})
            last_save = mark_time = time.time()
    spent += time.time() - began
    if step < updates:   # the test run's stop
        began = time.time()
        seconds = save(ckpt)
        log({"time": time.strftime("%H:%M:%S"), "event": "stopped after the requested updates and saved", "update": step,
             "save_seconds": round(seconds, 1)})
        return curve[-1][1] if curve else None
    if lead:
        weights = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
        torch.save({"name": args.title, "cfg": {**cfg, "rope_base": model.c.rope_base}, "model": weights, "updates": step,
                    "pieces": step * args.per_update,
                    "training_seconds": round(spent), "curve": curve}, out / f"{name}.pt")
        tok = Path(args.data) / "tokenizer.json"
        if tok.exists():
            (out / f"{name}.tokenizer.json").write_bytes(tok.read_bytes())
        log({"time": time.strftime("%H:%M:%S"), "event": "done", "updates": step, "hours": round(spent / 3600, 2),
             "saved": f"{name}.pt (16-bit weights) and the full copy {name}.{billions(step, args)}B.ckpt"})
    return curve[-1][1] if curve else None


def billions(step, args):
    return f"{round(step * args.per_update / 1e9, 4):g}"   # 25, 50, 70, 74, 78, 80 in the real run


def describe(plan):
    """What a save must match to be carried on: the schedule and shares."""
    text = json.dumps([[s[k] if k != "cum" else [round(float(c), 9) for c in s[k]]
                        for k in ("start", "stop", "window", "phase", "names", "cum")] for s in plan.segments])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load(ckpt, prev):
    """The newest save that reads back whole: the last regular save, else the one before, else a finished .tmp."""
    best = None
    for path in (ckpt, prev, Path(str(ckpt) + ".tmp")):
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"could not read {path.name}: {e}", flush=True)
            continue
        if best is None or state["step"] > best["step"]:
            best = state
    return best


def held_out(streams, plan, device, window=VAL_WINDOW):
    """The same windows of every source's held-out text each time, so scores compare through the run."""
    out = {}
    for name in plan.sources:
        data, per = streams.val(name)
        if data is None or len(data) < window + 1:
            continue
        n = min(VAL_WINDOWS, len(data) // (window + 1))
        at = np.linspace(0, len(data) - window - 1, n).astype(np.int64)
        rows = np.stack([data[a:a + window + 1] for a in at]).astype(np.int64)
        out[name] = (torch.from_numpy(rows).to(device), per)
    return out


@torch.no_grad()
def score(model, val, plan):
    """Wrongness per letter: average -ln p per piece divided by letters per piece, per source and weighted overall."""
    if not val:
        return None, {}
    model.eval()
    parts = {}
    for name, (rows, per) in val.items():
        with torch.autocast(rows.device.type, dtype=torch.bfloat16):
            logits = model.out(model.features(rows[:, :-1]))
        parts[name] = F.cross_entropy(logits.float().flatten(0, 1), rows[:, 1:].flatten()).item() / per
    model.train()
    w = {n: plan.weight.get(n, 0.0) for n in parts}
    overall = sum(w[n] * parts[n] for n in parts) / max(1e-12, sum(w.values()))
    return round(overall, 4), {k: round(v, 4) for k, v in parts.items()}


def same_weights(model):
    """Whether every GPU holds the same weights (a sum of all of them, compared across GPUs)."""
    if not dist.is_initialized():
        return True
    s = torch.stack([p.detach().double().sum() for p in model.parameters()]).sum().reshape(1)
    every = [torch.zeros_like(s) for _ in range(dist.get_world_size())]
    dist.all_gather(every, s)
    return all(torch.equal(every[0], e) for e in every)


# --------------------------------------------------------------------------------------------------- start and stop

def launch(args):
    """Starts one process per GPU. The launcher itself ignores stop signals and passes them on, so each process
    can save; it does not kill them after a time limit (torchrun would, after 30 seconds)."""
    if "RANK" in os.environ:   # started by torchrun
        return worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    if args.gpus == 1:
        return worker(0, 1, args)
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, args.gpus, args)) for r in range(args.gpus)]
    for p in procs:
        p.start()

    def forward(n, f):
        for p in procs:
            if p.is_alive():
                os.kill(p.pid, n)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, forward)
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode]
    if bad:
        sys.exit(f"a training process failed (exit codes {bad})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["train", "ratecheck", "pause", "status"])
    p.add_argument("--name", default="slingergpt_492m")
    p.add_argument("--data", default="/data/492m", help="the prepared text (prepare_492m.py's output)")
    p.add_argument("--out", default="", help="saves and logs (default <data>/runs)")
    p.add_argument("--mix", default=str(HERE / "492m" / "mix_492m.json"))
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--port", type=int, default=29512)
    p.add_argument("--lr", type=float, default=0.002, help="the peak rate, chosen by the rate check")
    p.add_argument("--rates", default="0.0015,0.002,0.003")
    p.add_argument("--warm", type=float, default=0.02, help="share of the updates over which the rate rises")
    p.add_argument("--warm-updates", type=int, default=0, help="the warm-up in updates instead (the rate check uses 50)")
    p.add_argument("--micro-pieces", type=int, default=65536, help="pieces each GPU works on at once (memory)")
    p.add_argument("--attention", default="cudnn", choices=["cudnn", "flash", "efficient"])
    p.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fused-ce", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--doc-mask-all", action="store_true", help="mask attention at document ends in every phase")
    p.add_argument("--bucket-mb", type=int, default=25)
    p.add_argument("--save-minutes", type=float, default=15)
    p.add_argument("--score-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--warn-ahead", type=int, default=2000, help="warn when a source has fewer updates of text ready")
    p.add_argument("--stop-after", type=int, default=0, help="the test run: save and stop after this many more updates")
    p.add_argument("--updates-limit", type=int, default=0, help="stop (and save) at this update number")
    p.add_argument("--price-per-hour", type=float, default=24.0, help="for the cost in the log (8 GPUs at about $3)")
    p.add_argument("--title", default="SlingerGPT 492M")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lines", type=int, default=10)
    p.add_argument("--allow-plan-change", action="store_true", help="carry on although the shares changed")
    # for the CPU tests
    p.add_argument("--size", default="492m", help="492m, or a small arch.CONFIGS size for tests (1m, 10m)")
    p.add_argument("--total-pieces", type=int, default=TOTAL_PIECES)
    p.add_argument("--per-update", type=int, default=PIECES_PER_UPDATE)
    p.add_argument("--windows", default="", help="test only: the five window lengths, e.g. 32,64,128,128,256")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--trace", action="store_true", help="test only: each GPU writes a fingerprint of its windows")
    args = p.parse_args()
    if args.what == "ratecheck":
        args.stop_after = args.stop_after or 200
        args.warm_updates = args.warm_updates or 50
    out, ckpt, _, logfile = paths(args)
    if args.what == "pause":
        out.mkdir(parents=True, exist_ok=True)
        Path(str(ckpt) + ".pause").touch()
        print("asked the training to save and stop")
    elif args.what == "status":
        lines = logfile.read_text(encoding="utf-8").splitlines() if logfile.exists() else ["no log yet"]
        print("\n".join(lines[-args.lines:]))
    else:
        launch(args)


if __name__ == "__main__":
    main()
