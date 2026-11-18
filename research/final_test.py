"""The head-to-head test: the normal way of training against the new way, on the 10M model or the 30M model.

    python -P research/final_test.py rates [10m|30m]    learning rates tried first, so each way is compared at its best
    python -P research/final_test.py pick               10m only: the new way with the lookup memory switched on
    python -P research/final_test.py run [10m|30m]      the test itself
    python -P research/final_test.py report [10m|30m]   the tables

The 30M model is the same model made wider (29.5M weights in the body against 10.7M). Its runs take three times
as long, so it is trained from one starting point, not three; the three 10M starting points agreed within 1.5%.

Normal way: GPT-2 block, AdamW, the text as GPT-2's 50,257 word pieces, every piece scored at every position.
That is what gpt128.py does (lab.py size 10m-gpt2).
New way: Muon, rotary positions, QK-norm, SwiGLU, the text as 4,096 word pieces (lab.py size 10m-bpe4096).
Same model body, both compiled, both read 128 x 256 units per round, three starting points each.
Scores are wrongness per letter on the same held-out text (the score per piece divided by the letters per
piece), so the two are comparable. The new way is then trained again in a short run sized to end at the normal
way's final score, so "same quality in less time" is measured directly and not only read off a longer run.
The new way's long run is 2,700 rounds, which is one pass over the 297 MB of text both train on.
One run at a time, two and a half minutes of rest after each. A run is stopped if the GPU reaches 90 C, and
nothing starts while the GPU is busy.
"""
import json, math, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
LOG, MINE = HERE / "logs" / "lab.jsonl", HERE / "logs" / "final_test.log"
ROUND = 128 * 256   # units read per round
NEW = {"muon": True, "rope": True, "qknorm": True, "swiglu": True, "compiled": True}
PLANS = {   # per model body: normal way and new way as (size, settings, rounds), then the starting points
    # 10m: rate 0.002 is the best of `rates` (1.073 per letter; 0.001 gave 1.118, 0.004 gave 1.127);
    #      `pick`: lookup memory 0.906 in 123 s, without 0.910 in 118 s; a tie, kept in. 2,700 rounds = one pass over the text.
    "10m": (("10m-gpt2", {"lr": 0.002, "compiled": True}, 1000), ("10m-bpe4096", {**NEW, "engram": True, "lr": 0.002}, 2700), (0, 1, 2)),
    # 30m: the trained models are kept in models/ so compare_text.py can have them answer the same prompts
    #      `rates`, 400 rounds: normal way 0.0005 gave 1.188, 0.001 gave 1.140, 0.002 gave 1.187; new way 0.001 gave 1.034, 0.002 gave 0.991.
    #      700 normal rounds take about 417 s (0.595 s each); 1,700 new rounds take about the same (0.25 s each).
    "30m": (("30m-gpt2", {"lr": 0.001, "compiled": True, "save": "final_30m_normal"}, 700),
            ("30m-bpe4096", {**NEW, "engram": True, "lr": 0.002, "save": "final_30m_new_same_time"}, 1700), (0,)),
}
RATE_CHECK = {   # rounds, rates for the normal way, rates for the new way
    "10m": (600, (0.001, 0.002, 0.004), ()),
    "30m": (400, (0.0005, 0.001, 0.002), (0.001, 0.002)),
}
BODY = sys.argv[2] if len(sys.argv) > 2 else "10m"   # second argument: 10m (default) or 30m
NORMAL, NEW_WAY, SEEDS = PLANS[BODY]
PER = {s: json.loads((HERE.parent / "data" / "pieces" / s[4:] / "manifest.json").read_text())["letters per piece, val"]
       for s in (NORMAL[0], NEW_WAY[0])}   # letters per unit
REST, TOO_HOT = 150, 90   # seconds of rest after each run; a run is stopped at TOO_HOT (Celsius)
# The GPU reads about 55 C idle within a minute of any run, so its temperature cannot tell when the laptop has
# cooled; one minute of rest was not enough (87 C in the run after a nine-minute one), hence the fixed 150 s.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
mean = lambda v: sum(v) / len(v)


def records(stage, size, cfg=None, rounds=None):
    return [r for r in map(json.loads, LOG.open(encoding="utf-8")) if r["stage"] == f"{stage}@{size}" and "final" in r
            and cfg in (None, r["config"]) and rounds in (None, r["tokens"] // ROUND)]


def gpu():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return [float(v) for v in out.split(",")]


def one(stage, size, cfg, rounds):
    if records(stage, size, cfg, rounds):
        return
    temp, busy = gpu()
    if busy > 50:
        sys.exit(f"not starting: the GPU is {busy:.0f}% busy with something else")
    began, peak = time.time(), temp
    job = subprocess.Popen([sys.executable, "-P", str(HERE / "lab.py"), "custom", stage, "--size", size,
                            "--tokens", str(rounds * ROUND), "--configs", json.dumps([cfg])])
    while job.poll() is None:
        time.sleep(10)
        peak = max(peak, gpu()[0])
        if peak >= TOO_HOT:
            job.kill()
    with MINE.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {stage} {size} {rounds} rounds {json.dumps(cfg)}: {time.time() - began:.0f} s "
                f"start to finish, exit {job.returncode}, started at {temp:.0f} C, hottest {peak:.0f} C\n")
    if peak >= TOO_HOT:
        sys.exit(f"stopped: the GPU reached {peak:.0f} C")
    if not records(stage, size, cfg, rounds):
        sys.exit(f"this run did not finish, stopping: {stage} {size} {cfg}")
    time.sleep(REST)


def reach(curve, target, per=1.0, x=2):
    """Where the per-letter score first gets down to `target`, on the straight line between two progress marks.
    x=2 gives seconds of training, x=0 gives units read."""
    prev = None
    for mark in curve:
        v = mark[1] / per
        if v <= target:
            return mark[x] if prev is None else prev[0] + (mark[x] - prev[0]) * (prev[1] - target) / (prev[1] - v)
        prev = (mark[x], v)
    return None


def at(curve, seconds, per=1.0):
    """The per-letter score after this many seconds of training."""
    prev = None
    for _, loss, t in curve:
        if t >= seconds:
            return loss / per if prev is None else prev[1] + (loss / per - prev[1]) * (seconds - prev[0]) / (t - prev[0])
        prev = (t, loss / per)
    return prev[1]


def run():
    for seed in SEEDS:   # the two take turns, so both see the same machine conditions
        for size, cfg, rounds in (NORMAL, NEW_WAY):
            one("final", size, {**cfg, "seed": seed}, rounds)
    target = mean([r["final"] for r in records("final", NORMAL[0], rounds=NORMAL[2])]) / PER[NORMAL[0]]
    size, cfg, rounds = NEW_WAY
    needed = max(reach(r["curve"], target, PER[size], x=0) for r in records("final", size, rounds=rounds))
    for seed in SEEDS:   # the slowest starting point's crossing, rounded up to ten rounds
        short = {**cfg, "seed": seed, **({"save": cfg["save"].replace("same_time", "same_loss")} if "save" in cfg else {})}
        one("final", size, short, math.ceil(needed / ROUND / 10) * 10)


def table(runs, per):
    print(f"{'settings':40} {'rounds':>6} {'per letter':>10} {'seconds':>8}")
    for r in runs:
        name = "+".join(k if v is True else f"{k}={v}" for k, v in r["config"].items() if k not in NEW) or "as in pieces.json"
        print(f"{name:40} {r['tokens'] // ROUND:6} {r['final'] / per:10.4f} {r['seconds']:8.1f}")


def report():
    (sa, _, ra), (sb, _, rb) = NORMAL, NEW_WAY
    pa, pb = PER[sa], PER[sb]
    a, b = records("final", sa, rounds=ra), records("final", sb, rounds=rb)
    short = [r for r in records("final", sb) if r["tokens"] // ROUND != rb]
    for name, runs, p in (("normal way", a, pa), ("new way", b, pb), ("new way, short run", short, pb)):
        for r in runs:
            print(f"{name:20} start {r['config'].get('seed', 0)}  {r['tokens'] // ROUND:5} rounds  {r['seconds']:6.1f} s  per letter {r['final'] / p:.4f}")
    if not (a and b):
        return
    seconds, target = mean([r["seconds"] for r in a]), mean([r["final"] for r in a]) / pa
    both = min(seconds, mean([r["seconds"] for r in b]))
    print(f"\nSame time ({both:.0f} s): normal way {mean([at(r['curve'], both, pa) for r in a]):.4f} per letter, "
          f"new way {mean([at(r['curve'], both, pb) for r in b]):.4f}")
    if short:
        took = mean([r["seconds"] for r in short])
        print(f"Same quality, measured directly: normal way {seconds:.0f} s to end at {target:.4f}; new way {took:.1f} s to end at "
              f"{mean([r['final'] / pb for r in short]):.4f} (worst start {max(r['final'] / pb for r in short):.4f}) -> {seconds / took:.1f}x")
    print("\nBy quality level (the normal way's score at each quarter of its run):")
    for i in (4, 9, 14, 19):
        level, t_a = mean([r["curve"][i][1] for r in a]) / pa, mean([r["curve"][i][2] for r in a])
        t_b = mean([reach(r["curve"], level, pb) for r in b])
        print(f"  {level:.4f} per letter: normal way {t_a:6.1f} s, new way {t_b:5.1f} s -> {t_a / t_b:.1f}x")


if __name__ == "__main__":
    assert abs(reach([[0, 2.0, 10.0], [0, 1.0, 20.0]], 1.5) - 15.0) < 1e-9 and abs(at([[0, 2.0, 10.0], [0, 1.0, 20.0]], 15.0) - 1.5) < 1e-9
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "rates":
        rounds, normal_rates, new_rates = RATE_CHECK[BODY]
        for lr in normal_rates:
            one("rate", NORMAL[0], {"lr": lr, "compiled": True}, rounds)
        for lr in new_rates:
            one("rate", NEW_WAY[0], {**NEW, "engram": True, "lr": lr}, rounds)
        for size in (NORMAL[0], NEW_WAY[0]):
            table(records("rate", size), PER[size])
    elif what == "pick":
        one("pick", NEW_WAY[0], {**NEW, "engram": True, "lr": 0.002}, 1000)
        table(records("pieces", NEW_WAY[0], rounds=1000) + records("pick", NEW_WAY[0]), PER[NEW_WAY[0]])
    elif what == "run":
        run()
        report()
    else:
        report()
