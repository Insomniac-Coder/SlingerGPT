"""Equal-time learning test on the 124M model: the normal way, the new way, and two faster-reading versions of it.

    python -P research/test_124m.py run       about 50 minutes of GPU, one run at a time with rests
    python -P research/test_124m.py report

Every run trains for about 8 minutes; its number of weight updates is set from the timings in
research/speed_124m.py. One update = 4 slices of 16 x 1,024 pieces, as in gpt128.py. Scores are wrongness per
letter on held-out text, read off each run's 20 progress marks, so the four can be compared second for second.
The faster versions keep the new way and change only the list of pieces and how the last step scores them:
"sampled" scores the right piece and 4,096 random others instead of the whole list (lab.py head="sampled").
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]   # final_test reads its own arguments when imported
import final_test as ft

NEW = {**ft.NEW, "engram": True, "lr": 0.002}
RUNS = [   # name, lab.py size, settings, weight updates (about 8 minutes each)
    ("normal way", "124m-gpt2", {"lr": 0.0006, "compiled": True}, 250),   # gpt128.py's own learning rate
    ("new way", "124m-bpe4096", NEW, 390),
    ("new way, GPT-2's list, sampled scoring", "124m-gpt2", {**NEW, "head": "sampled"}, 390),
    ("new way, phrase list, sampled scoring", "124m-phrase65k", {**NEW, "head": "sampled"}, 390),
    # gpt128.py's rate suits a 17-hour run; a short run may do better with a higher one, which would make the
    # gains above look larger than they are. The gains are counted against whichever normal run ends better.
    ("normal way, learning rate 0.0012", "124m-gpt2", {"lr": 0.0012, "compiled": True}, 250),
]
PER = {size: json.loads((HERE.parent / "data" / "pieces" / size[5:] / "manifest.json").read_text())["letters per piece, val"]
       for _, size, _, _ in RUNS}
TWO = 2   # final_test counts in rounds of 32,768 pieces; one update here is 65,536


def run():
    ft.REST = 60
    for _, size, cfg, _ in (RUNS[3], RUNS[0]):   # two updates each, so a mistake shows in minutes, not after a long run
        ft.one("smoke124", size, cfg, 2 * TWO)
    ft.REST = 150
    for _, size, cfg, n in RUNS:
        ft.one("t124", size, cfg, n * TWO)


def report():
    rows = [(name, size, ft.records("t124", size, cfg, n * TWO)) for name, size, cfg, n in RUNS]
    rows = [(name, size, r[0]) for name, size, r in rows if r]
    if not rows:
        return print("no runs yet")
    shortest = min(r["seconds"] for _, _, r in rows)
    normal = min((r for name, _, r in rows if name.startswith("normal way")), key=lambda r: r["final"], default=None)
    target = normal["final"] / PER["124m-gpt2"] if normal else None
    print(f"{'':42} {'updates':>7} {'seconds':>7} {'per letter':>10} {f'per letter at {shortest:.0f} s':>18} "
          f"{'s to normal end':>15} {'gain':>6}")
    for name, size, r in rows:
        p = PER[size]
        got = ft.reach(r["curve"], target, p) if target else None
        print(f"{name:42} {r['tokens'] // 65536:7} {r['seconds']:7.1f} {r['final'] / p:10.4f} {ft.at(r['curve'], shortest, p):18.4f} "
              f"{got if got else float('nan'):15.1f} {normal['seconds'] / got if got else float('nan'):5.2f}x")


if __name__ == "__main__":
    run() if ARGS[:1] == ["run"] else None
    report()
