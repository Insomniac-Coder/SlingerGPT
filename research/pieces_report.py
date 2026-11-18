"""Letters against word pieces: the same 10M model on the same text, compared by wrongness per letter.

    python -P research/pieces_report.py [stage]          (default stage: pieces)

A model that reads word pieces is scored per piece, so its score is divided by the letters per piece of the
held-out text. That gives wrongness per letter, which is comparable across every version of the text.
Also shown: the score each version had reached at the same number of seconds, read off its progress marks.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ORDER = ["10m", "10m-bpe1024", "10m-bpe4096", "10m-bpe16384", "10m-gpt2", "10m-phrase65k"]
NAMES = {"10m": "letters", "10m-bpe1024": "1,024 pieces", "10m-bpe4096": "4,096 pieces", "10m-bpe16384": "16,384 pieces",
         "10m-gpt2": "GPT-2 pieces", "10m-phrase65k": "phrase pieces"}


def letters_per_unit(size):
    if size == "10m":
        return 1.0
    return json.loads((ROOT / "data" / "pieces" / size[4:] / "manifest.json").read_text())["letters per piece, val"]


def at_seconds(curve, t):
    """Loss per unit at t seconds of training, on the straight line between the two nearest progress marks."""
    before = [c for c in curve if c[2] <= t]
    after = [c for c in curve if c[2] > t]
    if not before:
        return None
    if not after:
        return before[-1][1]
    a, b = before[-1], after[0]
    return a[1] + (b[1] - a[1]) * (t - a[2]) / (b[2] - a[2])


def main(stage):
    runs = {}
    for line in (HERE / "logs" / "lab.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if "final" in r and r["stage"].startswith(stage + "@"):
            runs.setdefault(r["stage"].split("@")[1], []).append(r)
    rows = [(s, max(runs[s], key=lambda r: r["tokens"])) for s in ORDER if s in runs]
    shortest = min(r["seconds"] for _, r in rows)
    print(f"{'version':15} {'letters/unit':>12} {'weights':>9} {'rounds':>7} {'seconds':>8} {'letters/s':>10} {'per unit':>9} {'PER LETTER':>11} "
          f"{'per letter at ' + str(round(shortest)) + ' s':>22}")
    for s, r in rows:
        u, rounds = letters_per_unit(s), r["tokens"] // 32768
        early = at_seconds(r["curve"], shortest)
        print(f"{NAMES[s]:15} {u:12.2f} {r['parameters_M']:8.1f}M {rounds:7} {r['seconds']:8.1f} {r['tokens'] * u / r['seconds']:10,.0f} "
              f"{r['final']:9.4f} {r['final'] / u:11.4f} {early / u if early else float('nan'):22.4f}")
    for s in ORDER:   # every run of each version, for matched-time comparisons
        for r in sorted(runs.get(s, []), key=lambda r: r["tokens"]):
            if len(runs[s]) > 1:
                print(f"   {NAMES[s]:15} {r['tokens'] // 32768:6} rounds  {r['seconds']:7.1f} s  per letter {r['final'] / letters_per_unit(s):.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pieces")
