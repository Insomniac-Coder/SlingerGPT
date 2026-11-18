"""Is the fix copy at least as good as SlingerGPT 1 pico on every test? (owner, 16 November: "keep it in a way that it only
gets better or remains the same")

    python -P research/fix_gate.py [candidate]       (default slingergpt1_pico_fix)

Reads the test results of both models (research/test_slinger.py run and extra) and checks each bar. Prints every
check and the verdict: ADOPT only if all pass, otherwise KEEP (the current model stays).
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
CANDIDATE = sys.argv[1] if len(sys.argv) > 1 else "slingergpt1_pico_fix"
BASELINE = sys.argv[2] if len(sys.argv) > 2 else "slingergpt1_pico_final"   # the model it is compared with
sys.argv = sys.argv[:1]
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import facts_quiz as fq
import report_card as rc


def numbers(name, extra=None):
    """Every measure for one model; `extra` reads another extra-results file (e.g. one kept from before the search fix).
    Measures from tests the model never had (broad questions, no real answer) are left out."""
    t = json.loads((HERE / "logs" / f"slinger_test_{name}.json").read_text(encoding="utf-8"))
    x = json.loads((HERE / "logs" / (extra or f"slinger_extra_{name}.json")).read_text(encoding="utf-8"))
    kind = lambda key, v, part="lookup": sum(r[key] == v for r in x[part])
    out = {}
    if "nonsense" in x:
        out["no real answer: made up"] = sum(not r["declined"] for r in x["nonsense"])
    if "broad" in x:
        out.update({"broad questions with lookup: right": kind("with_kind", "right", "broad"),
                    "broad questions with lookup: wrong": kind("with_kind", "wrong", "broad"),
                    "broad questions without lookup: wrong": kind("without_kind", "wrong", "broad")})
    return out | {
        "held-out wrongness per letter": sum(h["wrongness"] for h in t["held_out"].values()) / sum(h["letters"] for h in t["held_out"].values()),
        "quiz picking": sum(q["choice"] for q in t["quiz"]),
        "quiz own words": sum(q["free"] for q in t["quiz"]),
        "quiz wrong answers stated": sum(rc.own_words(i, q["said"]).startswith("named") for i, q in zip(fq.QUIZ, t["quiz"])),
        "sums right": sum(r["right"] for r in x["arithmetic"]),
        "who it is": sum(r["right"] for r in x["identity"]),
        "look it up: right": kind("with_kind", "right"),
        "look it up: wrong": kind("with_kind", "wrong"),
        "quiz with passage: picking": sum(q["choice"] for q in x["quiz_with_passage"]),
        "quiz with passage: own words": sum(q["free"] for q in x["quiz_with_passage"]),
        "without lookup: wrong": kind("without_kind", "wrong"),
        "without lookup: right": kind("without_kind", "right"),
        "without lookup: 'doesn't say' with no passage": kind("without_kind", "not said"),
    }


LOWER_IS_BETTER = {"held-out wrongness per letter", "quiz wrong answers stated", "look it up: wrong", "without lookup: wrong",
                   "without lookup: 'doesn't say' with no passage", "broad questions with lookup: wrong",
                   "broad questions without lookup: wrong", "no real answer: made up"}
INFORMATION_ONLY = {"without lookup: right", "without lookup: 'doesn't say' with no passage"}   # what the fix aims at; not a bar


if __name__ == "__main__":
    now, fix = numbers(BASELINE), numbers(CANDIDATE)
    failed = []
    for k in [k for k in now if k in fix]:   # measures both were tested on
        a, b = now[k], fix[k]
        better = b < a if k in LOWER_IS_BETTER else b > a
        same = abs(b - a) < 1e-9 if isinstance(a, float) else b == a
        ok = better or same
        tag = "info" if k in INFORMATION_ONLY else ("pass" if ok else "FAIL")
        if tag == "FAIL":
            failed.append(k)
        show = (lambda v: f"{v:.4f}") if isinstance(a, float) else str
        print(f"{tag:4s}  {k:48s} now {show(a):>7s}   fix {show(b):>7s}   {'better' if better else 'same' if same else 'worse'}")
    print("\nVERDICT:", "ADOPT the fix (nothing got worse)" if not failed else f"KEEP the current model (worse on: {', '.join(failed)})")
