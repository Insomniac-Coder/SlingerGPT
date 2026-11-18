"""Builds research/492m/mix_492m.json, the per-phase shares the trainer reads, from the numbers in data_mix.md.

    python research/492m/make_mix.py                       the draft, from data_mix.md's estimates
    python research/492m/make_mix.py --measured <data>     after round 1: checks each source's supply (passes)

Phases (TRAINING_PLAN.md section 2): main 0-70B, anneal 70-78B (the best material only), long 78-80B (60% long
documents, 40% the anneal mix). A source with a best slice (FineWeb-Edu >= 4, Stack Exchange top answers...) is
two sources: the whole source for the main phase, and its best slice for the anneal. Pages of the best slice are in
both, so they are read up to twice, as data_mix.md says.

Main-phase pieces of a source = its total in data_mix.md's table - what the anneal reads - what the long phase reads.
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).parent
MAIN_B, ANNEAL_B, LONG_B = 70.0, 8.0, 2.0
SHORT_IN_LONG = 0.4

# source: (billions of pieces over the whole run in data_mix.md, the anneal's best slice or None, its anneal %)
ROWS = {
    "fineweb_edu":      (19.70, "fineweb_edu_4", 17.0),
    "dclm_edu":         (11.90, "dclm_edu_4", 8.0),
    "cosmopedia":       (7.44, "cosmopedia", 5.0),
    "nemotron_qa":      (1.93, "nemotron_qa", 2.0),
    "stack_python":     (3.35, "stack_python_4", 7.0),
    "stack_other":      (4.80, "stack_other_4", 6.0),
    "stack_html":       (0.21, None, 0.0),
    "kaggle":           (0.62, "kaggle", 1.5),
    "opencoder":        (0.31, "opencoder", 3.5),
    "se_tech":          (2.45, "se_tech_top", 4.0),
    "essential_cs":     (1.58, "essential_cs", 2.0),
    "docs":             (0.39, "docs", 1.0),
    "finepdfs_cs":      (0.61, "finepdfs_cs", 1.0),
    "finemath_4":       (2.72, "finemath_4", 7.0),
    "finemath_3":       (2.10, None, 0.0),
    "infiwebmath":      (1.05, None, 0.0),
    "se_math":          (0.81, "se_math_top", 2.0),
    "worked_maths":     (0.95, "worked_maths", 6.0),
    "finewiki":         (5.41, "finewiki", 7.5),      # +0.02B from Simple Wikipedia (owner, 18 November)
    "simplewiki":       (0.32, "simplewiki", 1.5),    # 0.40% (not 0.43%): read at most ~4 times (owner, 18 November)
    "gutenberg":        (2.07, "gutenberg", 1.0),
    "doab":             (2.20, "doab", 2.0),
    "se_other":         (2.01, "se_other_top", 3.0),
    "arxiv_abstracts":  (0.79, "arxiv_abstracts", 1.0),
    "pes2o":            (1.24, "pes2o", 1.0),
    "smoltalk":         (1.28, "smoltalk", 5.0),
    "ultrachat":        (0.41, "ultrachat", 1.5),
    "soda":             (0.41, "soda", 1.5),
    "wildchat":         (0.53, "wildchat", 2.0),
    "code_repos":       (0.24, None, 0.0),
    "arxiv_papers":     (0.20, None, 0.0),
}
# the long phase's long documents (60% of its 2B): long-document source, % of the 2B, the row its pieces count against
LONG = {
    "gutenberg_long": (15.0, "gutenberg"), "doab_long": (10.0, "doab"), "arxiv_papers": (10.0, "arxiv_papers"),
    "pes2o_long": (5.0, "pes2o"), "code_repos": (12.0, "code_repos"), "finepdfs_long": (5.0, "finepdfs_cs"),
    "finewiki_long": (2.0, "finewiki"), "docs_long": (1.0, "docs"),
}


def build():
    anneal = {best: pct for _, best, pct in ROWS.values() if best}
    assert abs(sum(anneal.values()) - 100) < 1e-6, sum(anneal.values())
    long_docs = {k: pct for k, (pct, _) in LONG.items()}
    assert abs(sum(long_docs.values()) - 100 * (1 - SHORT_IN_LONG)) < 1e-6
    long = {**{k: v * SHORT_IN_LONG for k, v in anneal.items()}, **long_docs}
    main = {}
    for name, (total, best, pct) in ROWS.items():
        later = (ANNEAL_B + LONG_B * SHORT_IN_LONG) * pct / 100 if best else 0.0
        later += sum(LONG_B * p / 100 for p, row in LONG.values() if row == name)
        left = total - later
        assert left > -0.005, f"{name}: data_mix.md gives {total}B but the anneal and long phases read {later:.3f}B"
        if left > 0.005:
            main[name] = round(100 * left / MAIN_B, 3)
    scale = 100 / sum(main.values())   # data_mix.md's rows are rounded; spread the rounding evenly
    main = {k: round(v * scale, 3) for k, v in main.items()}
    return {"status": "Data list APPROVED by the owner on 18 November 2026 (free model, no commercial use; Stack Exchange's "
                      "older copy; Nemotron Diverse QA; Simple Wikipedia trimmed to 0.40%). Round 1's measured piece counts "
                      "may still adjust shares before the run starts; the chat behaviour filters wait for their own yes.",
            "phases": {"main": main, "anneal": anneal, "long": {k: round(v, 3) for k, v in long.items()}},
            "billions": {"main": MAIN_B, "anneal": ANNEAL_B, "long": LONG_B}}


def passes(mix, data):
    """How many times each source would be read, given what prepare_492m.py has made of it so far."""
    manifest = json.loads((Path(data) / "manifest.json").read_text(encoding="utf-8"))
    need = {}
    for phase, shares in mix["phases"].items():
        for k, v in shares.items():
            need[k] = need.get(k, 0.0) + mix["billions"][phase] * 1e9 * v / 100
    for k in sorted(need):
        info = manifest["sources"].get(k, {})
        have = sum(f["pieces"] for f in info.get("files", []))
        state = "complete" if info.get("complete") else "still being prepared"
        print(f"{k:18s} needs {need[k] / 1e9:7.3f}B, has {have / 1e9:7.3f}B ({state})"
              + (f": {need[k] / have:.2f} passes" if have else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--measured", default="")
    a = p.parse_args()
    mix = build()
    if a.measured:
        passes(mix, a.measured)
    else:
        (HERE / "mix_492m.json").write_text(json.dumps(mix, indent=1) + "\n", encoding="utf-8")
        for phase, shares in mix["phases"].items():
            print(f"{phase}: {len(shares)} sources, shares add up to {sum(shares.values()):.2f}%")
        print("main phase:", ", ".join(f"{k} {v}" for k, v in sorted(mix["phases"]["main"].items(), key=lambda kv: -kv[1])))
