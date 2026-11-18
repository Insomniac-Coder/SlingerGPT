"""More training text for carrying SlingerGPT 1 pico on (the 19-hour plan of 15 November), in data/slinger_more/.

    python -P research/prepare_slinger_more.py [source ...]     all sources, or only the ones named

Each source reads its files in the first cut's order (research/prepare_slinger.py), skips the documents that cut
already used (data/slinger/manifest.json counts them), and carries on through the rest of the local files and the
14 files downloaded on 15 November (FineWeb-Edu 002-004, Cosmopedia v2 1-3, FineMath 4+ 3-6, Wikipedia 2-5).
Same list of 4,096 pieces, same filters, one document in every 100 held out. Simple Wikipedia and Orca-Math have
nothing new: the first cut used all of them.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import prepare_slinger as ps

DATA, MORE = ps.OUT, ps.OUT.parent / "slinger_more"
FIRST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["sources"]
LIMIT = {"fineweb_best": 2_000_000_000, "fineweb_good": 500_000_000, "cosmopedia": 1_300_000_000, "smoltalk": 2_000_000_000,
         "soda": 2_000_000_000, "wikipedia": 2_000_000_000, "finemath": 2_000_000_000}   # pieces to cut at most


def fineweb_good():
    """As the first cut: pages rated 3 scoring in the top 30% of those in the first two files (the same bar)."""
    first = sorted(ps.LOCAL.glob("fineweb_edu_*.parquet"))
    scores = np.concatenate([pq.read_table(f, columns=["score", "int_score"]).filter(
        pc.equal(pc.field("int_score"), 3))["score"].to_numpy() for f in first[:2]])
    bar = np.quantile(scores, 0.70)
    print(f"  fineweb_good: pages rated 3 with a score of at least {bar:.3f}", flush=True)
    return (r["text"] for r in ps.rows(first, ["text", "int_score", "score"]) if r["int_score"] == 3 and r["score"] >= bar)


def cosmopedia():
    """The first cut's order (Cosmopedia v2 file 0, then the local v1 files), then the new v2 files."""
    v2 = sorted(ps.NEW.glob("cosmopedia-v2/*.parquet"))
    return (r["text"] for r in ps.rows(v2[:1] + sorted(ps.LOCAL.glob("cosmo_*.parquet")) + v2[1:], ["text"]))


ps.SOURCES.update(fineweb_good=fineweb_good, cosmopedia=cosmopedia)


def after_first_cut(name, texts):
    """The source's documents from where the first cut stopped (it counted only non-empty ones)."""
    used = FIRST[name]["train"]["documents"] + FIRST[name]["val"]["documents"]
    return itertools.islice((t for t in texts() if t and t.strip()), used, None)


def main(names):
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    MORE.mkdir(parents=True, exist_ok=True)
    for name in names:
        plain = ps.SOURCES[name]
        ps.SOURCES[name] = lambda name=name, plain=plain: after_first_cut(name, plain)
        ps.OUT, ps.SPARE = MORE, 1.0   # cut() writes MORE/<name>/ and reports against the limit itself
        try:
            info = ps.cut(tok, name, LIMIT[name])
        finally:
            ps.SOURCES[name], ps.OUT = plain, DATA
        (MORE / name / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    # the manifest gathers every source's info.json, so several copies of this script can run side by side
    sources = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(MORE.glob("*/info.json"))}
    (MORE / "manifest.json").write_text(json.dumps({"name": "SlingerGPT 1 pico, more training text", "pieces in the list": 4096,
                                                    "end of text": 0, "sources": sources}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or list(LIMIT))
