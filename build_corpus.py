"""Builds data/corpus/mix.txt, the training text for the 10M model: stories, classic books,
dialogues and Simple English Wikipedia, cleaned to plain characters and shuffled together one
document at a time (so the unseen 10% that training checks against holds all four kinds).

    python build_corpus.py                       the 860M-character mix (data/corpus/mix.txt)
    python build_corpus.py --stories 600 --books 600 --dialogues 400 --wikipedia 265 --out data/corpus/mix_big.txt

Sources, all in data/raw/:
  tinystories_part*.txt   TinyStories (CDLA-Sharing-1.0): bytes 0-380M and 380M-700M of TinyStories-train.txt
  gutenberg_en_*.parquet  Project Gutenberg books (public domain), via manu/project_gutenberg
  soda_*.parquet          SODA dialogues (CC BY 4.0), via allenai/soda
  simplewiki.parquet      Simple English Wikipedia, 2023-11-01 (CC BY-SA 3.0), via wikimedia/wikipedia
"""
import argparse, random, re, sys, time, unicodedata
from pathlib import Path
import pyarrow.parquet as pq

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
MAX_BOOK = 600_000   # cap any one book, so a dictionary or a Bible cannot crowd out the rest
SWAPS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"', "‟": '"',
                       "–": "-", "—": "--", "―": "--", "…": "...", " ": " ", "\t": " ", "\r": ""})
NOT_PLAIN = re.compile(r"[^\n -~]")   # anything other than printable ASCII and line breaks
START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean(text):
    """Curly quotes and dashes to plain ones, accents dropped (é -> e), anything else non-ASCII removed."""
    text = NOT_PLAIN.sub("", unicodedata.normalize("NFKD", text.translate(SWAPS)))
    text = re.sub(r" +\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def stories(raw=RAW, all_start_mid_story=False):
    out = []
    for n, path in enumerate(sorted(raw.glob("tinystories_part*.txt"))):
        parts = path.read_text(encoding="utf-8", errors="ignore").split("<|endoftext|>")
        if len(parts) < 1000:
            raise SystemExit(f"{path.name}: no <|endoftext|> separators found; the file is not what was expected")
        # each download stops mid-story, and every part after the first also starts mid-story
        out += [clean(p) for p in parts[1 if n or all_start_mid_story else 0:-1]]
    return out


def books(raw=RAW):
    out = []
    for path in sorted(raw.glob("gutenberg_en_*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=["text"]):
            for raw in batch.column("text").to_pylist():
                start, end = START.search(raw), END.search(raw)
                if not start or not end or end.start() <= start.end():
                    continue   # no clear licence markers: skip rather than train on the licence
                body = raw[start.end():end.start()].replace("\r", "")
                text = clean(body)
                if len(text) < 0.9 * len(body.strip()):
                    continue   # mostly not in the Latin alphabet
                out.append(text[:MAX_BOOK])
    return out


def dialogues():
    out = []
    for path in sorted(RAW.glob("soda_*.parquet")):
        for row in pq.read_table(path, columns=["narrative", "dialogue", "speakers"]).to_pylist():
            lines = [f"{who}: {said}" for who, said in zip(row["speakers"], row["dialogue"])]
            out.append(clean(row["narrative"] + "\n\n" + "\n".join(lines)))
    return out


def wikipedia():
    out = []
    for batch in pq.ParquetFile(RAW / "simplewiki.parquet").iter_batches(batch_size=4096, columns=["title", "text"]):
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            if len(text) >= 200:   # skip stubs
                out.append(clean(f"{title}\n\n{text}"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stories", type=float, default=340, help="millions of characters")
    p.add_argument("--books", type=float, default=220)
    p.add_argument("--dialogues", type=float, default=150)
    p.add_argument("--wikipedia", type=float, default=150)
    p.add_argument("--out", default=str(HERE / "data" / "corpus" / "mix.txt"))
    args = p.parse_args()
    budget = {kind: int(getattr(args, kind) * 1e6) for kind in ("stories", "books", "dialogues", "wikipedia")}
    out_path = Path(args.out)
    rng = random.Random(1337)
    chosen = []
    for kind, read in (("stories", stories), ("books", books), ("dialogues", dialogues), ("wikipedia", wikipedia)):
        began = time.time()
        docs = [d for d in read() if d]
        rng.shuffle(docs)
        taken, size = [], 0
        for d in docs:
            if size >= budget[kind]:
                break
            taken.append(d)
            size += len(d) + 2
        chosen += taken
        available = sum(len(d) for d in docs)
        print(f"{kind:10} {len(taken):>9,} of {len(docs):,} documents, {size / 1e6:6.1f}M characters "
              f"(of {available / 1e6:,.0f}M available)  {time.time() - began:.0f} s")
    rng.shuffle(chosen)
    text = "\n\n".join(chosen) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="ascii", newline="\n")
    print(f"\n{len(text) / 1e6:.1f}M characters, {len(set(text))} different ones -> {out_path}")


if __name__ == "__main__":
    main()
