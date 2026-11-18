"""Builds data/corpus/mix_v2.txt for carrying the 50M model on (mix50m.pt -> mix50m_v2.pt): about 2 billion
characters, 80% new text and 20% from the text it first learned from, so it learns more without drifting
away from what it can already do.

    python build_corpus_v2.py

New text, in data/raw_v2/:
  fineweb_edu_000.parquet   FineWeb-Edu, sample-10BT part 000 (ODC-By): educational web pages
  tinystories_part3.txt     TinyStories (CDLA-Sharing-1.0), bytes 700M-1100M of TinyStories-train.txt
  gutenberg_en_2/3.parquet  Project Gutenberg books not used before (public domain)
Also SODA dialogues from data/raw/ (a random pick, so about a third were in the first mix), and a slice of
data/corpus/mix_big.txt, the first mix.
"""
import random, sys, time
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))   # find build_corpus.py next to this file
from build_corpus import HERE, clean, stories, books, dialogues

RAW2 = HERE / "data" / "raw_v2"
OLD, OUT = HERE / "data" / "corpus" / "mix_big.txt", HERE / "data" / "corpus" / "mix_v2.txt"
BUDGET = {"knowledge": 800_000_000, "stories": 300_000_000, "books": 250_000_000, "dialogues": 250_000_000}
REPLAY = 400_000_000   # characters of the first mix
BLOCK = 50_000         # the replayed text is shuffled in blocks of about this size, cut at paragraph breaks


def fineweb(enough):
    """FineWeb-Edu pages until there is comfortably more than the budget (the file holds ~3B characters)."""
    out, size = [], 0
    for batch in pq.ParquetFile(RAW2 / "fineweb_edu_000.parquet").iter_batches(batch_size=4096, columns=["text"]):
        for text in batch.column("text").to_pylist():
            text = clean(text)
            if len(text) >= 200:
                out.append(text)
                size += len(text)
        if size >= enough:
            break
    return out


def replay():
    old = OLD.read_text(encoding="ascii")[:REPLAY]   # the first mix is shuffled by document, so its start is a fair sample
    blocks, at = [], 0
    while at < len(old):
        cut = old.find("\n\n", at + BLOCK)
        cut = len(old) if cut < 0 else cut
        blocks.append(old[at:cut].strip())
        at = cut + 2
    return blocks


def main():
    rng = random.Random(2026)
    chosen = []
    sources = (("knowledge", lambda: fineweb(BUDGET["knowledge"] * 5 // 4)),
               ("stories", lambda: stories(RAW2, all_start_mid_story=True)),
               ("books", lambda: books(RAW2)),
               ("dialogues", dialogues))
    for kind, read in sources:
        began = time.time()
        docs = [d for d in read() if d]
        rng.shuffle(docs)
        taken, size = [], 0
        for d in docs:
            if size >= BUDGET[kind]:
                break
            taken.append(d)
            size += len(d) + 2
        chosen += taken
        print(f"{kind:10} {len(taken):>9,} of {len(docs):,} documents, {size / 1e6:6.1f}M characters  {time.time() - began:.0f} s")
    old = replay()
    chosen += old
    print(f"{'first mix':10} {len(old):>9,} blocks, {sum(len(b) + 2 for b in old) / 1e6:6.1f}M characters")
    rng.shuffle(chosen)
    text = "\n\n".join(chosen) + "\n"
    OUT.write_text(text, encoding="ascii", newline="\n")
    print(f"\n{len(text) / 1e6:.1f}M characters, {len(set(text))} different ones -> {OUT}")


if __name__ == "__main__":
    main()
