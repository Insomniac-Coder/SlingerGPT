"""Training text for SlingerGPT 1 pico: the best of the local text plus five Hugging Face downloads, cut into pieces.

    python -P research/prepare_slinger.py

Reads data/raw128/ (FineWeb-Edu, Cosmopedia v1, English Wikipedia, SODA) and data/raw_slinger/ (smol-smoltalk,
FineMath 4+, Cosmopedia v2, Orca-Math, Simple English Wikipedia). Each source becomes its own pair of files in
data/slinger/<source>/ (train.bin, val.bin: 16-bit piece numbers), so the trainer can choose how much of each
to read, and change the mix for the last fifth of the training. One document in every 100 is held out.
Every document ends with <|endoftext|>. Conversations are written as "User: ..." / "Assistant: ..." lines.

The list of pieces (data/slinger/tokenizer.json) is built the same way as Change 6: 4,096 pieces learned from a
sample of this text, in the shares the training will read it, with <|endoftext|> as piece 0.
"""
import json, random, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).parent
LOCAL, NEW, OUT = HERE.parent / "data" / "raw128", HERE.parent / "data" / "raw_slinger", HERE.parent / "data" / "slinger"
END = "<|endoftext|>"
TOTAL = 2_000_000_000   # pieces the training reads (about 10 hours)
SHARE = {               # share of the training's pieces, over the whole run (the trainer shifts it for the last fifth)
    "fineweb_best": 0.13, "fineweb_good": 0.176, "cosmopedia": 0.222, "smoltalk": 0.112, "soda": 0.058,
    "wikipedia": 0.106, "simplewiki": 0.044, "finemath": 0.112, "orca_math": 0.04}
SPARE = 1.25            # cut this much more than the training needs, so less of it is read twice
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rows(files, columns, batch=2048):
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def fineweb(best):
    """FineWeb-Edu pages: rated 4 or 5 (best), or the highest-rated pages among those rated 3 (good)."""
    files = sorted(LOCAL.glob("fineweb_edu_*.parquet"))
    if best:
        return (r["text"] for r in rows(files, ["text", "int_score"]) if r["int_score"] >= 4)
    scores = np.concatenate([pq.read_table(f, columns=["score", "int_score"]).filter(
        pc.equal(pc.field("int_score"), 3))["score"].to_numpy() for f in files])
    cut = np.quantile(scores, 0.70)   # the top 30% of the pages rated 3
    print(f"  fineweb_good: pages rated 3 with a score of at least {cut:.3f}", flush=True)
    return (r["text"] for r in rows(files, ["text", "int_score", "score"]) if r["int_score"] == 3 and r["score"] >= cut)


def chat(messages):
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    return "\n".join(f"{names.get(m['role'], m['role'].title())}: {m['content'].strip()}" for m in messages)


SOURCES = {
    "fineweb_best": lambda: fineweb(True),
    "fineweb_good": lambda: fineweb(False),
    "cosmopedia": lambda: (r["text"] for r in rows(sorted(NEW.glob("cosmopedia-v2/*.parquet")) + sorted(LOCAL.glob("cosmo_*.parquet")), ["text"])),
    "smoltalk": lambda: (chat(r["messages"]) for r in rows(sorted(NEW.glob("smol-smoltalk/*.parquet")), ["messages"])),
    "soda": lambda: (r["narrative"] + "\n\n" + "\n".join(f"{w}: {s}" for w, s in zip(r["speakers"], r["dialogue"]))
                     for r in rows([LOCAL / "soda_train.parquet"], ["narrative", "dialogue", "speakers"])),
    "wikipedia": lambda: (f"{r['title']}\n\n{r['text']}" for r in rows(sorted(LOCAL.glob("wiki_en_*.parquet")), ["title", "text"])),
    "simplewiki": lambda: (f"{r['title']}\n\n{r['text']}" for r in rows(sorted(NEW.glob("simplewiki/*.parquet")), ["title", "text"])),
    "finemath": lambda: (r["text"] for r in rows(sorted(NEW.glob("finemath/*.parquet")), ["text"])),
    "orca_math": lambda: (chat([{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}])
                          for r in rows(sorted(NEW.glob("orca-math/*.parquet")), ["question", "answer"])),
}


def piece_list():
    """4,096 pieces learned from about 120 MB of the text, in the shares the training reads it."""
    sample = []
    for name, share in SHARE.items():
        want, got = 120e6 * share, 0
        for text in SOURCES[name]():
            sample.append(text)
            got += len(text)
            if got >= want:
                break
    random.Random(0).shuffle(sample)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(sample, trainers.BpeTrainer(vocab_size=4096, special_tokens=[END], show_progress=False,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    assert tok.token_to_id(END) == 0
    return tok


def cut(tok, name, limit):
    """Writes data/slinger/<name>/train.bin and val.bin; stops once the training file holds `limit` pieces."""
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}   # documents, pieces, letters (bytes of text)
    batch, began = [], time.time()
    with open(folder / "train.bin", "wb") as train, open(folder / "val.bin", "wb") as val:
        def flush():
            for i, e in enumerate(tok.encode_batch([t + END for t in batch])):
                part = "val" if (counts["train"][0] + counts["val"][0]) % 100 == 99 else "train"
                np.asarray(e.ids, dtype=np.uint16).tofile(train if part == "train" else val)
                counts[part][0] += 1
                counts[part][1] += len(e.ids)
                counts[part][2] += len(batch[i].encode("utf-8"))
            batch.clear()
        for text in SOURCES[name]():
            if text and text.strip():
                batch.append(text)
            if len(batch) == 4096:
                flush()
                if counts["train"][1] >= limit:
                    break
        if batch:
            flush()
    info = {part: {"documents": c[0], "pieces": c[1], "letters": c[2], "letters per piece": round(c[2] / max(1, c[1]), 4)}
            for part, c in counts.items()}
    print(f"  {name}: {info['train']['pieces'] / 1e6:,.0f}M pieces to train on (needs {limit / SPARE / 1e6:,.0f}M), "
          f"{info['train']['letters per piece']} letters per piece, {info['val']['documents']:,} documents held out "
          f"({time.time() - began:.0f} s)", flush=True)
    return info


def main():
    began = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = piece_list()
    tok.save(str(OUT / "tokenizer.json"))
    probe = "The capital of France is Paris. 3 + 4 = 7\nUser: hi\nAssistant: Hello!"
    assert tok.decode(tok.encode(probe).ids) == probe, "the list does not turn back into the same text"
    print(f"piece list built ({time.time() - began:.0f} s)", flush=True)
    manifest = {"name": "SlingerGPT 1 pico training text", "pieces in the list": 4096, "end of text": 0,
                "training pieces planned": TOTAL, "share": SHARE, "sources": {}}
    for name, share in SHARE.items():
        manifest["sources"][name] = cut(tok, name, int(TOTAL * share * SPARE))
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"done ({time.time() - began:.0f} s)")


if __name__ == "__main__":
    main()
