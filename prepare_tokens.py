"""Turns data/raw128/ into word-piece token files for gpt128.py: data/tokens128/train.bin and val.bin.

    python prepare_tokens.py

GPT-2's tokenizer (tiktoken "gpt2", 50,257 word pieces). Every document ends with <|endoftext|>, so the
model learns where a text stops. Tokens are stored as 16-bit numbers. 1 in 100 documents of each source
goes to val.bin and is never trained on. Wikipedia is repeated to reach its share: facts need repetition.

Budget (tokens):  FineWeb-Edu 1.25B · Cosmopedia 0.5B · Wikipedia 0.5B (repeated) · SODA 0.25B = 2.5B
Sources: FineWeb-Edu (ODC-By), Cosmopedia (Apache 2.0), English Wikipedia 2023-11-01 (CC BY-SA 3.0),
SODA (CC BY 4.0). Writes data/tokens128/manifest.json with what went in.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import tiktoken

HERE = Path(__file__).parent
RAW, OUT = HERE / "data" / "raw128", HERE / "data" / "tokens128"
BUDGET = {"fineweb-edu": 1_250_000_000, "cosmopedia": 500_000_000, "soda": 250_000_000, "wikipedia": 500_000_000}
ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def documents(kind):
    """Yields lists of document texts, a batch at a time."""
    if kind == "fineweb-edu":
        files, cols = sorted(RAW.glob("fineweb_edu_*.parquet")), ["text"]
        fmt = lambda r: r["text"]
    elif kind == "cosmopedia":
        files, cols = sorted(RAW.glob("cosmo_*.parquet")), ["text"]
        fmt = lambda r: r["text"]
    elif kind == "wikipedia":
        files, cols = sorted(RAW.glob("wiki_en_*.parquet")), ["title", "text"]
        fmt = lambda r: f"{r['title']}\n\n{r['text']}"
    else:
        files, cols = sorted(RAW.glob("soda_*.parquet")), ["narrative", "dialogue", "speakers"]
        fmt = lambda r: r["narrative"] + "\n\n" + "\n".join(f"{w}: {s}" for w, s in zip(r["speakers"], r["dialogue"]))
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=cols):
            yield [fmt(r) for r in batch.to_pylist()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val = open(OUT / "train.bin", "wb"), open(OUT / "val.bin", "wb")
    manifest, n = {}, 0
    for kind, budget in BUDGET.items():
        began, written, docs, held = time.time(), 0, 0, 0
        unique = []   # Wikipedia only: kept to repeat
        for texts in documents(kind):
            for tokens in ENC.encode_ordinary_batch(texts, num_threads=16):
                arr = np.array(tokens + [EOT], dtype=np.uint16)
                n += 1
                if n % 100 == 0:   # never trained on
                    arr.tofile(val)
                    held += 1
                    continue
                if written >= budget and kind != "wikipedia":
                    break
                if kind == "wikipedia":
                    unique.append(arr)
                else:
                    arr.tofile(train)
                written += len(arr)
                docs += 1
            if written >= budget and kind != "wikipedia":
                break
        if kind == "wikipedia":   # repeat the articles until the budget is reached
            once = np.concatenate(unique)
            repeats = budget / len(once)
            full = int(repeats)
            for _ in range(full):
                once.tofile(train)
            once[: budget - full * len(once)].tofile(train)
            written = budget
            manifest[kind] = {"documents": docs, "unique tokens": int(len(once)), "times seen": round(repeats, 2)}
        else:
            manifest[kind] = {"documents": docs, "tokens": written}
        manifest[kind]["held out for testing"] = held
        print(f"{kind:12} {manifest[kind]}  ({time.time() - began:.0f} s)")
    train.close(), val.close()
    sizes = {name: (OUT / name).stat().st_size // 2 for name in ("train.bin", "val.bin")}
    manifest |= {"train tokens": sizes["train.bin"], "val tokens": sizes["val.bin"], "tokenizer": "gpt2", "end of text": EOT}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\ntrain.bin {sizes['train.bin']:,} tokens · val.bin {sizes['val.bin']:,} tokens")


if __name__ == "__main__":
    main()
