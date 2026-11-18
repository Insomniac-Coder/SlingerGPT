"""Makes more compact versions of the same text, for the question: how compact should a model's units be?

    python -P research/prepare_pieces.py

Takes the first 300 MB of data/corpus/mix_v2.txt (the text the 1m and 10m sizes read letter by letter) and
rewrites it as word pieces of four kinds: three vocabularies trained on this text (1,024, 4,096 and 16,384
pieces) and GPT-2's 50,257-piece vocabulary. The held-out text is the same megabytes lab.py holds out (every
100th megabyte of the whole file), so every version is scored on identical text.
Writes data/pieces/<name>/train.bin and val.bin (16-bit numbers), tokenizer.json where one was trained, and
manifest.json with the number of letters per piece. Every version is checked to turn back into the exact text.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import tiktoken
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = Path(__file__).parent
TEXT = HERE.parent / "data" / "corpus" / "mix_v2.txt"
OUT = HERE.parent / "data" / "pieces"
MB = 1 << 20
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    data = np.memmap(TEXT, dtype=np.uint8, mode="r")
    chunk = lambda i: data[i * MB:(i + 1) * MB].tobytes().decode("ascii")
    train = [chunk(i) for i in range(300) if i % 100 != 99]
    val = [chunk(i) for i in range(len(data) // MB) if i % 100 == 99]
    print(f"{len(train)} MB to train on, {len(val)} MB held out", flush=True)

    def save(name, encode, decode, vocab, tok=None):
        began = time.time()
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {"vocabulary": vocab}
        for part, texts in (("train", train), ("val", val)):
            ids = encode(texts)
            flat = np.concatenate([np.asarray(i, dtype=np.uint16) for i in ids])
            flat.tofile(folder / f"{part}.bin")
            manifest[f"{part} pieces"] = int(flat.size)
            manifest[f"letters per piece, {part}"] = round(sum(map(len, texts)) / flat.size, 4)
        assert decode(list(ids[0][:400])) == val[0][: len(decode(list(ids[0][:400])))], "does not turn back into the text"
        if tok is not None:
            tok.save(str(folder / "tokenizer.json"))
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"{name}: {manifest}  ({time.time() - began:.0f} s)", flush=True)

    gpt2 = tiktoken.get_encoding("gpt2")
    save("gpt2", lambda texts: gpt2.encode_ordinary_batch(texts, num_threads=16), gpt2.decode, 50257)
    for size in (1024, 4096, 16384):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(train[:100], trainers.BpeTrainer(vocab_size=size, show_progress=False,
                                                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        save(f"bpe{size}", lambda texts: [e.ids for e in tok.encode_batch(texts)], tok.decode, size, tok)


if __name__ == "__main__":
    main()
