"""More of the same text, cut with the lists already made, for runs on the 124M model long enough to learn facts.

    python -P research/prepare_more_text.py [megabytes, default 1000]

Takes the first N megabytes of data/corpus/mix_v2.txt, leaving out every 100th megabyte as prepare_pieces.py does
(those are the held-out text), and cuts it with GPT-2's list and with our list of 4,096
(data/pieces/bpe4096/tokenizer.json, the same list as in every other test). Writes data/pieces1g/<list>/train.bin
50 megabytes at a time; val.bin and its letters per piece are copied from data/pieces/<list>/, because the
held-out text is the same.
"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
import tiktoken
from tokenizers import Tokenizer

HERE = Path(__file__).parent
TEXT = HERE.parent / "data" / "corpus" / "mix_v2.txt"
SRC, OUT = HERE.parent / "data" / "pieces", HERE.parent / "data" / "pieces1g"
MB, STEP = 1 << 20, 50
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(total):
    data = np.memmap(TEXT, dtype=np.uint8, mode="r")
    keep = [i for i in range(total) if i % 100 != 99]
    gpt2, ours = tiktoken.get_encoding("gpt2"), Tokenizer.from_file(str(SRC / "bpe4096" / "tokenizer.json"))
    cut = {"gpt2": lambda texts: gpt2.encode_ordinary_batch(texts, num_threads=16),
           "bpe4096": lambda texts: [e.ids for e in ours.encode_batch(texts)]}
    for name, encode in cut.items():
        began, folder, count = time.time(), OUT / name, 0
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "train.bin", "wb") as f:
            for at in range(0, len(keep), STEP):
                texts = [data[i * MB:(i + 1) * MB].tobytes().decode("ascii") for i in keep[at:at + STEP]]
                if at == 0:   # the list turns the text back into exactly the same text
                    first = encode(texts[:1])[0][:400]
                    back = gpt2.decode(first) if name == "gpt2" else ours.decode(first)
                    assert texts[0].startswith(back), f"{name} does not turn back into the text"
                for ids in encode(texts):
                    a = np.asarray(ids, dtype=np.uint16)
                    a.tofile(f)
                    count += a.size
        shutil.copy(SRC / name / "val.bin", folder / "val.bin")
        old = json.loads((SRC / name / "manifest.json").read_text())
        manifest = {"vocabulary": old["vocabulary"], "train megabytes": len(keep), "train pieces": count,
                    "letters per piece, train": round(len(keep) * MB / count, 4), "val pieces": old["val pieces"],
                    "letters per piece, val": old["letters per piece, val"]}
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"{name}: {manifest}  ({time.time() - began:.0f} s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
