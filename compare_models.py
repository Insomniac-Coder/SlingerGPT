"""Scores trained models on the same unseen text, so their losses can be compared fairly.

    python compare_models.py models/mix50m.pt models/mix50m_v2.pt

Uses the last 10% of each mix (the part no run trains on), the first 20M characters of it, and the same
100 random batches of 64 passages for every model. Writes logs/scores.json.
"""
import json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import tiny_gpt as t

TESTS = {"new mix": HERE / "data/corpus/mix_v2.txt", "first mix": HERE / "data/corpus/mix_big.txt"}


def load(path):
    saved = torch.load(path, map_location=t.DEVICE)
    t.use_size(saved["size"])
    model = t.TinyGPT(len(saved["vocab"])).to(t.DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, {c: i for i, c in enumerate(saved["vocab"])}


def unseen_text(path, size=20_000_000):
    text = path.read_text(encoding="ascii")
    return text[int(len(text) * 0.9):][:size]


@torch.no_grad()
def score(model, stoi, text, batches=100, seed=7):
    data = torch.tensor([stoi[c] for c in text])
    g = torch.Generator().manual_seed(seed)   # the same batches for every model
    total = 0.0
    for _ in range(batches):
        at = torch.randint(len(data) - t.CONTEXT - 1, (64,), generator=g)
        x = torch.stack([data[i:i + t.CONTEXT] for i in at]).to(t.DEVICE)
        y = torch.stack([data[i + 1:i + t.CONTEXT + 1] for i in at]).to(t.DEVICE)
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    return total / batches


def main():
    texts = {name: unseen_text(path) for name, path in TESTS.items()}
    results = {}
    for path in sys.argv[1:]:
        model, stoi = load(path)
        results[Path(path).stem] = {name: round(score(model, stoi, text), 3) for name, text in texts.items()}
        print(Path(path).stem, results[Path(path).stem])
    (HERE / "logs" / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
