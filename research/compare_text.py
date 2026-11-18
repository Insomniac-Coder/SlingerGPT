"""Has the trained models from final_test.py continue the same prompts, so their writing can be read side by side.

    python -P research/compare_text.py [name ...]      (default: the three 30M models)

Each model continues every prompt twice: once always taking its most likely piece ("steady"), once drawing
pieces at random in proportion to their scores, cooled a little ("varied", the same random seed for all).
Every continuation is cut to the same number of letters, because the models' pieces differ in length.
Writes research/logs/compare_text.md and prints it.
"""
import sys
from pathlib import Path
import torch
import tiktoken
from tokenizers import Tokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import arch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NAMES = sys.argv[1:] or ["final_30m_normal", "final_30m_new_same_time", "final_30m_new_same_loss"]
PROMPTS = ["The weather was terrible yesterday, so we decided to", "Once upon a time there was a small village where",
           "The most important thing to remember about cooking rice is", "Scientists have recently discovered that",
           "She opened the door slowly and saw", "The capital of France is", "In 1969, the first humans",
           "To fix a flat bicycle tyre, first"]
LETTERS, CONTEXT = 300, 256


def load(name):
    saved = torch.load(ROOT / "models" / f"{name}.pt")
    model = arch.build(**{**saved["cfg"], "fp8": False})   # 8-bit is a training speed-up; answering runs in 16-bit
    model.load_state_dict(saved["model"])
    if saved["cfg"]["vocab"] == 50304:
        enc = tiktoken.get_encoding("gpt2")
        return model.eval(), enc.encode_ordinary, enc.decode, enc.n_vocab
    tok = Tokenizer.from_file(str(ROOT / "data" / "pieces" / "bpe4096" / "tokenizer.json"))
    return model.eval(), (lambda s: tok.encode(s).ids), tok.decode, saved["cfg"]["vocab"]


@torch.no_grad()
def continue_text(model, encode, decode, real, prompt, varied):
    ids = encode(prompt)
    start = len(ids)
    torch.manual_seed(0)
    while len(decode(ids[start:])) < LETTERS:
        scores = model(torch.tensor([ids[-CONTEXT:]]).cuda())[0, -1, :real].float()   # the padding pieces are never offered
        if varied:
            top = torch.topk(scores / 0.8, 40)
            ids.append(int(top.indices[torch.multinomial(torch.softmax(top.values, -1), 1)]))
        else:
            ids.append(int(scores.argmax()))
    return decode(ids[start:])[:LETTERS].replace("\n", " ")


if __name__ == "__main__":
    models = {n: load(n) for n in NAMES}
    out = []
    for prompt in PROMPTS:
        out.append(f"\n## {prompt}\n")
        for varied in (False, True):
            for n, m in models.items():
                out.append(f"- **{n}, {'varied' if varied else 'steady'}:** {continue_text(*m, prompt, varied)}")
    (HERE / "logs" / ("compare_text.md" if NAMES[0].startswith("final_30m") else f"compare_text_{NAMES[0]}.md")).write_text(
        "\n".join(out), encoding="utf-8")
    print("\n".join(out))
