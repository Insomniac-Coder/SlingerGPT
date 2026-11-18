"""Distillation: a 10M student learns to answer like the 50M v2 teacher.

    python distill.py train --alpha 0.9 --save models/distill10m.pt   student B: learns mostly from the teacher
    python distill.py train --alpha 0   --save models/plain10m.pt     student A: learns from the text alone
    python distill.py compare models/plain10m.pt models/distill10m.pt

Both students start from the same random numbers and see the same batches of the same text (the
teacher's own training text, data/corpus/mix_v2.txt) for the same number of steps. Only the lesson
differs. Student A is told which character actually came next. Student B is mostly shown the teacher's
odds for every possible next character ("e 40%, a 20%, o 15%..."), which carries far more information
per character than the one right answer. The temperature softens those odds so that the teacher's
second and third choices show up clearly too.
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import tiny_gpt as t

TEXT = HERE / "data" / "corpus" / "mix_v2.txt"
TEACHER = HERE / "models" / "mix50m_v2.pt"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_model(path):
    saved = torch.load(path, map_location=t.DEVICE)
    t.use_size(saved["size"])
    model = t.TinyGPT(len(saved["vocab"])).to(t.DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["vocab"]


def load_text(vocab):
    """The text as one byte per character, numbered as in the teacher's vocabulary.
    Returns (first 90%, last 10%); no model ever trained on the last 10%."""
    raw = torch.frombuffer(bytearray(TEXT.read_bytes()), dtype=torch.uint8)
    table = torch.zeros(256, dtype=torch.uint8)
    for i, c in enumerate(vocab):
        table[ord(c)] = i
    data = torch.empty_like(raw)
    for at in range(0, len(raw), 50_000_000):
        data[at:at + 50_000_000] = table[raw[at:at + 50_000_000].long()]
    split = int(len(data) * 0.9)
    return data[:split], data[split:]


def batch(src, size, generator, context=256):
    at = torch.randint(len(src) - context - 1, (size,), generator=generator)
    x = torch.stack([src[i:i + context] for i in at]).long()
    y = torch.stack([src[i + 1:i + context + 1] for i in at]).long()
    return x.to(t.DEVICE), y.to(t.DEVICE)


def train(args):
    teacher, vocab = load_model(TEACHER)
    if args.alpha == 0:
        teacher = None   # student A never asks the teacher anything
    train_text, _ = load_text(vocab)
    t.use_size("10m")
    t.DROPOUT = 0.0   # 2 billion characters: nothing repeats, so nothing to guard against memorising
    torch.manual_seed(1337)   # both students start from exactly the same random numbers
    student = t.TinyGPT(len(vocab)).to(t.DEVICE)
    print(f"student: {sum(p.numel() for p in student.parameters()):,} parameters; "
          f"{'learning from the teacher' if teacher else 'learning from the text alone'} (alpha {args.alpha})\n")
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    batches = torch.Generator().manual_seed(2024)   # and see exactly the same batches
    soft = args.temperature
    began = time.time()
    for step in range(args.steps):
        if step < 200:   # the same warm-up and cosine ease-down as tiny_gpt.py
            lr = args.lr * (step + 1) / 200
        else:
            lr = args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * (step - 200) / (args.steps - 200)))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = batch(train_text, args.batch, batches)
        scores = student(x).flatten(0, 1)
        from_text = F.cross_entropy(scores, y.flatten())   # how wrong it was about the actual next character
        loss = from_text
        if teacher:
            with torch.no_grad():
                teacher_scores = teacher(x).flatten(0, 1)
            # how far its odds are from the teacher's odds, both softened by the temperature
            from_teacher = F.kl_div(F.log_softmax(scores / soft, dim=-1), F.log_softmax(teacher_scores / soft, dim=-1),
                                    log_target=True, reduction="batchmean") * soft * soft
            loss = args.alpha * from_teacher + (1 - args.alpha) * from_text
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step:6}  loss on the text {from_text.item():.3f}  ({time.time() - began:.0f} s)")
    torch.save({"model": student.state_dict(), "vocab": vocab, "tokens": "chars", "size": "10m"}, args.save)
    print(f"done after {time.time() - began:.0f} s -> {args.save}")


@torch.no_grad()
def compare(args):
    """On unseen text: how well each model predicts it, and how often a student's first choice for
    the next character is the teacher's first choice."""
    teacher, vocab = load_model(TEACHER)
    _, unseen = load_text(vocab)
    unseen = unseen[:20_000_000]
    models = {"teacher (50M v2)": teacher} | {Path(p).stem: load_model(p)[0] for p in args.students}
    results = {name: {"loss": 0.0, "agrees": 0.0, "gap": 0.0} for name in models}
    batches = torch.Generator().manual_seed(7)   # the same 100 batches for every model
    for _ in range(100):
        x, y = batch(unseen, 64, batches)
        teacher_scores = teacher(x).flatten(0, 1)
        for name, model in models.items():
            scores = teacher_scores if model is teacher else model(x).flatten(0, 1)
            r = results[name]
            r["loss"] += F.cross_entropy(scores, y.flatten()).item() / 100
            r["agrees"] += (scores.argmax(-1) == teacher_scores.argmax(-1)).float().mean().item() / 100
            r["gap"] += F.kl_div(F.log_softmax(scores, -1), F.log_softmax(teacher_scores, -1),
                                 log_target=True, reduction="batchmean").item() / 100
    for name, r in results.items():
        print(f"{name:18} loss on unseen text {r['loss']:.3f}   same first choice as the teacher {r['agrees']:.1%}"
              f"   distance from the teacher's odds {r['gap']:.3f}")
    (HERE / "logs" / "distill_scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


@torch.no_grad()
def odds(args):
    """The teacher's odds for the next character after `context`: as they are, and softened by the
    temperature, which is what student B is taught."""
    teacher, vocab = load_model(TEACHER)
    stoi = {c: i for i, c in enumerate(vocab)}
    scores = teacher(torch.tensor([[stoi[c] for c in args.context]], device=t.DEVICE))[0, -1]
    result = {"context": args.context, "temperature": args.temperature, "odds": {}}
    for name, soft in (("as is", 1.0), ("softened", args.temperature)):
        p = F.softmax(scores / soft, dim=-1)
        top = torch.topk(p, 6)
        result["odds"][name] = [[vocab[i], round(v, 4)] for v, i in zip(top.values.tolist(), top.indices.tolist())]
        print(name, result["odds"][name])
    (HERE / "logs" / "distill_odds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--alpha", type=float, default=0.9, help="share of the lesson from the teacher (0 = text only)")
    tr.add_argument("--temperature", type=float, default=2.0, help="how much to soften the odds before comparing")
    tr.add_argument("--steps", type=int, default=10_000)
    tr.add_argument("--batch", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--save", required=True)
    c = sub.add_parser("compare")
    c.add_argument("students", nargs="+")
    o = sub.add_parser("odds")
    o.add_argument("context")
    o.add_argument("--temperature", type=float, default=2.0)
    args = p.parse_args()
    {"train": train, "compare": compare, "odds": odds}[args.cmd](args)


if __name__ == "__main__":
    main()
