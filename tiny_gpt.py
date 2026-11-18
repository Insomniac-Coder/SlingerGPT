"""A small GPT (about 1M or 10M parameters) that learns to write from any text file.
Tokens are characters by default; --tokens words uses the text's most common words instead.

Train:     python tiny_gpt.py train input.txt            (2,000 steps; about 2 minutes on a laptop GPU)
Generate:  python tiny_gpt.py write "ROMEO:" --temperature 0.8 --top-k 10
No GPU:    add --size 1m to train (about 1 minute on a laptop processor)
Words:     add --tokens words to train on whole words, so it can only write real words
"""
import argparse, math, os, re, signal, sys, time
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# layers, heads, numbers per token, characters of memory
SIZES = {"1m": (4, 4, 128, 64), "10m": (6, 6, 384, 256), "50m": (10, 10, 640, 256)}
LAYERS, HEADS, WIDTH, CONTEXT = SIZES["10m"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT = 0.2   # randomly switch off 20% of the numbers while training, so the model cannot just memorise
# a word (don’t and to-morrow stay whole), a dash, a punctuation mark, or a line break
WORD = re.compile(r"\w+(?:[’'-]\w+)*|--+|[^\w\s]|\n")
UNKNOWN = "<unk>"                                  # stands in for words too rare to get their own token
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252


class Layer(nn.Module):
    """One layer: attention (Q, K, V, O), then feed-forward. Both add onto the token."""
    def __init__(self):
        super().__init__()
        self.width, self.heads, self.p = WIDTH, HEADS, DROPOUT   # its own size, so models of different sizes can coexist
        self.norm1, self.norm2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)     # the Q, K and V matrices side by side
        self.o = nn.Linear(WIDTH, WIDTH)           # the O matrix
        self.ff = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(self.width, dim=2)
        q, k, v = (m.view(b, t, self.heads, self.width // self.heads).transpose(1, 2) for m in (q, k, v))
        blend = F.scaled_dot_product_attention(q, k, v, is_causal=True,   # softmax(QK^T / sqrt(d)) V, earlier tokens only
                                               dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(blend.transpose(1, 2).reshape(b, t, self.width)))
        return x + self.drop(self.ff(self.norm2(x)))


def join_words(tokens):
    """Words back into text: a space between words, none before punctuation, after an opening quote,
    or on either side of a dash (Dickens writes "streets--the")."""
    out = ""
    for t in tokens:
        if t == "\n" or not out or out[-1] in "\n“‘(-" or t in ".,;:!?’”)" or t.startswith("--"):
            out += t
        else:
            out += " " + t
    return out


def dry_penalties(seq, multiplier, base, allowed, breakers):
    """DRY ("don't repeat yourself", as in llama.cpp): if the text now ends with a passage that appeared earlier,
    the token that followed it last time is penalised, more the longer the match. Matches stop at
    `breakers` (line breaks, colons, quotes), so repeating a speaker's name or starting a new line is free.
    Returns {token: amount to subtract from its score}."""
    n, out = len(seq), {}
    for end in range(n - 1):   # an earlier passage ending at `end`, followed by seq[end + 1]
        k = 0
        while k <= end and seq[end - k] == seq[n - 1 - k] and seq[n - 1 - k] not in breakers and n - 1 - k > end:
            k += 1
        if k >= allowed:
            token = seq[end + 1]
            out[token] = max(out.get(token, 0.0), multiplier * base ** min(k - allowed, 40))
    return out


class TinyGPT(nn.Module):
    def __init__(self, vocab, tie=False):
        super().__init__()
        self.context = CONTEXT
        self.tokens = nn.Embedding(vocab, WIDTH)   # the vocabulary table: one row of coordinates per character
        self.places = nn.Embedding(CONTEXT, WIDTH) # where in the line each character sits
        self.layers = nn.Sequential(*[Layer() for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(WIDTH)
        self.out = nn.Linear(WIDTH, vocab)         # final numbers -> one score per character
        self.drop = nn.Dropout(DROPOUT)
        for table in (self.tokens, self.places):   # start small, as GPT-2 does; large starting values waste early steps
            nn.init.normal_(table.weight, std=0.02)
        if tie:   # with a big word vocabulary, reuse the vocabulary table as the output matrix to save numbers
            self.out.weight = self.tokens.weight

    def forward(self, idx):
        x = self.drop(self.tokens(idx) + self.places(torch.arange(idx.shape[1], device=idx.device)))
        with torch.autocast(device_type=idx.device.type, dtype=torch.bfloat16, enabled=idx.device.type == "cuda"):
            x = self.layers(x)
        return self.out(self.norm(x.float()))

    @torch.no_grad()
    def write(self, idx, count, temperature=1.0, top_k=None, ban=None, dry=0.0, dry_base=1.75, dry_allowed=12, breakers=(),
              dry_from=0):
        """dry_from: the first piece the anti-repetition check looks at (the chat page skips a looked-up passage, which
        the reply should be free to quote)."""
        for _ in range(count):
            scores = self(idx[:, -self.context:])[:, -1, :] / max(temperature, 1e-6)
            if ban is not None:   # never write the stand-in for rare words
                scores[:, ban] = -math.inf
            if dry:   # a generation setting, not a weight: lower the odds of continuing an earlier passage
                for row in range(idx.shape[0]):
                    seq = idx[row, max(dry_from, idx.shape[1] - 512):].tolist()
                    for token, penalty in dry_penalties(seq, dry, dry_base, dry_allowed, breakers).items():
                        scores[row, token] -= penalty
            if top_k:
                cut = torch.topk(scores, min(top_k, scores.shape[-1])).values[:, [-1]]
                scores[scores < cut] = -math.inf
            idx = torch.cat([idx, torch.multinomial(F.softmax(scores, dim=-1), 1)], dim=1)
        return idx


def train(args, resume=None):
    """Trains from scratch, or carries on from a checkpoint (`resume`) exactly where it paused."""
    torch.manual_seed(1337)   # the same run every time; writing (below) stays random
    text = open(args.text, encoding="utf-8", errors="ignore").read()
    if args.tokens == "words":
        pieces = WORD.findall(text)
        vocab = [UNKNOWN] + [w for w, _ in Counter(pieces).most_common(args.vocab - 1)]
    else:
        pieces, vocab = text, sorted(set(text))
    init = None
    if args.init:   # carry on from a trained model: its size, token kind and vocabulary must fit this run
        init = torch.load(args.init, map_location=DEVICE)
        init["vocab"] = init.get("vocab", init.get("chars"))   # older model files call it "chars"
        if init["size"] != args.size or init.get("tokens", "chars") != args.tokens:
            raise SystemExit(f"{args.init} is a {init['size']} {init.get('tokens', 'chars')} model; this run is {args.size} {args.tokens}")
        unseen = set(vocab) - set(init["vocab"]) if args.tokens == "chars" else set()
        if unseen:
            raise SystemExit(f"the text has characters {args.init} never learned: {sorted(unseen)[:20]}")
        vocab = init["vocab"]
    if resume:
        vocab = resume["vocab"]   # the vocabulary the paused run was using
    stoi = {c: i for i, c in enumerate(vocab)}
    if args.tokens == "chars" and all(ord(c) < 256 for c in vocab):
        # one byte per character: 900M characters take 0.9 GB instead of 7 GB
        table = torch.zeros(256, dtype=torch.uint8)
        for c, i in stoi.items():
            table[ord(c)] = i
        raw = torch.frombuffer(bytearray(text.encode("latin-1")), dtype=torch.uint8)
        data = torch.empty_like(raw)
        for at in range(0, len(raw), 50_000_000):
            data[at:at + 50_000_000] = table[raw[at:at + 50_000_000].long()]
        del raw
        covered = 1.0
    else:
        data = torch.tensor([stoi.get(c, 0) for c in pieces])
        covered = sum(1 for c in pieces if c in stoi) / len(pieces)
    split = int(len(data) * 0.9)
    model = TinyGPT(len(vocab), tie=args.tokens == "words").to(DEVICE)
    if init and not resume:
        model.load_state_dict(init["model"])
        print(f"starting from {args.init}")
    print(f"{len(pieces):,} {args.tokens}, {len(vocab)} in the vocabulary ({covered:.0%} of the text), "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, on {DEVICE}\n")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    decode = lambda ids: join_words([vocab[i] for i in ids]) if args.tokens == "words" else "".join(vocab[i] for i in ids)
    ban = 0 if args.tokens == "words" else None

    def batch(part):
        src = data[:split] if part == "train" else data[split:]
        at = torch.randint(len(src) - CONTEXT - 1, (args.batch,))
        x = torch.stack([src[i:i + CONTEXT] for i in at]).long()
        y = torch.stack([src[i + 1:i + CONTEXT + 1] for i in at]).long()
        return x.to(DEVICE), y.to(DEVICE)

    def rate(step):
        """Learning rate: warm up over 200 steps, then ease down to a tenth along a cosine curve."""
        if step < 200:
            return args.lr * (step + 1) / 200
        progress = (step - 200) / max(1, args.steps - 200)
        return args.lr / 10 + 0.5 * (args.lr - args.lr / 10) * (1 + math.cos(math.pi * progress))

    def sample(step):
        model.eval()
        start = torch.tensor([[stoi.get("\n", 0)]], device=DEVICE)
        print(f"--- what it writes after {step} steps ---\n{decode(model.write(start, 200 if ban is None else 50, 0.8, ban=ban)[0].tolist())}\n---\n")
        model.train()

    @torch.no_grad()
    def unseen_loss():
        """How wrong it is on text it never trains on (the last 10%), averaged over 10 batches."""
        model.eval()
        total = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in (batch("val") for _ in range(10)))
        model.train()
        return total / 10

    checkpoint = args.checkpoint or os.path.splitext(args.save)[0] + ".ckpt"
    pause_file = checkpoint + ".pause"
    start, best, spent = 0, math.inf, 0.0
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        start, best, spent = resume["step"], resume["best"], resume["elapsed"]
        torch.set_rng_state(resume["rng"])   # the same batches come next, as if it had never stopped
        if DEVICE == "cuda" and resume["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(f"carrying on from step {start:,} of {args.steps:,}\n")

    def save_checkpoint(step):
        """Everything needed to carry on: the model, the optimiser's memory, the step and the random state."""
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "best": best,
                 "elapsed": time.time() - began, "vocab": vocab, "args": {**vars(args), "show": sorted(args.show)},
                 "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None}
        torch.save(state, checkpoint + ".tmp")
        os.replace(checkpoint + ".tmp", checkpoint)   # a crash mid-save never leaves a broken checkpoint

    asked = []   # Ctrl+C asks for a pause; it takes effect at the start of the next step
    previous_handler = signal.signal(signal.SIGINT, lambda *_: asked.append(True))
    began = time.time() - spent
    for step in range(start, args.steps + 1):
        if asked or os.path.exists(pause_file):
            save_checkpoint(step)
            if os.path.exists(pause_file):
                os.remove(pause_file)
            signal.signal(signal.SIGINT, previous_handler)
            print(f"paused at step {step:,} after {time.time() - began:.0f} s; carry on with:\n"
                  f"    python tiny_gpt.py resume {checkpoint}")
            return
        if step in args.show:
            sample(step)
        if step % args.eval_every == 0 or step == args.steps:
            val = unseen_loss()
            mark = ""
            if val < best:   # keep the best version, not the last one
                best, mark = val, "  saved"
                torch.save({"model": model.state_dict(), "vocab": vocab, "tokens": args.tokens, "size": args.size}, args.save)
            print(f"step {step:5}  loss {val:.2f}  ({time.time() - began:.0f} s){mark}")
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = rate(step)
        x, y = batch("train")
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # no single step may move the numbers too far
        opt.step()
        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(step + 1)
    signal.signal(signal.SIGINT, previous_handler)
    peak =f"; graphics memory peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if DEVICE == "cuda" else ""
    print(f"done after {time.time() - began:.0f} s; best loss {best:.2f} is in {args.save}{peak}")


def write(args):
    saved = torch.load(args.save, map_location=DEVICE)
    use_size(saved["size"])
    vocab, words = saved.get("vocab", saved.get("chars")), saved.get("tokens") == "words"
    model = TinyGPT(len(vocab), tie=words).to(DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()
    stoi = {c: i for i, c in enumerate(vocab)}
    prompt = WORD.findall(args.prompt) if words else args.prompt
    start = torch.tensor([[stoi[c] for c in prompt if c in stoi] or [0]], device=DEVICE)
    breakers = {stoi[c] for c in ("\n", ":", '"') if c in stoi}
    ids = model.write(start, args.length // 5 if words else args.length, args.temperature, args.top_k, ban=0 if words else None,
                      dry=args.dry, dry_allowed=args.dry_allowed, breakers=breakers)[0].tolist()
    print(join_words([vocab[i] for i in ids]) if words else "".join(vocab[i] for i in ids))


def use_size(name):
    global LAYERS, HEADS, WIDTH, CONTEXT
    LAYERS, HEADS, WIDTH, CONTEXT = SIZES[name]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("text")
    t.add_argument("--size", choices=SIZES, default="10m")
    t.add_argument("--tokens", choices=["chars", "words"], default="chars")
    t.add_argument("--vocab", type=int, default=2000, help="how many different words get a token (words only)")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--show", type=lambda s: {int(v) for v in s.split(",")}, default={0, 250, 1000, 2000})
    t.add_argument("--eval-every", type=int, default=100, help="check the loss on unseen text every this many steps")
    t.add_argument("--lr", type=float, default=1e-3, help="learning rate at its peak")
    t.add_argument("--dropout", type=float, default=0.2, help="0 when the text is so big that nothing repeats")
    t.add_argument("--save", default="tiny_gpt.pt")
    t.add_argument("--init", default=None, help="start from this trained model instead of random numbers")
    t.add_argument("--checkpoint", default=None, help="where to keep the pause/resume state (default: next to --save, .ckpt)")
    t.add_argument("--checkpoint-every", type=int, default=1000, help="save the pause/resume state every this many steps")
    r = sub.add_parser("resume")
    r.add_argument("checkpoint")
    w = sub.add_parser("write")
    w.add_argument("prompt")
    w.add_argument("--length", type=int, default=400)
    w.add_argument("--temperature", type=float, default=0.8)
    w.add_argument("--top-k", type=int, default=None)
    w.add_argument("--dry", type=float, default=0.0, help="anti-repetition strength (0 = off; 0.8 is typical)")
    w.add_argument("--dry-allowed", type=int, default=12, help="how long a repeat may run before it is penalised")
    w.add_argument("--save", default="tiny_gpt.pt")
    args = p.parse_args()
    if args.cmd == "train":
        use_size(args.size)
        DROPOUT = args.dropout
        train(args)
    elif args.cmd == "resume":
        state = torch.load(args.checkpoint, map_location=DEVICE)
        saved = argparse.Namespace(**state["args"])
        saved.show, saved.checkpoint = set(saved.show), args.checkpoint
        use_size(saved.size)
        DROPOUT = saved.dropout
        train(saved, resume=state)
    else:
        write(args)
