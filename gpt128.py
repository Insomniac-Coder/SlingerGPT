"""Pre-trains a GPT-2-small-shaped model (12 layers, 768 numbers per token, 1,024-token memory, ~124M
parameters) on word-piece tokens from data/tokens128/ (made by prepare_tokens.py), built for speed.

    python gpt128.py train --tokens 2500000000 --save models/gpt128.pt [--compile] [--fp8]
    python gpt128.py resume models/gpt128.ckpt          (pause: Ctrl+C, or create models/gpt128.ckpt.pause)
    python gpt128.py write "Once upon a time" --save models/gpt128.pt

Speed: bf16 maths everywhere including the output layer (a third of the work at 50k word pieces), the
vocabulary padded to 50,304 (a multiple of 64, which the matrix units handle in blocks), flash attention,
a fused optimiser, TF32 for anything left in 32-bit, pinned memory so the next batch loads while the card
computes, and optionally torch.compile and FP8 matrix maths (both need Triton).
"""
import argparse, math, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import tiny_gpt as t

TOKENS = HERE / "data" / "tokens128"
VOCAB, CONTEXT = 50304, 1024   # GPT-2's 50,257 word pieces padded to a multiple of 64; tokens of memory (--context)
ENC = tiktoken.get_encoding("gpt2")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class GPT(t.TinyGPT):
    """The same model as tiny_gpt.py, run entirely in bf16, output layer included."""
    def forward(self, idx):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.tokens(idx) + self.places(torch.arange(idx.shape[1], device=idx.device))
            return self.out(self.norm(self.layers(x)))


def build():
    t.LAYERS, t.HEADS, t.WIDTH, t.CONTEXT, t.DROPOUT = 12, 12, 768, CONTEXT, 0.0
    torch.manual_seed(1337)
    model = GPT(VOCAB, tie=True)
    for name, p in model.named_parameters():   # GPT-2's starting values: small, and smaller still for the
        if p.dim() == 2 and "tokens" not in name and "places" not in name:   # matrices that add onto the token
            nn.init.normal_(p, std=0.02 / math.sqrt(2 * 12) if name.endswith(("o.weight", "ff.2.weight")) else 0.02)
        elif name.endswith("bias"):
            nn.init.zeros_(p)
    return model.cuda()


class Batches:
    """Random 1,025-token windows from a token file on disk, in pinned memory for a fast copy to the card."""
    def __init__(self, path, size, seed):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.size, self.rng = size, np.random.default_rng(seed)

    def next(self):
        at = self.rng.integers(0, len(self.data) - CONTEXT - 1, self.size)
        chunk = torch.from_numpy(np.stack([self.data[i:i + CONTEXT + 1] for i in at]).astype(np.int64)).pin_memory()
        chunk = chunk.cuda(non_blocking=True)
        return chunk[:, :-1], chunk[:, 1:]


def loss_of(model, x, y):
    return F.cross_entropy(model(x).float().view(-1, VOCAB), y.reshape(-1))


def train(args, resume=None):
    model = build()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters · micro-batch {args.micro} × {args.accum} "
          f"= {args.micro * args.accum * CONTEXT:,} tokens per step · {args.steps:,} steps")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    train_batches = Batches(Path(args.data) / "train.bin", args.micro, seed=2024)
    start, best, spent = 0, math.inf, 0.0
    if resume:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        train_batches.rng.bit_generator.state = resume["batch_rng"]
        start, best, spent = resume["step"], resume["best"], resume["elapsed"]
        print(f"carrying on from step {start:,}")
    raw = model
    if args.fp8:
        from torchao.float8 import convert_to_float8_training
        convert_to_float8_training(model, module_filter_fn=lambda m, name: isinstance(m, nn.Linear) and name != "out")
    if args.compile:
        model = torch.compile(model)
    checkpoint = os.path.splitext(args.save)[0] + ".ckpt"

    def save_checkpoint(step):
        state = {"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step, "best": best,
                 "elapsed": time.time() - began, "batch_rng": train_batches.rng.bit_generator.state, "args": vars(args)}
        torch.save(state, checkpoint + ".tmp")
        os.replace(checkpoint + ".tmp", checkpoint)

    @torch.no_grad()
    def unseen_loss():
        model.eval()
        val = Batches(Path(args.data) / "val.bin", args.micro, seed=7)   # the same test batches every time
        result = sum(loss_of(model, *val.next()).item() for _ in range(args.eval_batches)) / args.eval_batches
        model.train()
        return result

    asked = []
    previous = signal.signal(signal.SIGINT, lambda *_: asked.append(True))
    began = time.time() - spent
    window = time.time()
    for step in range(start, args.steps + 1):
        if asked or os.path.exists(checkpoint + ".pause"):
            save_checkpoint(step)
            if os.path.exists(checkpoint + ".pause"):
                os.remove(checkpoint + ".pause")
            signal.signal(signal.SIGINT, previous)
            print(f"paused at step {step:,}; carry on with: python gpt128.py resume {checkpoint}")
            return
        if step % args.eval_every == 0 or step == args.steps:
            val = unseen_loss()
            mark = ""
            if val < best:
                best, mark = val, "  saved"
                torch.save({"model": raw.state_dict(), "tokenizer": "gpt2", "vocab": VOCAB, "context": CONTEXT}, args.save)
            rate = (step - start) * args.micro * args.accum * CONTEXT / max(time.time() - window, 1e-9) if step > start else 0
            print(f"step {step:6}  unseen loss {val:.3f}  {rate:,.0f} tokens/s  ({time.time() - began:.0f} s){mark}", flush=True)
        if step == args.steps:
            break
        progress = step / args.steps
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(args.accum):   # several micro-batches add up into one step's worth
            loss = loss_of(model, *train_batches.next()) / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(step + 1)
    signal.signal(signal.SIGINT, previous)
    print(f"done after {time.time() - began:.0f} s; best unseen loss {best:.3f} in {args.save}; "
          f"graphics memory peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


@torch.no_grad()
def write(args):
    global CONTEXT
    saved = torch.load(args.save, map_location="cuda")
    CONTEXT = saved["context"]
    model = build()
    model.load_state_dict(saved["model"])
    model.eval()
    ids = torch.tensor([ENC.encode_ordinary(args.prompt)], device="cuda")
    out = model.write(ids, args.length, args.temperature, args.top_k, dry=args.dry, dry_allowed=args.dry_allowed,
                      breakers={ENC.eot_token, *ENC.encode_ordinary("\n"), *ENC.encode_ordinary(":")})
    text = ENC.decode([i for i in out[0].tolist() if i < ENC.n_vocab])
    print(text.split("<|endoftext|>")[0] if args.stop_at_end else text)


def main():
    global CONTEXT
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--tokens", type=int, default=2_500_000_000)
    tr.add_argument("--micro", type=int, default=16, help="sequences per micro-batch (graphics memory)")
    tr.add_argument("--accum", type=int, default=4, help="micro-batches per step")
    tr.add_argument("--lr", type=float, default=6e-4)
    tr.add_argument("--warmup", type=int, default=700)
    tr.add_argument("--eval-every", type=int, default=1000)
    tr.add_argument("--eval-batches", type=int, default=20)
    tr.add_argument("--checkpoint-every", type=int, default=1000)
    tr.add_argument("--compile", action="store_true")
    tr.add_argument("--fp8", action="store_true")
    tr.add_argument("--save", default="models/gpt128.pt")
    tr.add_argument("--data", default=str(TOKENS), help="folder with train.bin and val.bin")
    tr.add_argument("--context", type=int, default=CONTEXT, help="tokens of memory; fixed once trained")
    r = sub.add_parser("resume")
    r.add_argument("checkpoint")
    w = sub.add_parser("write")
    w.add_argument("prompt")
    w.add_argument("--length", type=int, default=120, help="word pieces to write")
    w.add_argument("--temperature", type=float, default=0.7)
    w.add_argument("--top-k", type=int, default=50)
    w.add_argument("--dry", type=float, default=0.0)
    w.add_argument("--dry-allowed", type=int, default=3, help="in word pieces")
    w.add_argument("--stop-at-end", action="store_true", help="stop at the end-of-text token")
    w.add_argument("--save", default="models/gpt128.pt")
    args = p.parse_args()
    if args.cmd == "train":
        CONTEXT = args.context
        args.steps = args.tokens // (args.micro * args.accum * CONTEXT)
        train(args)
    elif args.cmd == "resume":
        state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        CONTEXT = state["args"]["context"]
        if not Path(state["args"]["data"]).exists():   # the project moved: use the token folder next to this script
            state["args"]["data"] = str(TOKENS)
        train(argparse.Namespace(**state["args"]), resume=state)
    else:
        write(args)


if __name__ == "__main__":
    main()
