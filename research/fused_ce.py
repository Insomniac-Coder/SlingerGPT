"""Output layer + cross-entropy in slices, so the full table of word scores (tokens x 50,304) never exists.

    loss = fused_linear_ce(hidden, weight, targets[, bias])      hidden: (N, width) bf16; weight: (vocab, width)
    python -P research/fused_ce.py                                self-check against the plain computation

Same maths as F.cross_entropy(F.linear(hidden, weight).float(), targets): scores are made in BF16 (as autocast
does), then softmax in FP32. Because the loss is the very end of the network, its gradient is computed during the
forward pass, slice by slice (scores - one-hot, over N), and handed back in backward: no extra multiplies, and only
one slice of scores (1,024 x 50,304 x 4 bytes = 0.2 GB) is alive at a time instead of ~2.5 GB.
"""
import torch
import torch.nn.functional as F

SLICE = 1024


class _FusedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets):
        n = hidden.shape[0]
        w = weight.to(hidden.dtype)
        loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        d_hidden = torch.empty_like(hidden)
        d_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
        d_bias = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.float32) if bias is not None else None
        for at in range(0, n, SLICE):
            h, t = hidden[at:at + SLICE], targets[at:at + SLICE]
            scores = F.linear(h, w, None if bias is None else bias.to(h.dtype)).float()
            lse = torch.logsumexp(scores, dim=-1)
            loss += (lse - scores.gather(1, t[:, None]).squeeze(1)).sum()
            grad = torch.softmax(scores, dim=-1)                    # d loss / d scores = softmax - one-hot, / n
            grad.scatter_add_(1, t[:, None], torch.full_like(t[:, None], -1, dtype=grad.dtype))
            grad = (grad / n).to(h.dtype)
            d_hidden[at:at + SLICE] = grad @ w
            d_weight += (grad.t() @ h).float()   # BF16 multiply, slices summed in FP32
            if d_bias is not None:
                d_bias += grad.float().sum(0)
        ctx.save_for_backward(d_hidden, d_weight, d_bias)
        ctx.weight_dtype = weight.dtype
        return loss / n

    @staticmethod
    def backward(ctx, g):
        d_hidden, d_weight, d_bias = ctx.saved_tensors
        return (d_hidden * g.to(d_hidden.dtype), (d_weight * g).to(ctx.weight_dtype),
                None if d_bias is None else d_bias * g, None)


def fused_linear_ce(hidden, weight, targets, bias=None):
    return _FusedLinearCE.apply(hidden, weight, bias, targets)


if __name__ == "__main__":
    torch.manual_seed(0)
    n, width, vocab = 4096, 1280, 50304
    h = (torch.randn(n, width, device="cuda") * 0.5).bfloat16().requires_grad_()
    w = (torch.randn(vocab, width, device="cuda") * 0.02).requires_grad_()
    b = torch.zeros(vocab, device="cuda", requires_grad=True)
    t = torch.randint(0, 50257, (n,), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ref = F.cross_entropy(F.linear(h, w.bfloat16(), b.bfloat16()).float(), t)
    ref.backward()
    ref_mem = torch.cuda.max_memory_allocated() - base
    rh, rw, rb = h.grad.clone(), w.grad.clone(), b.grad.clone()
    h.grad = w.grad = b.grad = None

    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fused_linear_ce(h, w, t, b)
    out.backward()
    mem = torch.cuda.max_memory_allocated() - base
    rel = lambda a, r: ((a.float() - r.float()).norm() / r.float().norm()).item()
    print(f"loss {out.item():.6f} vs {ref.item():.6f}; gradient difference: hidden {rel(h.grad, rh):.1e}, "
          f"weight {rel(w.grad, rw):.1e}, bias {rel(b.grad, rb):.1e}")
    print(f"extra graphics memory: {mem / 2**30:.2f} GB fused vs {ref_mem / 2**30:.2f} GB plain")
    assert abs(out.item() - ref.item()) < 1e-4 and rel(h.grad, rh) < 1e-2 and rel(w.grad, rw) < 1e-2
    print("OK: same loss and gradients")
