"""torchao 0.18's NVFP4 training layer, made to run on consumer Blackwell (RTX 50, sm_120).

    from research.fp4_sm120 import NVFP4Linear          (import before building the model)
    python -P research/fp4_sm120.py                     self-check: the random rounding is unbiased

Three problems, three fixes (the third is the NVFP4Linear class below):
1. torchao.prototype.moe_training/__init__ imports an unrelated Triton kernel that fails with Triton 3.9.
   The NVFP4 training files only import each other, so the package is registered without running that file.
2. Random ("stochastic") rounding of gradients uses cvt.rs...e2m1x4, a B200-only instruction (sm_100a);
   sm_120a refuses it. Here it is done in arithmetic instead: a value between two neighbouring FP4 numbers
   rounds up with probability equal to its distance from the lower one, then the exact grid value goes
   through the round-to-nearest instruction sm_120 has. Same result on average, as the hardware version.
"""
import sys, types
from pathlib import Path
import torch
import triton
import triton.language as tl
import torchao.prototype as _proto

if "torchao.prototype.moe_training" not in sys.modules:
    _pkg = types.ModuleType("torchao.prototype.moe_training")
    _pkg.__path__ = [str(Path(_proto.__file__).parent / "moe_training")]
    sys.modules["torchao.prototype.moe_training"] = _pkg

from torchao.prototype.moe_training.nvfp4_training import hadamard_quantize_row_col_triton as _rowcol
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import convert_8xfp32_to_4xfp4_packed
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4Linear as _NVFP4Linear


from torchao.prototype.moe_training.nvfp4_training import nvfp4_linear as _lin
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale as _scale

_F = torch.nn.functional
_SCALED = dict(scale_recipe_a=[_F.ScalingType.BlockWise1x16, _F.ScalingType.TensorWise],
               scale_recipe_b=[_F.ScalingType.BlockWise1x16, _F.ScalingType.TensorWise],
               swizzle_a=[_F.SwizzleType.SWIZZLE_32_4_4, _F.SwizzleType.NO_SWIZZLE],
               swizzle_b=[_F.SwizzleType.SWIZZLE_32_4_4, _F.SwizzleType.NO_SWIZZLE], output_dtype=torch.bfloat16)


class _FP4MatmulCachedWeight(torch.autograd.Function):
    """torchao's nvfp4_mm_triton (nvfp4_linear.py), line for line, except that the weight's FP4 copies arrive
    ready-made (`wq`) instead of being rebuilt on every call. Same three FP4 multiplies, same rounding."""
    @staticmethod
    def forward(ctx, x, weight, wq, sr_seed, sign):
        W_fp4, W_bs, W_gs, Wt_fp4, Wt_sf, W_amax = wq
        sign = list(sign)
        x2 = x.to(torch.bfloat16).reshape(-1, x.shape[-1]).contiguous()
        x_col_amax, x_row_amax = _lin.triton_rht_amax(x2, sign_vector=sign)
        x_col, x_col_sf, x_row, x_row_sf = _lin.triton_rht_quantize_row_col(
            x2, stochastic_rounding=False, sign_vector=sign, col_global_amax=x_col_amax, row_global_amax=x_row_amax)
        out = _F.scaled_mm(x_row.view(torch.float4_e2m1fn_x2), W_fp4.t(), scale_a=[x_row_sf.flatten(), _scale(x_row_amax)],
                           scale_b=[W_bs, W_gs], **_SCALED)
        ctx.save_for_backward(x_col, x_col_sf, x_col_amax, Wt_fp4, Wt_sf, W_amax, sr_seed)
        ctx.shape, ctx.sign = x.shape, sign
        return out.reshape(*x.shape[:-1], W_fp4.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x_col, x_col_sf, x_col_amax, Wt_fp4, Wt_sf, W_amax, sr_seed = ctx.saved_tensors
        dy2 = dy.contiguous().reshape(-1, dy.shape[-1])
        off_row = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=dy.device)
        off_col = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=dy.device)
        dy_col_amax, dy_row_amax = _lin.triton_rht_amax(dy2, sign_vector=ctx.sign)
        dy_col, dy_col_sf, dy_row, dy_row_sf = _lin.triton_rht_quantize_row_col(
            dy2, stochastic_rounding=True, sign_vector=ctx.sign, col_seed_base=sr_seed, row_seed_base=sr_seed ^ 1,
            col_offset_base=off_col, row_offset_base=off_row, col_global_amax=dy_col_amax, row_global_amax=dy_row_amax)
        dx = _F.scaled_mm(dy_row.view(torch.float4_e2m1fn_x2), Wt_fp4.t(), scale_a=[dy_row_sf.flatten(), _scale(dy_row_amax)],
                          scale_b=[Wt_sf.flatten(), _scale(W_amax)], **_SCALED)
        dw = _F.scaled_mm(dy_col.view(torch.float4_e2m1fn_x2), x_col.view(torch.float4_e2m1fn_x2).t(),
                          scale_a=[dy_col_sf.flatten(), _scale(dy_col_amax)], scale_b=[x_col_sf.flatten(), _scale(x_col_amax)],
                          **_SCALED)
        return dx.reshape(ctx.shape), dw, None, None, None


class NVFP4Linear(_NVFP4Linear):
    """torchao's FP4 training layer with two changes, neither touching the maths:
    - kept out of torch.compile (fix 3): compiling it together with recomputation crashes on Windows (the compiled
      seed handling hands a CPU value to a GPU kernel, T2); it is hand-written GPU kernels already.
    - the weight is converted to FP4 once per optimiser step, not on every call (16 micro-batches x 2 with
      recomputation): torchao converted FP32 -> BF16 -> FP32 -> max -> FP4 each time, most of FP4's overhead in
      the T2 profile. The cache is keyed on the weight's version counter, which every in-place update bumps."""
    @torch.compiler.disable
    def forward(self, x):
        w = self.weight
        key = (w.data_ptr(), w._version)
        if getattr(self, "_wq_key", None) != key:
            self._wq = _lin._triton_weight_quantize_2d(w.detach().to(torch.bfloat16))
            self._wq_key = key
        out = _FP4MatmulCachedWeight.apply(x, w, self._wq, self._sr_seed, self.rht_sign_vector)
        return out if self.bias is None else out + self.bias


@triton.jit
def _uniform(idx, key):
    """A uniform number in [0, 1) per element from a cheap integer hash (murmur3's finaliser) of the element's
    position and a per-call key: a few multiplies instead of a full Philox per value (11 ms -> fast; T1)."""
    M: tl.constexpr = 0xFFFFFFFF   # 32-bit hash done in 64-bit integers, masked after each multiply
    h = ((idx.to(tl.uint64) * 0x9E3779B1) ^ key) & M
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M
    h ^= h >> 16
    return (h >> 8).to(tl.float32) * (1.0 / 16777216.0)   # 24 random bits (the hardware instruction uses 8)


@triton.jit
def _sr_to_grid(scaled, idx, key):
    """Rounds values already scaled into FP4 range onto the E2M1 grid (0, .5, 1, 1.5, 2, 3, 4, 6) at random,
    unbiased. Grid spacing is 0.5 below 2, 1 below 4, 2 below 6."""
    a = tl.abs(scaled)
    step = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    lo = tl.floor(a / step) * step
    up = _uniform(idx, key) < (a - lo) / step
    q = tl.minimum(lo + tl.where(up, step, 0.0), 6.0)
    return tl.where(scaled < 0, -q, q)


@triton.jit
def _pack_fp4(scaled, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, STOCHASTIC_ROUNDING: tl.constexpr,
              seed_ptr, offset_base_ptr, tile_id):
    """Drop-in for torchao's hadamard_utils._pack_fp4 (same arguments) without the sm_100-only instruction."""
    if STOCHASTIC_ROUNDING:
        # key: the layer's seed mixed with this call's fresh random offset, so every step rounds differently
        key = (tl.load(seed_ptr).to(tl.uint64) ^ (tl.load(offset_base_ptr).to(tl.uint64) * 0x9E3779B97F4A7C15)) >> 32
        local = tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        scaled = _sr_to_grid(scaled, tile_id * (BLOCK_N * BLOCK_M) + local, key)
    return convert_8xfp32_to_4xfp4_packed(scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split())


_rowcol._pack_fp4 = _pack_fp4   # the only kernel file that uses it; must be set before its first compile


# --- Fix 4 (speed): the FP4 multiply as one sealed custom op, so torch.compile keeps its graph whole around it ---
# torchao's converters are "triton ops", which torch.compile may open up and fuse; with recomputation it fused the
# random-seed code wrongly and crashed (T2). A plain custom op is never opened: one node in the compiled graph,
# no graph breaks, no rewriting inside. The weight's FP4 copies live in fixed buffers, refreshed once per optimiser
# step by refresh_fp4_weights(model), which the training loop calls at the start of every step.

@torch.library.custom_op("slinger::fp4_mm", mutates_args=())
def fp4_mm(x: torch.Tensor, weight: torch.Tensor, w_fp4: torch.Tensor, w_sf: torch.Tensor, w_amax: torch.Tensor,
           wt_fp4: torch.Tensor, wt_sf: torch.Tensor, sr_seed: torch.Tensor, sign: list[int]
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x (M, K) @ weight.T in NVFP4 using the cached weight codes. Returns the output and x's column-wise FP4
    copy (codes, scales, max), which the backward pass needs. Same kernels and rounding as torchao's layer."""
    x2 = x.to(torch.bfloat16).contiguous()
    col_amax, row_amax = _lin.triton_rht_amax(x2, sign_vector=sign)
    x_col, x_col_sf, x_row, x_row_sf = _lin.triton_rht_quantize_row_col(
        x2, stochastic_rounding=False, sign_vector=sign, col_global_amax=col_amax, row_global_amax=row_amax)
    out = _F.scaled_mm(x_row.view(torch.float4_e2m1fn_x2), w_fp4.view(torch.float4_e2m1fn_x2).t(),
                       scale_a=[x_row_sf.flatten(), _scale(row_amax)], scale_b=[w_sf, _scale(w_amax)], **_SCALED)
    return out, x_col, x_col_sf, col_amax


@fp4_mm.register_fake
def _(x, weight, w_fp4, w_sf, w_amax, wt_fp4, wt_sf, sr_seed, sign):
    m, k = x.shape
    return (x.new_empty((m, weight.shape[0]), dtype=torch.bfloat16), x.new_empty((k, m // 2), dtype=torch.uint8),
            x.new_empty((k // 128, m // 64, 32, 16), dtype=torch.float8_e4m3fn), x.new_empty((), dtype=torch.float32))


@torch.library.custom_op("slinger::fp4_mm_backward", mutates_args=())
def fp4_mm_backward(dy: torch.Tensor, x_col: torch.Tensor, x_col_sf: torch.Tensor, x_col_amax: torch.Tensor,
                    wt_fp4: torch.Tensor, wt_sf: torch.Tensor, w_amax: torch.Tensor, sr_seed: torch.Tensor,
                    sign: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients for the input and the weight, as in torchao's layer (random rounding of dy, Hadamard on wgrad)."""
    dy2 = dy.contiguous()
    off_row = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=dy.device)
    off_col = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=dy.device)
    dy_col_amax, dy_row_amax = _lin.triton_rht_amax(dy2, sign_vector=sign)
    dy_col, dy_col_sf, dy_row, dy_row_sf = _lin.triton_rht_quantize_row_col(
        dy2, stochastic_rounding=True, sign_vector=sign, col_seed_base=sr_seed, row_seed_base=sr_seed ^ 1,
        col_offset_base=off_col, row_offset_base=off_row, col_global_amax=dy_col_amax, row_global_amax=dy_row_amax)
    dx = _F.scaled_mm(dy_row.view(torch.float4_e2m1fn_x2), wt_fp4.view(torch.float4_e2m1fn_x2).t(),
                      scale_a=[dy_row_sf.flatten(), _scale(dy_row_amax)], scale_b=[wt_sf.flatten(), _scale(w_amax)], **_SCALED)
    dw = _F.scaled_mm(dy_col.view(torch.float4_e2m1fn_x2), x_col.view(torch.float4_e2m1fn_x2).t(),
                      scale_a=[dy_col_sf.flatten(), _scale(dy_col_amax)], scale_b=[x_col_sf.flatten(), _scale(x_col_amax)],
                      **_SCALED)
    return dx, dw


@fp4_mm_backward.register_fake
def _(dy, x_col, x_col_sf, x_col_amax, wt_fp4, wt_sf, w_amax, sr_seed, sign):
    return dy.new_empty((dy.shape[0], wt_fp4.shape[0])), dy.new_empty((dy.shape[1], x_col.shape[0]))


def _fp4_setup(ctx, inputs, output):
    x, weight, w_fp4, w_sf, w_amax, wt_fp4, wt_sf, sr_seed, sign = inputs
    _, x_col, x_col_sf, x_col_amax = output
    ctx.save_for_backward(x_col, x_col_sf, x_col_amax, wt_fp4, wt_sf, w_amax, sr_seed)
    ctx.sign, ctx.x_dtype, ctx.weight_dtype = sign, x.dtype, weight.dtype


def _fp4_backward(ctx, d_out, *unused):
    x_col, x_col_sf, x_col_amax, wt_fp4, wt_sf, w_amax, sr_seed = ctx.saved_tensors
    dx, dw = fp4_mm_backward(d_out, x_col, x_col_sf, x_col_amax, wt_fp4, wt_sf, w_amax, sr_seed, ctx.sign)
    return dx.to(ctx.x_dtype), dw.to(ctx.weight_dtype), None, None, None, None, None, None, None


fp4_mm.register_autograd(_fp4_backward, setup_context=_fp4_setup)


class NVFP4LinearOp(_NVFP4Linear):
    """The FP4 layer as one sealed custom op inside the compiled graph (fix 4). Its weight's FP4 copies sit in
    fixed buffers; call refresh_fp4_weights(model) at the start of every training step (it skips layers whose
    weight has not changed). Same maths as torchao's layer and as NVFP4Linear above."""
    def _alloc(self):
        n, k, dev = self.out_features, self.in_features, self.weight.device
        self.register_buffer("w_fp4", torch.zeros(n, k // 2, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("w_sf", torch.zeros(n * k // 16, dtype=torch.float8_e4m3fn, device=dev), persistent=False)
        self.register_buffer("w_amax", torch.zeros((), dtype=torch.float32, device=dev), persistent=False)
        self.register_buffer("wt_fp4", torch.zeros(k, n // 2, dtype=torch.uint8, device=dev), persistent=False)
        self.register_buffer("wt_sf", torch.zeros(k // 128, n // 64, 32, 16, dtype=torch.float8_e4m3fn, device=dev), persistent=False)
        self._key = None

    @torch.no_grad()
    def refresh(self):
        if not hasattr(self, "w_fp4"):
            self._alloc()
        key = (self.weight.data_ptr(), self.weight._version)
        if key != self._key:
            w_fp4, w_sf, _, wt_fp4, wt_sf, w_amax = _lin._triton_weight_quantize_2d(self.weight.detach().to(torch.bfloat16))
            self.w_fp4.copy_(w_fp4.view(torch.uint8))
            self.w_sf.copy_(w_sf)
            self.w_amax.copy_(w_amax)
            self.wt_fp4.copy_(wt_fp4.view(torch.uint8))
            self.wt_sf.copy_(wt_sf)
            self._key = key

    def forward(self, x):
        shape = x.shape
        out = fp4_mm(x.reshape(-1, shape[-1]), self.weight, self.w_fp4, self.w_sf, self.w_amax, self.wt_fp4,
                     self.wt_sf, self._sr_seed, list(self.rht_sign_vector))[0]
        out = out.reshape(*shape[:-1], out.shape[-1])
        return out if self.bias is None else out + self.bias


def refresh_fp4_weights(model):
    """Bring every NVFP4LinearOp's FP4 weight copy up to date. Call at the start of each training step."""
    for m in model.modules():
        if isinstance(m, NVFP4LinearOp):
            m.refresh()


@triton.jit
def _check_kernel(x_ptr, out_ptr, seed, N: tl.constexpr):
    i = tl.program_id(0)
    offs = i * N + tl.arange(0, N)
    q = _sr_to_grid(tl.load(x_ptr + offs), offs, seed.to(tl.uint64))
    tl.store(out_ptr + offs, q)


if __name__ == "__main__":
    # 4,096 copies of each test value, rounded at random; the average must come back to the value.
    values = torch.tensor([0.1, 0.3, 0.74, 1.2, 1.9, 2.2, 2.5, 3.7, 4.4, 5.9, -0.3, -2.5, -5.0, 0.0, 6.0, 7.5])
    x = values.repeat_interleave(4096).cuda()
    out = torch.empty_like(x)
    _check_kernel[(len(values),)](x, out, 1234, N=4096)
    means = out.view(len(values), 4096).mean(1).cpu()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    on_grid = torch.isin(out.abs().cpu(), grid).all().item()
    expect = values.clamp(-6, 6)
    for v, m in zip(values.tolist(), means.tolist()):
        print(f"  {v:5.2f} -> average of 4,096 roundings {m:6.3f}")
    assert on_grid, "a rounded value is not an FP4 number"
    assert torch.allclose(means, expect, atol=0.05), "random rounding is biased"   # 4 standard errors at the widest step
    other = torch.empty_like(x)
    _check_kernel[(len(values),)](x, other, 99, N=4096)
    differ = (other != out).float().mean().item()
    assert differ > 0.2, "a different seed should round differently"
    print(f"OK: every result is an FP4 number, the rounding is unbiased, and another seed changes {differ:.0%} of the roundings")

    # The cached-weight layer against torchao's original: same seed -> identical output and gradients.
    ref = _NVFP4Linear(1280, 3840, device="cuda", dtype=torch.float32)
    new = NVFP4Linear.from_linear(ref)          # shares the weight
    new._sr_seed.copy_(ref._sr_seed)
    new._rht_sign_vector.copy_(ref._rht_sign_vector)
    new._refresh_rht_sign_vector_tuple()
    xin = torch.randn(4096, 1280, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(4096, 3840, device="cuda", dtype=torch.bfloat16) * 0.01

    def run(layer):
        x = xin.clone().requires_grad_()
        layer.weight.grad = None
        torch.manual_seed(5)
        out = layer(x)
        out.backward(dy)
        return out.detach(), x.grad, layer.weight.grad.clone()
    for name, a, b in zip(("output", "input gradient", "weight gradient"), run(ref), run(new)):
        assert torch.equal(a, b), f"{name} differs from torchao's layer"
    before = run(new)[0]
    with torch.no_grad():
        new.weight.mul_(1.5)                     # an in-place update, as the optimiser makes
    assert not torch.equal(run(new)[0], before), "the cached FP4 weight was not refreshed after the weight changed"
    assert torch.equal(run(new)[0], run(ref)[0]), "after the update the layers disagree"
    print("OK: cached-weight layer matches torchao's exactly, and refreshes when the weight changes")

    # The sealed custom-op layer: identical to torchao's, and it survives torch.compile + recomputation (the T2 crash).
    sealed = NVFP4LinearOp.from_linear(ref)
    sealed._sr_seed.copy_(ref._sr_seed)
    sealed._rht_sign_vector.copy_(ref._rht_sign_vector)
    sealed._refresh_rht_sign_vector_tuple()
    sealed.refresh()
    for name, a, b in zip(("output", "input gradient", "weight gradient"), run(ref), run(sealed)):
        assert torch.equal(a, b), f"sealed layer: {name} differs from torchao's layer"
    norm = torch.nn.RMSNorm(1280, device="cuda")
    block = lambda x: x + sealed(norm(x))[:, :1280]
    step = torch.compile(lambda x: torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False).float().pow(2).mean())
    xc = torch.randn(4096, 1280, device="cuda", requires_grad=True)
    for _ in range(2):
        loss = step(xc)
        loss.backward()
    assert torch.isfinite(xc.grad).all() and sealed.weight.grad is not None
    print("OK: sealed custom-op layer matches torchao's exactly and runs under torch.compile with recomputation")
