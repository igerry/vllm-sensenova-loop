"""SenseNova Loop 2 流 mHC 的融合实现（21 自用，补 fork 只写了 4 流的缺口）。

PyTorch 回退版每次路由约 20 个小 kernel，FP32 线性层走 cuBLAS gemv（M=1、N=8，只有 8 个并行块，每次 35 µs），
每词元 96 次路由 + 96 次合并。这里换成：
  router2：两个 kernel（分块部分和 → 汇总 + sigmoid + 2×2 Sinkhorn + pre_aggregate）
  merge2 ：一个 kernel（mix @ residual + post * branch，BF16 舍入位置与 PyTorch 一致）
数值：logits 用 scale·(x·w) 代替 (scale·x)·w，舍入有极小差异；其余与 PyTorch 回退版同序。
"""
import torch

from vllm.triton_utils import tl, tldevice, triton

PART = 16  # 每个分块存 8 个点积 + 1 个平方和，补齐到 16


@triton.jit
def _router2_partial_kernel(R, W, P, H2: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    offs = b * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(R + t * H2 + offs).to(tl.float32)
    base = P + (t * NB + b) * 16
    tl.store(base + 8, tl.sum(x * x, 0))
    for o in tl.static_range(8):
        w = tl.load(W + o * H2 + offs)
        tl.store(base + o, tl.sum(x * w, 0))


@triton.jit
def _pick(v, idx, i):
    return tl.sum(tl.where(idx == i, v, 0.0), 0)


@triton.jit
def _router2_finish_kernel(
    R, P, Bias, BR, POST, MIX,
    H: tl.constexpr, NB: tl.constexpr, ITERATIONS: tl.constexpr, BLOCKH: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    nb = tl.arange(0, NB)
    k = tl.arange(0, 16)
    part = tl.load(P + t * NB * 16 + nb[:, None] * 16 + k[None, :])
    sums = tl.sum(part, 0)
    ss = _pick(sums, k, 8)
    scale = 1.0 / tl.sqrt(ss / (2 * H) + 1e-6)
    bias = tl.load(Bias + k, k < 8, other=0.0)
    logits = scale * sums + bias
    l0 = _pick(logits, k, 0)
    l1 = _pick(logits, k, 1)
    l2 = _pick(logits, k, 2)
    l3 = _pick(logits, k, 3)
    x00 = _pick(logits, k, 4)
    x01 = _pick(logits, k, 5)
    x10 = _pick(logits, k, 6)
    x11 = _pick(logits, k, 7)
    pre0 = 1.0 / (1.0 + tldevice.exp(-l0))
    pre1 = 1.0 / (1.0 + tldevice.exp(-l1))
    # branch = (pre.to(bf16) @ residual) —— 一个 BF16 matmul，FP32 累加后舍入一次
    p0 = pre0.to(tl.bfloat16).to(tl.float32)
    p1 = pre1.to(tl.bfloat16).to(tl.float32)
    offs = hb * BLOCKH + tl.arange(0, BLOCKH)
    r0 = tl.load(R + t * 2 * H + offs).to(tl.float32)
    r1 = tl.load(R + t * 2 * H + H + offs).to(tl.float32)
    tl.store(BR + t * H + offs, (p0 * r0 + p1 * r1).to(tl.bfloat16))
    if hb == 0:
        tl.store(POST + t * 2 + 0, 2.0 * (1.0 / (1.0 + tldevice.exp(-l2))))
        tl.store(POST + t * 2 + 1, 2.0 * (1.0 / (1.0 + tldevice.exp(-l3))))
        for _ in range(ITERATIONS):
            m = tl.maximum(x00, x01)
            l = tldevice.log(tldevice.exp(x00 - m) + tldevice.exp(x01 - m)) + m
            x00 = x00 - l
            x01 = x01 - l
            m = tl.maximum(x10, x11)
            l = tldevice.log(tldevice.exp(x10 - m) + tldevice.exp(x11 - m)) + m
            x10 = x10 - l
            x11 = x11 - l
            m = tl.maximum(x00, x10)
            l = tldevice.log(tldevice.exp(x00 - m) + tldevice.exp(x10 - m)) + m
            x00 = x00 - l
            x10 = x10 - l
            m = tl.maximum(x01, x11)
            l = tldevice.log(tldevice.exp(x01 - m) + tldevice.exp(x11 - m)) + m
            x01 = x01 - l
            x11 = x11 - l
        tl.store(MIX + t * 4 + 0, tldevice.exp(x00))
        tl.store(MIX + t * 4 + 1, tldevice.exp(x01))
        tl.store(MIX + t * 4 + 2, tldevice.exp(x10))
        tl.store(MIX + t * 4 + 3, tldevice.exp(x11))


@triton.jit
def _merge2_kernel(R, BR, POST, MIX, OUT, H: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK + tl.arange(0, BLOCK)
    m00 = tl.load(MIX + t * 4 + 0).to(tl.bfloat16).to(tl.float32)
    m01 = tl.load(MIX + t * 4 + 1).to(tl.bfloat16).to(tl.float32)
    m10 = tl.load(MIX + t * 4 + 2).to(tl.bfloat16).to(tl.float32)
    m11 = tl.load(MIX + t * 4 + 3).to(tl.bfloat16).to(tl.float32)
    q0 = tl.load(POST + t * 2 + 0).to(tl.bfloat16).to(tl.float32)
    q1 = tl.load(POST + t * 2 + 1).to(tl.bfloat16).to(tl.float32)
    r0 = tl.load(R + t * 2 * H + offs).to(tl.float32)
    r1 = tl.load(R + t * 2 * H + H + offs).to(tl.float32)
    br = tl.load(BR + t * H + offs).to(tl.float32)
    # torch: matmul(mix.bf16, residual)（舍入一次）+ (post.bf16 * branch)（舍入一次），再相加舍入
    a0 = (m00 * r0 + m01 * r1).to(tl.bfloat16).to(tl.float32)
    a1 = (m10 * r0 + m11 * r1).to(tl.bfloat16).to(tl.float32)
    b0 = (q0 * br).to(tl.bfloat16).to(tl.float32)
    b1 = (q1 * br).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + t * 2 * H + offs, (a0 + b0).to(tl.bfloat16))
    tl.store(OUT + t * 2 * H + H + offs, (a1 + b1).to(tl.bfloat16))


def supported(residual: torch.Tensor) -> bool:
    return (
        residual.is_cuda
        and residual.dtype == torch.bfloat16
        and residual.ndim == 3
        and residual.shape[1] == 2
        and residual.shape[2] % 1024 == 0
        and residual.is_contiguous()
    )


def router2(residual, weight, bias, iterations):
    tokens, _, hidden = residual.shape
    block = 256
    nb = 2 * hidden // block
    partial = torch.empty((tokens, nb, PART), dtype=torch.float32, device=residual.device)
    _router2_partial_kernel[(tokens, nb)](residual, weight, partial, 2 * hidden, nb, block)
    branch = torch.empty((tokens, hidden), dtype=residual.dtype, device=residual.device)
    post = torch.empty((tokens, 2), dtype=torch.float32, device=residual.device)
    mix = torch.empty((tokens, 2, 2), dtype=torch.float32, device=residual.device)
    blockh = 1024
    _router2_finish_kernel[(tokens, hidden // blockh)](
        residual, partial, bias, branch, post, mix, hidden, nb, iterations, blockh,
        enable_fp_fusion=False,
    )
    return branch, post, mix


def merge2(residual, branch, post, mix):
    tokens, _, hidden = residual.shape
    out = torch.empty_like(residual)
    block = 1024
    _merge2_kernel[(tokens, hidden // block)](
        residual, branch.contiguous(), post.contiguous(), mix.contiguous(), out, hidden, block,
        enable_fp_fusion=False,
    )
    return out
