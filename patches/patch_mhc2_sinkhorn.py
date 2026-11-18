"""给 sensenova_loop_mhc.py 加 2 流（2×2）的 Triton Sinkhorn。

原版 Triton 融合只写了 4 流；Loop-9B 正式版是 2 流，Sinkhorn 退回 PyTorch：
每次 20 轮 × 两次 logsumexp ≈ 近 300 个小 kernel，每词元 96 次 ≈ 2.7 万个 kernel，decode 卡在 14 词元/秒。
这里把 20 轮放进一个 kernel，加法顺序与 PyTorch 两元素归约一致（a + b）。

用法：python patch_mhc2_sinkhorn.py <site-packages/vllm/model_executor/layers/sensenova_loop_mhc.py>
幂等。
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
MARK = "_sinkhorn2_kernel"
if MARK in s:
    print("skip: already patched")
    sys.exit(0)

KERNEL = '''

@triton.jit(do_not_specialize=["N"])
def _sinkhorn2_kernel(X, Y, N, STRIDE: tl.constexpr, ITERATIONS: tl.constexpr, BLOCK: tl.constexpr):
    # 2×2 log-Sinkhorn：先按行、再按列做 logsumexp 归一，FP32，全部迭代在寄存器里完成
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = token < N
    base = token * STRIDE
    x00 = tl.load(X + base + 0, mask, other=0)
    x01 = tl.load(X + base + 1, mask, other=0)
    x10 = tl.load(X + base + 2, mask, other=0)
    x11 = tl.load(X + base + 3, mask, other=0)
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
    out = token * 4
    tl.store(Y + out + 0, tldevice.exp(x00), mask)
    tl.store(Y + out + 1, tldevice.exp(x01), mask)
    tl.store(Y + out + 2, tldevice.exp(x10), mask)
    tl.store(Y + out + 3, tldevice.exp(x11), mask)


def log_sinkhorn('''

anchor = "\n\ndef log_sinkhorn("
assert s.count(anchor) == 1
s = s.replace(anchor, KERNEL, 1)

BRANCH = '''    if (
        backend == "triton"
        and log_mix.is_cuda
        and torch.version.hip is None
        and log_mix.dtype == torch.float32
        and log_mix.ndim == 3
        and log_mix.shape[-2:] == (2, 2)
        and log_mix.stride()[-2:] == (2, 1)
        and not log_mix.requires_grad
        and log_mix.shape[0] > 0
    ):
        output = torch.empty(log_mix.shape, dtype=log_mix.dtype, device=log_mix.device)
        _sinkhorn2_kernel[(triton.cdiv(log_mix.shape[0], 128),)](
            log_mix, output, log_mix.shape[0], log_mix.stride(0), iterations, 128,
            enable_fp_fusion=False,
        )
        logger.info_once("SenseNova Loop mHC Sinkhorn executing triton 2x2 (FP32)")
        sinkhorn_execution_counts["triton"] += 1
        return output
    if backend == "triton":
        logger.warning_once(
            "SenseNova Loop mHC Sinkhorn falling back to torch: unsupported input"
        )'''
old = '''    if backend == "triton":
        logger.warning_once(
            "SenseNova Loop mHC Sinkhorn falling back to torch: unsupported input"
        )'''
assert s.count(old) == 1
s = s.replace(old, BRANCH, 1)
p.write_text(s)
print("patched", p)
