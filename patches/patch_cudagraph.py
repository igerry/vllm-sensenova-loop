"""放开 vllm4spec 白名单里的「必须 enforce_eager」，让 Loop 走 CUDA Graph（FULL_DECODE_ONLY）。
eager 下 decode 只有约 3 词元/秒，开图后才有后面的优化空间。
用法：python patch_cudagraph.py <site-packages/vllm>。幂等。"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "model_executor/models/config.py"
s = p.read_text()
old = "        if not model.enforce_eager:\n"
new = "        if False and not model.enforce_eager:  # 21: 试 CUDA Graph\n"
if "# 21: 试 CUDA Graph" in s:
    print("skip")
else:
    assert s.count(old) == 1, "anchor not found"
    p.write_text(s.replace(old, new))
    print("patched", p)
