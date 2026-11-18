"""把 sensenova_loop_mhc2.py（2 流融合路由 + 合并）接进 vllm-loop 的 sensenova_loop.py。

用法：python patch_mhc2_fused.py <site-packages/vllm> <sensenova_loop_mhc2.py>
关掉：环境变量 VLLM_SENSENOVA_LOOP_MHC2=0。幂等。
"""
import shutil
import sys
from pathlib import Path

vllm, src = Path(sys.argv[1]), Path(sys.argv[2])
shutil.copy2(src, vllm / "model_executor/layers/sensenova_loop_mhc2.py")
print("copied sensenova_loop_mhc2.py")

p = vllm / "model_executor/models/sensenova_loop.py"
s = p.read_text()
if "sensenova_loop_mhc2" in s:
    print("skip: already patched")
    sys.exit(0)

old = "from vllm.model_executor.layers import sensenova_loop_mhc as mhc_ops\n"
assert s.count(old) == 1
s = s.replace(old, old + (
    "import os as _os\n"
    "from vllm.model_executor.layers import sensenova_loop_mhc2 as mhc2_ops\n"
    "_MHC2 = _os.environ.get(\"VLLM_SENSENOVA_LOOP_MHC2\", \"1\") == \"1\"  # 21: 2 流融合\n"
), 1)

old = (
    "        fused = mhc_ops.FUSION_MODES[self.mhc_fusion]\n"
    "        normalized = mhc_ops.normalize("
)
assert s.count(old) == 1
s = s.replace(old, (
    "        if _MHC2 and self.streams == 2 and mhc2_ops.supported(residual):\n"
    "            return mhc2_ops.router2(residual, self.weight, self.bias, self.iterations)\n"
) + old, 1)

old = "        return mhc_ops.residual_merge(residual, branch, post, mix, backend=backend)\n"
assert s.count(old) == 1
s = s.replace(old, (
    "        if _MHC2 and mhc2_ops.supported(residual):\n"
    "            return mhc2_ops.merge2(residual, branch, post, mix)\n"
) + old, 1)
p.write_text(s)
print("patched", p)
