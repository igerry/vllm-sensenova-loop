"""放开 fork 白名单里的「禁止量化」，让 vllm-loop 能用 --quantization fp8（Ampere 上走 Marlin W8A16）。
用法：python patch_fp8.py <site-packages/vllm>。幂等。"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "model_executor/models/config.py"
s = p.read_text()
old = "        if model.quantization is not None or vllm_config.quant_config is not None:\n"
new = "        if False and (model.quantization is not None or vllm_config.quant_config is not None):  # 21: 试 FP8\n"
if "# 21: 试 FP8" in s:
    print("skip")
else:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new, 1)); print("patched", p)

# 量化会新增 checkpoint 里没有的参数：线性层的 weight_scale / input_scale（在线量化），
# 以及离线 FP8 权重下注意力层的 k_scale / v_scale / q_scale / prob_scale（默认 1.0）。Loop 的两处完整性检查要跳过它们
QS_OLD = '_QSCALE = ("weight_scale", "input_scale", "weight_scale_inv", "workspace")  # 21: 量化参数'
QS_NEW = ('_QSCALE = ("weight_scale", "input_scale", "weight_scale_inv", "workspace",'
          ' "k_scale", "v_scale", "q_scale", "prob_scale")  # 21: 量化参数')
p = Path(sys.argv[1]) / "model_executor/models/sensenova_loop.py"
s = p.read_text()
if QS_OLD in s:  # 旧版补丁打过：只补上注意力的缩放参数
    p.write_text(s.replace(QS_OLD, QS_NEW, 1)); print("upgraded _QSCALE", p)
    s = p.read_text()
if "_QSCALE" not in s:
    s = s.replace("class SenseNovaLoopMHCRouter(nn.Module):",
                  QS_NEW + "\n\n\nclass SenseNovaLoopMHCRouter(nn.Module):", 1)
    old1 = "        for name, _ in self.named_parameters():\n"
    assert s.count(old1) == 1
    s = s.replace(old1, "        for name, _ in self.named_parameters():\n            if name.endswith(_QSCALE):\n                continue\n", 1)
    old2 = "        missing = {name for name, _ in self.named_parameters()} - loaded\n"
    assert s.count(old2) == 1
    s = s.replace(old2, "        missing = {name for name, _ in self.named_parameters() if not name.endswith(_QSCALE)} - loaded\n", 1)
    p.write_text(s); print("patched", p)
else:
    print("skip sensenova_loop.py")
