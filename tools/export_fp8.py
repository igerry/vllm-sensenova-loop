"""把 SenseNova-Loop-9B（含 MTP 头）导出成 FP8 权重，数值与 vLLM `--quantization fp8` 在线量化一致。

方案：语言模型和 MTP 的所有线性层权重 → float8_e4m3fn，逐张量一个 weight_scale = amax/448（vLLM 在线 fp8 同款）。
不量化：词嵌入、lm_head、各种 norm、mHC 路由（FP32）、loop_gate、线性注意力的 conv1d/A_log/dt_bias、整个视觉编码器。
config 写 quant_method=fp8、activation_scheme=dynamic，视觉层列进 ignored_layers。
用法：python export_fp8.py <源目录(Loop-9B-mtp)> <输出目录>
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
LIN = re.compile(
    r"^(model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|mlp\.(gate|up|down)_proj)"
    r"|mtp\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)|mtp\.fc)\.weight$"
)

idx = json.loads((src / "model.safetensors.index.json").read_text())
by_file = {}
for k, f in idx["weight_map"].items():
    by_file.setdefault(f, []).append(k)

new_map, nq = {}, 0
for i, (fname, keys) in enumerate(sorted(by_file.items()), 1):
    out = {}
    with safe_open(src / fname, "pt") as f:
        for k in keys:
            t = f.get_tensor(k)
            if LIN.match(k):
                w = t.float()
                scale = (w.abs().max() / FP8_MAX).clamp(min=1e-12)
                out[k] = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                out[k[: -len("weight")] + "weight_scale"] = scale.reshape(())
                nq += 1
            else:
                out[k] = t
    oname = f"model-{i:05d}-of-{len(by_file):05d}.safetensors"
    save_file(out, dst / oname, metadata={"format": "pt"})
    for k in out:
        new_map[k] = oname
    print(fname, "->", oname, len(out), flush=True)

total = sum(os.path.getsize(dst / f) for f in set(new_map.values()))
(dst / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}, indent=2))

cfg = json.loads((src / "config.json").read_text())
vis = sorted({re.sub(r"\.(weight|bias)$", "", k) for k in idx["weight_map"]
              if k.startswith("model.visual.") and k.endswith(".weight") and re.search(r"(qkv|proj|linear_fc\d)\.weight$", k)})
cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "ignored_layers": vis}
(dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

for f in src.iterdir():
    if f.name.endswith(".safetensors") or f.name in ("model.safetensors.index.json", "config.json"):
        continue
    p = f.resolve()
    (shutil.copytree if p.is_dir() else shutil.copy2)(p, dst / f.name)
print("quantized", nq, "linear weights; total", round(total / 2**30, 2), "GiB; ignored", len(vis))
