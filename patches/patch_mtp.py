"""让 vllm-loop 能给 SenseNova-Loop-9B 开 MTP 投机解码（复用 vLLM 自带的 Qwen3_5MTP）。

1. speculative.py：model_type `sensenova_loop` 归入 qwen3_5 一类 → 草稿头用 Qwen3_5MTP
   （Loop 的 text_config 本来就是 qwen3_5_text，mtp.safetensors 结构与 Qwen3.5 MTP 完全一致）。
2. config.py：fork 白名单禁投机解码 → 放开（与 TP、eager 同类，只是没验证过）。
3. 派生模型目录 <src>-mtp：全部文件软链回原目录，索引补上 mtp.*（原索引没列，加载器会把 mtp.safetensors 过滤掉）。

用法：python patch_mtp.py <site-packages/vllm> <原模型目录>
幂等。
"""
import json
import os
import sys
from pathlib import Path

vllm, src = Path(sys.argv[1]), Path(sys.argv[2])

p = vllm / "config/speculative.py"
s = p.read_text()
old = '''        if hf_config.model_type in (
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):'''
new = '''        if hf_config.model_type in (
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
            "sensenova_loop",  # 21: Loop 的 MTP 头就是 Qwen3.5 MTP
        ):'''
if '"sensenova_loop",  # 21' in s:
    print("skip speculative.py")
else:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new, 1))
    print("patched speculative.py")

p = vllm / "model_executor/models/config.py"
s = p.read_text()
old = "        if vllm_config.speculative_config is not None:\n            unsupported.append"
new = "        if False and vllm_config.speculative_config is not None:  # 21: 试 MTP\n            unsupported.append"
if "# 21: 试 MTP" in s:
    print("skip config.py")
else:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new, 1))
    print("patched config.py")

dst = src.with_name(src.name + "-mtp")
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.name == "model.safetensors.index.json":
        continue
    link = dst / f.name
    if not link.exists():
        os.symlink(f, link)
idx = json.loads((src / "model.safetensors.index.json").read_text())
from safetensors import safe_open
with safe_open(src / "mtp.safetensors", "pt") as m:
    for k in m.keys():
        idx["weight_map"][k] = "mtp.safetensors"
(dst / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
print("model dir", dst, "weights", len(idx["weight_map"]))
