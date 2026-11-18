"""给 vLLM 搭官方 AWQ 版的两个派生目录（不改原目录）。

Loop-9B-AWQ-x：主模型。官方把词嵌入也做了 4 位 row_major 量化，vLLM 不支持；
  这里把词嵌入所在分片重写：去掉 embed_tokens 的 qweight/qzeros/scales，换成 BF16 原版的 embed_tokens.weight。
  其余文件全部软链；config 里删掉 quantization_config.embedding。
Loop-9B-AWQ-mtpdraft：草稿头。官方 AWQ 版 mtp/ 是 BF16、mtp.norm 重新校准过（与原版差 0.229）；
  去掉 vLLM 用不上的 mtp.main_norm（= 主模型 final norm 的拷贝），text_config 补 mtp_num_hidden_layers。
用法：python make_awq_dirs.py <models/sensenova 目录>
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

root = Path(sys.argv[1])
awq, bf16 = root / "Loop-9B-AWQ", root / "Loop-9B"

# ---- 主模型
dst = root / "Loop-9B-AWQ-x"
dst.mkdir(exist_ok=True)
idx = json.loads((awq / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
ekey = "model.language_model.embed_tokens"
shard = wm[ekey + ".qweight"]
for f in awq.iterdir():
    if f.name in ("model.safetensors.index.json", "config.json", shard, "mtp"):
        continue
    if not (dst / f.name).exists():
        os.symlink(f, dst / f.name)
new_shard = "model-embed-bf16-" + shard
if not (dst / new_shard).exists():
    tensors = {}
    with safe_open(awq / shard, "pt") as f:
        for k in f.keys():
            if not k.startswith(ekey + "."):
                tensors[k] = f.get_tensor(k)
    bidx = json.loads((bf16 / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(bf16 / bidx[ekey + ".weight"], "pt") as f:
        tensors[ekey + ".weight"] = f.get_tensor(ekey + ".weight")
    save_file(tensors, dst / new_shard, metadata={"format": "pt"})
for k in [k for k in wm if k.startswith(ekey + ".")]:
    del wm[k]
for k, v in list(wm.items()):
    if v == shard:
        wm[k] = new_shard
wm[ekey + ".weight"] = new_shard
(dst / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
cfg = json.loads((awq / "config.json").read_text())
cfg["quantization_config"].pop("embedding", None)
# 官方写的是 model.visual，vLLM 里视觉编码器前缀是 visual.，匹配不上会被当成 AWQ 层
if "visual" not in cfg["quantization_config"]["modules_to_not_convert"]:
    cfg["quantization_config"]["modules_to_not_convert"].append("visual")
# vLLM 建 MTP 草稿头用的是主模型的 config 和量化配置（草稿目录只出权重）：
# 草稿头是 BF16，要在主模型的不量化列表里排除 mtp.*；AWQ 版把 mtp 层数写成了 0，改回 1
if "mtp" not in cfg["quantization_config"]["modules_to_not_convert"]:
    cfg["quantization_config"]["modules_to_not_convert"].append("mtp")
cfg["text_config"]["mtp_num_hidden_layers"] = 1
# AWQ 版架构名写的是 Qwen3_5LoopForConditionalGeneration，vLLM 只注册了原版的名字 → 否则落到 transformers 通用后端
cfg["architectures"] = ["SenseNovaLoopForConditionalGeneration"]
# AWQ 版写的是 flash_attention_2（21 没装这个包），原版是 sdpa
cfg["attn_implementation"] = "sdpa"
# 官方 AWQ 仓库没带远程代码，也没写 auto_map：从 BF16 原版补上
cfg.setdefault("auto_map", json.loads((bf16 / "config.json").read_text())["auto_map"])
if not (dst / "modeling_sensenova_loop.py").exists():
    os.symlink(bf16 / "modeling_sensenova_loop.py", dst / "modeling_sensenova_loop.py")
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
print("target", dst, "weights", len(wm))

# ---- 草稿头
d2 = root / "Loop-9B-AWQ-mtpdraft"
d2.mkdir(exist_ok=True)
src = awq / "mtp"
with safe_open(src / "model-00001-of-00001.safetensors", "pt") as f:
    t = {k: f.get_tensor(k) for k in f.keys() if k != "mtp.main_norm.weight"}
save_file(t, d2 / "model.safetensors", metadata={"format": "pt"})
mcfg = json.loads((src / "config.json").read_text())
tc = mcfg.get("text_config", mcfg)
tc.setdefault("mtp_num_hidden_layers", 1)
(d2 / "config.json").write_text(json.dumps(mcfg, indent=2))
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
             "preprocessor_config.json", "processor_config.json"):
    if (awq / name).exists() and not (d2 / name).exists():
        os.symlink(awq / name, d2 / name)
print("draft", d2, len(t), "tensors; text_config mtp layers", tc.get("mtp_num_hidden_layers"))
