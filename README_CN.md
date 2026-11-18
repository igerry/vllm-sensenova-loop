# vllm-sensenova-loop

[English](README.md) | 中文

这是一组补丁，让 [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) 能在 **vLLM 0.33** 和 **两张 RTX 3090 Ti（Ampere，TP=2）** 上流畅运行。

Patches that make SenseNova-Loop-9B run fast on stock vLLM 0.33 with 2× RTX 3090 Ti (TP=2): CUDA Graph, fused Triton kernels for the 2-stream mHC router, MTP speculative decoding and FP8 weights. Image input works as well.

做法是把官方 [vllm4spec](https://github.com/OpenSenseNova/vllm4spec) fork（commit `7b2e432`）里的 Loop 模型代码移植到 vLLM 0.33 的安装目录里，再在此基础上做提速。补丁不会改动 vLLM 源码仓库，只修改 site-packages 里已安装的那份 vLLM，并且可以重复执行（已打过的会跳过）。

## 提速过程

在许愿编程这类负载（写代码和网页，开思考）下实测的 decode 速度（单位：词元/秒）：

| 步骤 | 词元/秒 |
|---|---|
| fork 原样（eager，强制 TP 4/8 放宽到 2） | 3.2 |
| 开 CUDA Graph | 14 |
| 2 流 2×2 Sinkhorn 的 Triton kernel（原版只写了 4 流，2 流退回 PyTorch，每词元约 2.7 万个小 kernel） | 39.7 |
| 融合 2 流路由与合并（逐位对齐 PyTorch 的 BF16 舍入） | 48.7 |
| MTP 投机解码：2 个草稿，草稿头的 lm_head 用 nvfp4 | 85.6 |
| 主模型 FP8（Marlin W8A16） | **121.8** |

GSM8K 前 200 题（贪心解码，关闭思考）：BF16 是 189/200，FP8 也是 189/200。

## 用法

```bash
# 1. 准备装有 vllm==0.33.0 的环境
# 2. 打补丁（脚本会克隆 vllm4spec，并生成 <模型目录>-mtp 派生目录）
./install.sh /path/to/venv/bin/python /models/SenseNova-Loop-9B
# 3. 启动服务
PY=/path/to/venv/bin/python ./serve.sh /models/SenseNova-Loop-9B-mtp
```

## 补丁清单（`install.sh` 按以下顺序执行）

| 文件 | 作用 |
|---|---|
| `patches/graft_loop.py` | 把 fork 的 `sensenova_loop*.py`、模型注册项和配置类移植进来，TP 白名单放宽到 1/2/4/8 |
| `patches/patch_cudagraph.py` | 取消“必须 enforce_eager”的限制 |
| `patches/patch_mhc2_sinkhorn.py` | 2×2 Sinkhorn 的 Triton kernel |
| `patches/patch_mhc2_fused.py` + `sensenova_loop_mhc2.py` | 2 流路由与合并的融合 kernel（设 `VLLM_SENSENOVA_LOOP_MHC2=0` 可关闭） |
| `patches/patch_mtp.py` | 让 MTP 草稿头复用 vLLM 的 Qwen3_5MTP，并补齐索引里缺的 `mtp.*` 权重 |
| `patches/patch_fp8.py` | 取消量化限制；检查 checkpoint 是否完整时跳过 FP8 缩放参数 |

## 工具

- `tools/export_fp8.py`：把 BF16 导出成离线 FP8 权重，算法与 `--quantization fp8` 在线量化相同，每层一个缩放系数。视觉编码器、词嵌入和 lm_head 保持 BF16。
- `tools/make_awq_dirs.py`：让官方 AWQ 版能在 vLLM 上加载（换回 BF16 的词嵌入、修正 config）。AWQ 在同一负载下测得 150.7 词元/秒，GSM8K 191/200。
- `tools/wishbench.py`、`tools/gsm8k_eval.py`：上面数字所用的测速和测正确率脚本。

## 已知问题

- 补丁只在 vLLM 0.33.0、两张 3090 Ti、TP=2 的组合上验证过。
- 开 MTP 后，贪心解码的输出会和不开时有差异：验证批次的形状不同，会改变 BF16 的舍入。
- vllm4spec 白名单里限制的其他功能（prefix caching、LoRA）仍然不支持。

## 许可

Apache-2.0。vllm4spec 和 vLLM 的代码以各自上游的许可为准。
