# vllm-sensenova-loop

English | [中文](README_CN.md)

Patches that run [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) fast on stock **vLLM 0.33** with **2× RTX 3090 Ti (Ampere, TP=2)**. Image input works too.

The Loop model code is grafted from the official [vllm4spec](https://github.com/OpenSenseNova/vllm4spec) fork (commit `7b2e432`) into an installed vLLM 0.33. The speedups below are built on top of that. The patches modify only the installed package in site-packages, not a vLLM source checkout. They are idempotent: a patch that is already applied is skipped.

## Speedup ladder

Decode throughput measured on a "wish coding" workload (writing code and web pages, thinking enabled):

| Step | tok/s |
|---|---|
| Fork as-is (eager; TP whitelist relaxed from 4/8 to 2) | 3.2 |
| CUDA Graph | 14 |
| Triton kernel for the 2×2 Sinkhorn (upstream only has a 4-stream kernel; 2 streams fell back to PyTorch, about 27k tiny kernels per token) | 39.7 |
| Fused 2-stream router + merge (BF16 rounding bit-matched to PyTorch) | 48.7 |
| MTP speculative decoding (2 draft tokens, nvfp4 draft lm_head) | 85.6 |
| FP8 main model (Marlin W8A16) | **121.8** |

GSM8K, first 200 questions (greedy, thinking off): BF16 189/200, FP8 189/200.

## Usage

```bash
# 1. An environment with vllm==0.33.0
# 2. Apply the patches (clones vllm4spec and creates the derived <model>-mtp directory)
./install.sh /path/to/venv/bin/python /models/SenseNova-Loop-9B
# 3. Serve
PY=/path/to/venv/bin/python ./serve.sh /models/SenseNova-Loop-9B-mtp
```

## Patches (applied in this order by `install.sh`)

| File | What it does |
|---|---|
| `patches/graft_loop.py` | Copies the fork's `sensenova_loop*.py`, registry entry and config class; TP whitelist set to 1/2/4/8 |
| `patches/patch_cudagraph.py` | Lifts the "requires enforce_eager" restriction |
| `patches/patch_mhc2_sinkhorn.py` | Triton kernel for the 2×2 Sinkhorn |
| `patches/patch_mhc2_fused.py` + `sensenova_loop_mhc2.py` | Fused 2-stream router and merge kernels (disable with `VLLM_SENSENOVA_LOOP_MHC2=0`) |
| `patches/patch_mtp.py` | Reuses vLLM's Qwen3_5MTP for the MTP head; adds the `mtp.*` weights missing from the index |
| `patches/patch_fp8.py` | Lifts the quantization restriction; skips FP8 scale parameters in the checkpoint-completeness check |

## Tools

- `tools/export_fp8.py`: exports offline FP8 weights using the same algorithm as online `--quantization fp8` (per-tensor scale). The vision encoder, embeddings and lm_head stay BF16.
- `tools/make_awq_dirs.py`: makes the official AWQ release loadable in vLLM (BF16 embeddings, config fixes). AWQ reaches 150.7 tok/s on the same workload, GSM8K 191/200.
- `tools/wishbench.py`, `tools/gsm8k_eval.py`: the throughput and accuracy scripts behind the numbers above.

## Known limitations

- Verified only on vLLM 0.33.0, 2× 3090 Ti, TP=2.
- With MTP enabled, greedy output can differ from runs without MTP, because verification batch shapes change BF16 rounding.
- Other features blocked by the vllm4spec whitelist (prefix caching, LoRA) remain unsupported.

## License

Apache-2.0. Code from vllm4spec and vLLM remains under its upstream licenses.
