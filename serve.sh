#!/bin/bash
# 两张 3090 Ti，TP=2，FP8（Ampere 上 Marlin W8A16）+ MTP 2 草稿，26 万上下文，图文。
# 用法：PY=<python> ./serve.sh <模型目录>
#   模型目录：install.sh 生成的 <原目录>-mtp（加 --quantization fp8 在线量化），
#   或 HF 上的 igerry/SenseNova-Loop-9B-FP8（已是 FP8，去掉 QUANT 即可：QUANT= ./serve.sh ...）
export VLLM_USE_V2_MODEL_RUNNER=1
QUANT=${QUANT---quantization fp8}
exec ${PY:-python} -m vllm.entrypoints.openai.api_server \
  --model "$1" --served-model-name loop9b \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 2 --dtype bfloat16 $QUANT --max-model-len 262144 \
  --max-num-batched-tokens 8192 --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image":4,"video":0}' --mm-processor-kwargs '{"max_pixels":1048576}' \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --mamba-ssm-cache-dtype float32 --generation-config auto \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_lm_head_quantization":"nvfp4","use_local_argmax_reduction":true}'
