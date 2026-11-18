#!/bin/bash
# 在已装好 vllm==0.33.0 的 Python 环境里打上 SenseNova-Loop 支持。
# 用法：./install.sh <python> <原模型目录(SenseNova-Loop-9B)>
# 例：  ./install.sh /root/venvs/vllm-loop/bin/python /models/SenseNova-Loop-9B
set -euo pipefail
PY=$1; MODEL=$2; HERE=$(cd "$(dirname "$0")" && pwd)
VLLM=$($PY -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
$PY -c 'import vllm;assert vllm.__version__.startswith("0.33"),vllm.__version__'
FORK=${FORK:-$HERE/.vllm4spec}
if [ ! -d "$FORK" ]; then
  git clone https://github.com/OpenSenseNova/vllm4spec.git "$FORK"
  git -C "$FORK" checkout 7b2e432b1433a78b347a600203e293cae172e39f
fi
P=$HERE/patches
$PY $P/graft_loop.py "$FORK/vllm" "$VLLM"
$PY $P/patch_cudagraph.py "$VLLM"
$PY $P/patch_mhc2_sinkhorn.py "$VLLM/model_executor/layers/sensenova_loop_mhc.py"
$PY $P/patch_mhc2_fused.py "$VLLM" $P/sensenova_loop_mhc2.py
$PY $P/patch_mtp.py "$VLLM" "$MODEL"
$PY $P/patch_fp8.py "$VLLM"
echo "done. MTP 用的派生目录：${MODEL%/}-mtp"
