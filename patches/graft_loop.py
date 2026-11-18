"""把 vllm4spec 的 SenseNova Loop 支持（含视觉）移植进已安装的 vLLM 0.33。
用法：python graft_loop.py <fork 的 vllm 目录> <site-packages 下的 vllm 目录>
幂等：已打过就跳过。"""
import re
import shutil
import sys
from pathlib import Path

fork, dst = Path(sys.argv[1]), Path(sys.argv[2])

for rel in ("model_executor/models/sensenova_loop.py",
            "model_executor/layers/sensenova_loop_mhc.py"):
    shutil.copy2(fork / rel, dst / rel)
    print("copied", rel)


def patch(rel, anchor, text, marker, after=True):
    p = dst / rel
    s = p.read_text()
    if marker in s:
        print("skip", rel, marker)
        return
    i = s.index(anchor)
    if after:
        i += len(anchor)
    p.write_text(s[:i] + text + s[i:])
    print("patched", rel, marker)


# registry
patch("model_executor/models/registry.py",
      '    "SkyworkR1VChatModel": ("skyworkr1v", "SkyworkR1VChatModel"),\n',
      '    "SenseNovaLoopForConditionalGeneration": (\n'
      '        "sensenova_loop",\n'
      '        "SenseNovaLoopForConditionalGeneration",\n'
      '    ),\n',
      '"SenseNovaLoopForConditionalGeneration"', after=False)

# config 类：从 fork 原样取出
fc = (fork / "model_executor/models/config.py").read_text()
m = re.search(r"\nclass SenseNovaLoopForConditionalGenerationConfig\(.*?(?=\nclass )", fc, re.S)
cls = m.group(0)
# fork 白名单只放行验证过的 TP=4/8；4 个 KV 头 TP=1/2 也切得开，PR 64832 在 TP=1/2 下对过 HF
cls = cls.replace("parallel.tensor_parallel_size not in (4, 8)",
                  "parallel.tensor_parallel_size not in (1, 2, 4, 8)")
patch("model_executor/models/config.py",
      "\nMODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {",
      cls + "\n\n", "class SenseNovaLoopForConditionalGenerationConfig", after=False)
patch("model_executor/models/config.py",
      '    "Qwen3_5MoeForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,\n',
      '    "SenseNovaLoopForConditionalGeneration": SenseNovaLoopForConditionalGenerationConfig,\n',
      '"SenseNovaLoopForConditionalGeneration": SenseNova')

# qwen3_5.py：fork 把内层模型类改成可替换的钩子（默认值不变，原 Qwen 模型不受影响）
q = dst / "model_executor/models/qwen3_5.py"
s = q.read_text()
if "model_cls: type[nn.Module] = Qwen3_5Model" not in s:
    s = re.sub(r"(\nclass Qwen3_5ForCausalLMBase\([^)]*\):\n)",
               r"\1    model_cls: type[nn.Module] = Qwen3_5Model\n\n", s, count=1)
    assert s.count("self.model = Qwen3_5Model(") == 1
    s = s.replace("self.model = Qwen3_5Model(", "self.model = self.model_cls(")
    s = re.sub(r"(\nclass Qwen3_5ForConditionalGeneration\([^)]*\):\n)",
               r"\1    language_model_cls = Qwen3_5ForCausalLM\n\n", s, count=1)
    assert s.count("self.language_model = Qwen3_5ForCausalLM(") >= 1
    s = s.replace("self.language_model = Qwen3_5ForCausalLM(",
                  "self.language_model = self.language_model_cls(", 1)
    q.write_text(s)
    print("patched qwen3_5.py hooks")
else:
    print("skip qwen3_5.py")

# envs
patch("envs.py", "    VLLM_USE_V2_MODEL_RUNNER: bool | None = None\n",
      '    VLLM_SENSENOVA_LOOP_MHC_BACKEND: str = "triton"\n'
      '    VLLM_SENSENOVA_LOOP_MHC_FUSION: str = "all"\n',
      "VLLM_SENSENOVA_LOOP_MHC_BACKEND: str")
patch("envs.py", '    "VLLM_USE_V2_MODEL_RUNNER": lambda',
      '    "VLLM_SENSENOVA_LOOP_MHC_BACKEND": env_with_choices(\n'
      '        "VLLM_SENSENOVA_LOOP_MHC_BACKEND", "triton", ["torch", "triton"]\n'
      '    ),\n'
      '    "VLLM_SENSENOVA_LOOP_MHC_FUSION": env_with_choices(\n'
      '        "VLLM_SENSENOVA_LOOP_MHC_FUSION", "all", ["none", "norm", "pre", "merge", "all"]\n'
      '    ),\n',
      '"VLLM_SENSENOVA_LOOP_MHC_BACKEND": env_with', after=False)
