"""decode 速度随上下文长度：真实语料（vLLM 源码）拼到目标长度，流式计时只算首字之后。
用法：python3 ctxbench.py 1000 16000 32000 64000 96000"""
import json, sys, time, glob, urllib.request
B = "http://127.0.0.1:8000"
files = sorted(glob.glob("/root/venvs/vllm-loop/lib/python3.12/site-packages/vllm/**/*.py", recursive=True))
corpus = "".join(open(f).read() for f in files[:600])
def tok(text):
    r = urllib.request.Request(B + "/tokenize", json.dumps({"model": "loop9b", "prompt": text}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))["count"]
for target in map(int, sys.argv[1:]):
    n = target * 3; text = corpus[:n]
    c = tok(text); text = corpus[: int(n * target / c)]
    msg = "下面是一段代码库：\n" + text + "\n\n请用中文写一篇详细的代码结构说明。"
    body = {"model": "loop9b", "messages": [{"role": "user", "content": msg}], "max_tokens": 512, "min_tokens": 512,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if d["choices"] and d["choices"][0]["delta"].get("content") and first is None: first = time.time()
    end = time.time(); g = usage["completion_tokens"]
    print(f"ctx {usage['prompt_tokens']:6d}  prefill {first - t0:6.1f}s  decode {(g - 1) / (end - first):6.1f} tok/s", flush=True)
