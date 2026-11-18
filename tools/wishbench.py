"""贴近许愿场景的 decode 测速：6 个写代码/网页的提示词，开思考，流式计时（只算首字之后），汇总接受率。
用法：python3 wishbench.py [max_tokens]
"""
import json, sys, time, urllib.request

B = "http://127.0.0.1:8000"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 768
PROMPTS = [
    "用 Canvas 2D 写一个横版跑酷游戏的物理模块：跑、跳、滑铲、重力、平台碰撞，给出完整的 physics.js。",
    "用 three.js 写一个海边别墅场景的入口文件：相机、灯光、天空、海面、第一人称漫游控制，给出完整代码。",
    "Write a Python script that parses a CSV of sales records, aggregates revenue by month and region, and plots a stacked bar chart with matplotlib.",
    "用 Web Audio API 写一个 120 BPM 的鼓机和贝斯合成器，四小节循环，给出完整的 audio.js。",
    "Implement a single-pendulum rope swing in JavaScript: anchor point, fixed rope length, gravity, release on key press, and landing detection on a platform.",
    "写一个 index.html，带标题页、暂停菜单和结算页（距离、星星、最高纪录存 localStorage），样式要明亮的卡通风。",
]


def metrics():
    d = {}
    for l in urllib.request.urlopen(B + "/metrics").read().decode().splitlines():
        if l.startswith("vllm:spec_decode_num_") and "_created" not in l and "per_pos" not in l:
            k, v = l.rsplit(" ", 1)
            d[k.split("{")[0]] = float(v)
    return d


tot_tok = tot_t = 0.0
m0 = metrics()
for i, p in enumerate(PROMPTS):
    body = {"model": "loop9b", "messages": [{"role": "user", "content": p}], "max_tokens": N,
            "min_tokens": N, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time(); first = None; usage = None
    req = urllib.request.Request(B + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d["choices"]:
                delta = d["choices"][0]["delta"]
                if (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")) and first is None:
                    first = time.time()
    end = time.time(); c = usage["completion_tokens"]
    rate = (c - 1) / (end - first)
    tot_tok += c - 1; tot_t += end - first
    print(f"  #{i} {c} tok  {rate:6.1f} tok/s")
m1 = metrics()
drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
extra = f"  每步多出 {acc / drafts:.2f}" if drafts else ""
print(f"汇总 decode {tot_tok / tot_t:6.1f} tok/s{extra}")
