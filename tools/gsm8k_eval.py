"""GSM8K 前 N 题，关思考、贪心，并发 2，按最后一个数字判对错。
用法：python3 gsm8k_eval.py <test.parquet> <N> <输出 jsonl>
"""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = pq.read_table(path).to_pylist()[:n]
U = "http://127.0.0.1:8000/v1/chat/completions"


def num(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def ask(r):
    q = r["question"] + "\n\n一步步推理，最后一行只写「答案：」加一个数字。"
    body = {"model": "loop9b", "messages": [{"role": "user", "content": q}], "max_tokens": 768,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    a = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"] or ""
    gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
    pred = num(a.split("答案")[-1]) if "答案" in a else num(a)
    return {"q": r["question"][:80], "gold": gold, "pred": pred, "ok": pred is not None and abs(pred - gold) < 1e-6, "out": a}


with ThreadPoolExecutor(2) as ex:
    res = list(ex.map(ask, rows))
with open(out, "w") as f:
    for x in res:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(f"GSM8K {sum(x['ok'] for x in res)}/{len(res)} = {sum(x['ok'] for x in res) / len(res) * 100:.1f}%")
