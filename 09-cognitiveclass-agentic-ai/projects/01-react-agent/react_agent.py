"""ReAct agent from scratch — tanpa framework, tanpa dependensi eksternal.

Pola: Thought -> Action -> Observation -> ... -> Final Answer.
LLM: endpoint OpenAI-compatible (default 9Router lokal).
Jalankan: python react_agent.py "pertanyaan"
"""
from __future__ import annotations
import json, os, re, sys, urllib.request, datetime

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:20128/v1")
API_KEY = os.environ.get("LLM_API_KEY", "local")
MODEL = os.environ.get("LLM_MODEL", "Whistleblower")

SYSTEM = """Kamu agent ReAct. Jawab dengan format PERSIS:
Thought: <alasan singkat>
Action: <nama_tool>
Action Input: <input>
atau, kalau sudah cukup:
Thought: <alasan>
Final Answer: <jawaban>

Tool tersedia:
- calculator[ekspresi]  : hitung aritmetika, contoh calculator[17*23+5]
- now[]                : waktu sekarang
- wordcount[teks]      : jumlah kata
Maksimal 4 langkah."""

def calculator(expr: str) -> str:
    if not re.fullmatch(r"[0-9+\-*/(). %]+", expr):
        return "ERROR: hanya angka dan operator"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"ERROR: {e}"

def now(_: str = "") -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")

def wordcount(text: str) -> str:
    return str(len(text.split()))

TOOLS = {"calculator": calculator, "now": now, "wordcount": wordcount}

def llm(messages: list[dict]) -> str:
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0}).encode()
    req = urllib.request.Request(BASE_URL + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def parse(text: str):
    action = re.search(r"Action:\s*(\w+)\s*\[(.*?)\]", text, re.S)
    final = re.search(r"Final Answer:\s*(.+)", text, re.S)
    return (action.group(1), action.group(2).strip()) if action else (None, final.group(1).strip() if final else None)

def run(question: str, verbose: bool = True) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(4):
        out = llm(msgs)
        if verbose:
            print(f"--- langkah {step+1} ---\n{out.strip()}\n")
        tool, arg = parse(out)
        if tool is None:
            return arg or out.strip()
        result = TOOLS.get(tool, lambda _: f"ERROR: tool '{tool}' tidak ada")(arg)
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": f"Observation: {result}"}]
    return "BERHENTI: batas langkah tercapai"

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Berapa 17*23+5, dan jam berapa sekarang?"
    print("JAWABAN:", run(q))
