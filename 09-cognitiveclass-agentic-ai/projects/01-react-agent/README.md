# 01 — ReAct Agent from Scratch

Implementasi pola ReAct (Thought → Action → Observation → Final Answer) tanpa framework,
hanya stdlib Python + endpoint OpenAI-compatible (9Router lokal).

**Tools:** calculator (eval terbatas regex), now, wordcount.

## Jalankan
```bash
python react_agent.py "Berapa 17*23+5?"
```

## Kenapa ini penting
Pola dasar semua agent modern (LangChain/LangGraph/CrewAI). Memahami loop-nya membuat
framework jadi opsional.
