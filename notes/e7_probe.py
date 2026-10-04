#!/usr/bin/env python3
"""E7: prefix-coupled expert anticipation, measured on realistic text.

A code-agent-shaped conversation: base prompt, then six turns each appending
~1.5-2k tokens of REAL text (different source files, like an agent reading a
repo). Routing trace offsets are snapshotted between requests so each turn's
routed-expert set can be analyzed separately.

Writes /tmp/e7-offsets.json: [byte offset after each request].
"""
import json, os, time, urllib.request

TRACE = "/tmp/e7-routing.bin"
URL = "http://127.0.0.1:12341/v1/messages"
R = "/home/guy/code/strata-nvfp4"

# real text: prose + code, one file per turn (agent-reads-a-repo shape)
SOURCES = [
    (R + "/post-restore-slow-tail.md", 0, 7_000),
    (R + "/serve/frontend.py", 0, 7_000),
    (R + "/src/kernels/cpu/pool.cpp", 0, 7_000),
    (R + "/src/prefill/prefill.cpp", 1200, 8_200),
    (R + "/AGENTS.md", 0, 4_000),
    (R + "/serve/server.py", 1500, 8_000),
]

def call(messages, tag):
    body = json.dumps({"model": "CUDA1", "max_tokens": 24, "messages": messages}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    off = os.path.getsize(TRACE)
    offs.append(off)
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}  trace={off}")

offs = [0]
msgs = [{"role": "user", "content":
         "You are helping audit a C++ inference engine. Here is the base context.\n"
         "Discussion of the slow-tail investigation follows in later turns.\n" * 30}]
call(msgs, "base")
for i, (path, a, b) in enumerate(SOURCES, 1):
    with open(path, errors="ignore") as f:
        f.seek(a)
        text = f.read(b)
    msgs.append({"role": "assistant", "content": "Understood, continuing."})
    msgs.append({"role": "user", "content": f"Turn {i}: review this file.\n```\n{text}\n```\nSummarize briefly."})
    call(msgs, f"turn{i} {os.path.basename(path)[:24]}")

json.dump(offs, open("/tmp/e7-offsets.json", "w"))
print("offsets:", offs)
