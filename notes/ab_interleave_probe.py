#!/usr/bin/env python3
"""The live A/B probe for the working-set feature: TWO interleaved conversations.

Every request switches conversations, so the engine parks one and restores the
other - the shape real traffic has (several clients alternating) and the only
one where the anticipation slice can pay: turn N restores a conversation whose
parked routing then fills the slice.  Each turn appends ~1.5-2k tokens of real
text, like an agent reading a repo.

Usage: ab_interleave_probe.py [nonce]   (a fresh nonce = a fresh pair of
conversations, so repeated runs never restore an earlier run's lineage)
"""
import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:12341/v1/messages"
R = "/home/guy/code/strata-nvfp4"
NONCE = sys.argv[1] if len(sys.argv) > 1 else "0"

# conversation A reads engine sources, conversation B reads serve/python ones
A_SOURCES = [
    (R + "/post-restore-slow-tail.md", 0, 7_000),
    (R + "/src/kernels/cpu/pool.cpp", 0, 7_000),
    (R + "/src/prefill/prefill.cpp", 1200, 8_200),
]
B_SOURCES = [
    (R + "/serve/frontend.py", 0, 7_000),
    (R + "/AGENTS.md", 0, 4_000),
    (R + "/serve/server.py", 1500, 8_000),
]

def build(base_line, sources):
    msgs = [{"role": "user", "content":
             f"You are helping audit a C++ inference engine. Run {NONCE}.\n"
             "Discussion of the slow-tail investigation follows in later turns.\n" * 30 +
             base_line}]
    turns = []
    for path, off, n in sources:
        text = open(path, errors="replace").read()[off:off + n]
        msgs.append({"role": "user", "content": "Read this file and keep it in context:\n```\n" + text + "\n```\nBriefly confirm."})
        turns.append(list(msgs))
    return turns

def call(messages, tag):
    body = json.dumps({"model": "CUDA1", "max_tokens": 24, "messages": messages}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}")

# warm both conversations (the base turns), then interleave the file turns
a_turns = build("Conversation A: the engine's C++ side.", A_SOURCES)
b_turns = build("Conversation B: the server's python side.", B_SOURCES)
call(a_turns[0][0:1], "A-base")
call(b_turns[0][0:1], "B-base")
for i in range(1, 3):
    call(a_turns[i], f"A-turn{i}")
    call(b_turns[i], f"B-turn{i}")
