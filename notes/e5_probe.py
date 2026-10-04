#!/usr/bin/env python3
"""E5: phase breakdown of a cold ~1k tail vs a fast ~16k extension.

One conversation: fresh base, then a 1k extension (slow regime), then a 16k
extension (fast regime). STRATA_PREFILL_TIMING prints the per-run phase
breakdown to the engine log; comparing the two shows which phase holds the
cold ramp.
"""
import json, time, urllib.request

URL = "http://127.0.0.1:12341/v1/messages"
words = ["golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]

def filler(seed, n_chars):
    out, i = [], seed
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(words[i % len(words)] + str(i)); i += 1
    return " ".join(out)

def call(messages, tag):
    body = json.dumps({"model": "CUDA1", "max_tokens": 24, "messages": messages}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}")

msgs = [{"role": "user", "content": "Conversation GOLF base. " + filler(2, 190_000)}]
call(msgs, "base fresh ~100k")
msgs.append({"role": "assistant", "content": "ok"})
msgs.append({"role": "user", "content": "Cold short extension. " + filler(41, 2_000)})
call(msgs, "cold tail ~1k (slow regime)")
msgs.append({"role": "assistant", "content": "ok"})
msgs.append({"role": "user", "content": "Long extension to reach the fast regime. " + filler(43, 30_000)})
call(msgs, "extension ~16k (fast regime)")
