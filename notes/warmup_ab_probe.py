#!/usr/bin/env python3
"""Warm-up A/B/C, synthetic: one DEEP main conversation on the main engine,
then measure its next-append cost after three kinds of gap, round-rotated:

  U      - a timed gap, no traffic (the main conversation stayed live)
  I-side - a small UNRELATED task ran on the SIDE instance (CUDA0, port 12340)
  I-main - the same unrelated task ran on the MAIN engine (CUDA1, port 12341)

Only the interruption's target changes.  The engine log carries the
decomposition (park/restore ms, the elastic K/V gave/moved churn, read
walls); the log byte offset printed per call slices it.
"""
import json, os, time, urllib.request

MAIN = "http://127.0.0.1:12341/v1/messages"
SIDE = "http://127.0.0.1:12340/v1/messages"
LOG = "/home/guy/serve/services/mlab/state/logs/instances/Qwen_V3.8_Flash_125B_NVFP4_c524_g1_nvfp4.engine.log"
R = "/home/guy/code/strata-nvfp4"

def logoff():
    return os.path.getsize(LOG)

def call(url, model, messages, tag, max_tokens=8):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "messages": messages}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}  logoff={logoff()}")
    return time.time() - t0

# ---- build the deep main conversation: base + 6 file turns (~50k tokens)
msgs = [{"role": "user", "content":
         "You are auditing a C++ inference engine; the following turns carry the "
         "repository's own files. Acknowledge each briefly.\n" * 20}]
SOURCES = [(R + "/src/prefill/prefill.cpp", 4000, 13000),
           (R + "/serve/server.py", 2000, 11000),
           (R + "/src/kernels/cpu/pool.cpp", 0, 10000),
           (R + "/src/core/expert_cache.cpp", 0, 9000),
           (R + "/serve/frontend.py", 0, 11000),
           (R + "/post-restore-slow-tail.md", 2000, 12000)]
print("== building the deep main conversation")
for path, off, n in SOURCES:
    text = open(path, errors="replace").read()[off:off + n]
    msgs.append({"role": "user", "content": "Keep in context:\n```\n" + text + "\n```\nAcknowledge briefly."})
    call(MAIN, "CUDA1", msgs, f"build {path.split('/')[-1][:24]}")

APPEND_TEXT = None
def main_append(tag):
    global APPEND_TEXT
    if APPEND_TEXT is None:
        APPEND_TEXT = open(R + "/AGENTS.md", errors="replace").read()[0:2400]
    msgs.append({"role": "user", "content": "Also keep this:\n```\n" + APPEND_TEXT + "\n```\nAcknowledge briefly."})
    return call(MAIN, "CUDA1", msgs, tag)

INTERRUPT = ("Quick unrelated task, answer in one word: a recipe calls for folding "
             "whipped egg whites into a chocolate batter. Is that folding a mixing, "
             "a cutting or a heating step? Answer: mixing")

def interrupt(target, tag):
    call(target, "CUDA0" if target == SIDE else "CUDA1",
         [{"role": "user", "content": INTERRUPT}], tag)

# ---- rounds, rotated so drift hits all arms equally
CONDS = ["U", "I-side", "I-main"]
URLS = {"I-side": SIDE, "I-main": MAIN}
results = {c: [] for c in CONDS}
for r in range(5):
    for c in CONDS:
        if c != "U":
            interrupt(URLS[c], f"  r{r} {c} interrupt")
        else:
            time.sleep(1.5)   # a gap of the interrupt's rough wall, no traffic
        results[c].append(main_append(f"r{r} {c} MAIN-APPEND"))
    CONDS = CONDS[1:] + CONDS[:1]

print("\n== main-append walls by condition")
for c in results:
    v = results[c]
    print(f"{c:7s} mean {sum(v)/len(v):.2f}s   runs " + " ".join(f"{x:.2f}" for x in v))
