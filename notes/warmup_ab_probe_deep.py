#!/usr/bin/env python3
"""Warm-up A/B/C at REAL depth: the shallow probe (warmup_ab_probe.py) only
built ~26k tokens — park was 121 ms, restore 26 ms, and no arm separated.
This one quadruples the file turns (~110k tokens) so the switch machinery
runs where the felt 2 s penalty lives, then repeats the rotated U / I-side /
I-main rounds with the same 427-token appends and 5-token interrupts."""
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
    print(f"{tag}: {time.time()-t0:.2f}s  in={d['usage']['input_tokens']}  logoff={logoff()}",
          flush=True)
    return time.time() - t0

# ---- build the deep main conversation: 4 windows per file (~110k tokens)
msgs = [{"role": "user", "content":
         "You are auditing a C++ inference engine; the following turns carry the "
         "repository's own files. Acknowledge each briefly.\n" * 20}]
FILES = [R + "/src/prefill/prefill.cpp",
         R + "/serve/server.py",
         R + "/src/kernels/cpu/pool.cpp",
         R + "/src/core/expert_cache.cpp",
         R + "/serve/frontend.py",
         R + "/post-restore-slow-tail.md"]
print(f"== PROBE_DEEP_START logoff={logoff()}", flush=True)
n_turn = 0
for path in FILES:
    text = open(path, errors="replace").read()
    for off in range(0, len(text), 12000):
        win = text[off:off + 12000]
        if len(win) < 2000:
            break
        n_turn += 1
        msgs.append({"role": "user", "content":
                     f"Keep in context (part {off//12000 + 1}):\n```\n{win}\n```\nAcknowledge briefly."})
        call(MAIN, "CUDA1", msgs, f"build {n_turn:02d} {path.split('/')[-1][:20]} +{len(win)}ch")

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
for r in range(4):
    for c in CONDS:
        if c != "U":
            interrupt(URLS[c], f"  r{r} {c} interrupt")
        else:
            time.sleep(1.5)   # a gap of the interrupt's rough wall, no traffic
        results[c].append(main_append(f"r{r} {c} MAIN-APPEND"))
    CONDS = CONDS[1:] + CONDS[:1]

print("\n== deep main-append walls by condition", flush=True)
for c in results:
    v = results[c]
    print(f"{c:7s} mean {sum(v)/len(v):.2f}s   runs " + " ".join(f"{x:.2f}" for x in v), flush=True)
