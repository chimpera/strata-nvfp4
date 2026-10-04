# The post-restore slow-tail investigation

**Status: open. Started 2026-10-04. Owner: guy + Claude sessions.**
Engine: strata-nvfp4 fork (sergqwer base, local main) — **the primary engine as of 2026-10-04**.
Preset in question: `Qwen_V3.8_Flash_125B_NVFP4_c524_g1_nvfp4` (5090/CUDA1, port 12341).

This document is the durable record of the investigation. Update it — not the
chat — with each new finding so any future session can pick it up cold.

---

## 1. The question

Deep-context turns on this engine spend 1.4–2.5 s in the prompt phase even
though ~98% of the prompt is restored from the conversation cache. **Where does
that time go?** The working hypothesis at the start was a "restore floor"
(re-materializing the parked image). That turned out to be WRONG — see §4. The
real suspect is §5: the fresh tail read after a restore runs 5–16× slower per
token than a bulk fresh read.

## 2. How we got here (short)

1. mlab usage stats showed avg read rate ≈ 530 tok/s for ctx ≥ 100k; the
   min-tokens filter (fresh ≥ 8192) showed ≈ 5,989 tok/s. Question: is 530 the
   practical high-context read performance? → No: it's fresh tokens ÷ a wall
   dominated by something else.
2. Binning production `usage_events` (mlab now records `prompt_ms`,
   `prompt_eval_tokens`): turn wall ≈ fixed ~1.2–1.5 s + fresh at ~1.8–3k tok/s
   marginal. Initially mis-attributed the fixed part to "restore floor".
3. The fork's engine logs park/restore timings to stderr — but serve pointed
   engine stderr at /dev/null unless the config has a `log` key. Fixed
   2026-10-04: all three desktop presets now write engine stderr to
   `~/serve/services/mlab/state/logs/instances/<stem>.engine.log`.
4. Controlled A/B probe (two distinct 210k-token conversations, alternating,
   identical bodies on repeats) + the user's live traffic gave exact
   per-request numbers. Findings below.

## 3. Established facts (engine log, 2026-10-04, c524 on 5090)

All from `strata serve:` lines in the engine log — queue-free engine-side
timings (curl wall times are contaminated by single-slot queueing behind other
requests; never use them).

| operation | measured | notes |
|---|---|---|
| park (swap-out), 210k conv | **79–90 ms** steady, 583 ms first-ever | moves ~4.05 GB host-side |
| restore (swap-in), 210k conv | **166 ms** | ≈24 GB/s effective memcpy |
| restore, 45–50k conv | **43–47 ms** | ~1.9 GB image |
| bulk fresh read 210k | 32.7 s @ **6,442 tok/s** | matches bench curve |
| cold fresh read ~11k (new conv) | 2.95 s @ **3,657 tok/s** | no restore involved |
| **post-restore fresh tail** | **≈400–900 tok/s** | the anomaly — see §5 |
| fixed per-request overhead | ~200 ms | 5-fresh-token repeat: 166 restore + 215 ms rest |

Parked-image cost model (two points: 47k→1.89 GB, 210k→4.05 GB):
**bytes ≈ 1.27 GB fixed + ~13 KB/token.** The fixed component (GDN/PLE state,
per house stratas lore "118 MB/mark fixed, position-independent" — here it's
per-image) is why the 64 GB conversation budget fills fast: every parked
lineage, however short, costs ~1.3 GB.

The cache parks/restores **between every turn** when traffic interleaves
(switch-forced): each round-trip ≈ 130 ms at 45k ctx. Cheap, but paid per turn.

Production stats decomposition (mlab `usage_events`, 257 restore-heavy rows,
ctx ≥ 100k): `prompt_ms ≈ ~1.2–1.5 s + fresh/1764` — the fixed part is NOT
restore (restore is 43–166 ms); it is mostly the slow tail's fixed overhead +
the sub-1k tok/s marginal rate. (Earlier session note calling this a "restore
floor" is superseded.)

## 4. What is NOT the problem (ruled out)

- **The VRAM swap.** Park 79–90 ms + restore 43–166 ms = 0.12–0.25 s of a
  1.4–2.5 s turn (5–15%). Even zeroing it barely moves the turn.
- **Chunk granularity** (`--prompt-cache-every`, currently 4096). Turn-boundary
  marks already keep median fresh ≈ the new user message (~640 tokens);
  smaller E attacks overshoot that isn't there, halves divergence coverage
  ((cap−1)·E, cap 6), and costs ~118 MB/mark RAM. Bad trade — rejected.
- **Context depth as such.** Same-size tails at 100k vs 300k ctx: wall moves
  only 2.17 s → 2.45 s.

## 5. The anomaly: the post-restore fresh tail

Same engine, same model, same per-token work in principle:

- bulk fresh (210k, one request): 6,442 tok/s
- cold fresh (11k, new conversation): 3,657–4,584 tok/s
- fresh tail after restore (500–1,500 tokens on a 45–210k conversation):
  **400–900 tok/s**

Examples from live traffic (user's 45k session):
```
restored 43 ms → 1,046 fresh in ~1.9 s   (≈550 tok/s)
restored 47 ms →   622 fresh in ~1.5 s   (≈420 tok/s)
restored 47 ms → 1,230 fresh in ~2.6 s   (≈480 tok/s)
```

The tail is the dominant cost of every deep turn. Whatever makes it slow is
the thing to fix.

## 6. Hypotheses (ranked)

- **H1 — expert-cache churn during/after restore (leading).** Restore calls
  `kvg_ensure(parked + 256)`: the elastic KV must (re)grow to the parked
  length, taking VRAM from the expert cache (slot lending). The tail's tokens
  then route to experts evicted during that regrow → CPU-pool reads over
  PCIe (12 ccd workers, ~150 GB/s ceiling) → crawl. Prediction: engine trace
  lines show `lent slots` / `refilled slots` bursts during tails; tail speed
  improves with a fatter `--expert-cache` or when the expert cache is already
  warm for the tail's routing.
- **H2 — small-chunk latency-bound tail path.** Bulk reads process in 32k
  chunks; the tail path may use much smaller chunks with per-chunk overhead
  (PLE probes, sync). Prediction: tail rate rises with tail length toward
  bulk rate even post-restore (partial support: production bins showed
  235→589→1,107→4,695→6,347 tok/s as fresh grows — but those bins mix
  mechanisms).
- **H3 — PLE row-cache cold for the restored conversation.** `--ple-io ram`
  keeps the 48 GB table in RAM; the row cache is a cache. Bulk reads build
  locality; a restored conversation's tail probes miss. (Weak: PLE per-token
  cost should be similar in both regimes.)
- **H4 — attention over restored KV streams from RAM.** If restored KV pages
  are faulted lazily, each tail chunk's attention reads unpaged KV. (Weak:
  "restored whole" + 24 GB/s copy suggests VRAM-resident.)

H1 and H2 are not mutually exclusive; H1 explains a per-token penalty, H2 a
fixed overhead. The data shows BOTH (~200 ms fixed + ~0.5–0.9k tok/s
marginal).

## 7. Next experiments (in order)

1. **E1 — hot-lineage tail test (decisive for the residency idea).** Same
   conversation, back-to-back requests with a small fresh extension each, NO
   intervening switch (no park/restore). If the tail is fast (~3–6k tok/s),
   the slowness is restore-caused → keeping hot lineages live fixes ~80% of
   the turn. If still slow → fix is in the tail path itself, residency only
   buys the 0.25 s swap. Mechanics: disable thinking so the assistant reply
   replays byte-exact (or replay via the Anthropic endpoint's thinking
   blocks); extend by +N tokens per request.
2. **E2 — engine trace.** The fork has trace lines behind an env var
   (`STRATA_TRACE`; see `src/program/generate.cpp` ~5892/5946/6014:
   `refilled %lld slots`, `lent %lld slots for %lld tokens`, `read %lld
   tokens in %.1f ms`). Run one restored-tail request with tracing on →
   direct evidence for/against H1 (slot lend/refill bursts during the tail)
   and H2 (chunk sizes in the tail read).
3. **E3 — tail-length sweep post-restore.** Extend a restored 210k
   conversation by 100 / 1k / 10k / 50k → marginal-rate curve separates fixed
   overhead from per-token penalty.
4. **E4 — expert-cache A/B.** `--expert-cache` fixed-large vs auto for a
   tail-heavy workload; if H1, large cache shrinks the tail cost.

## 8. The hot-lineage idea (parked until E1/E2 decide it)

Keep the K hottest conversations live (no park on switch) to skip park+
restore. VRAM/RAM cost ≈ image size (~1.27 GB fixed + 13 KB/token; a 265k
lineage ≈ 4.7 GB). Value: 0.12–0.25 s/turn guaranteed, plus possibly the
whole tail fix IF H1 holds (expert cache never churned). Implementation
candidates: idle-delay before parking, or a hot-set exempt from eviction
(expectancy policy port — house stratas 0014 — is the RAM-side cousin).

## 9. Reproduction & artifacts

- Engine log (the data source):
  `~/serve/services/mlab/state/logs/instances/Qwen_V3.8_Flash_125B_NVFP4_c524_g1_nvfp4.engine.log`
  — exists because the presets now set the config `log` key (added 2026-10-04
  to all three desktop nvfp4 presets; blade's ~/apps/nvfp4 already had it).
- A/B probe script (also at /tmp/ab_probe.py — /tmp is wiped on reboot, so
  it is reproduced here):

```python
#!/usr/bin/env python3
"""Build two ~100k-token chat bodies with distinct prefixes (A/B switch probe)."""
import json

def filler(seed, n_chars):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    out, i = [], seed
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(words[i % len(words)] + str(i)); i += 1
    return " ".join(out)

for name, seed in (("a", 0), ("b", 1000)):
    body = {"model": "CUDA1", "stream": False, "max_tokens": 4, "temperature": 0,
            "messages": [{"role": "user", "content":
                f"Conversation {name.upper()}-PROBE {seed}. Remember this text. "
                + filler(seed, 380_000)}]}
    open(f"/tmp/probe_{name}.json", "w").write(json.dumps(body))
```

  Run: `for r in a b a b a b; do curl -s -m 300 http://127.0.0.1:12341/v1/chat/completions \
       -H 'Content-Type: application/json' -d @/tmp/probe_$r.json -o /dev/null; done`
  then read park/restore/prompt lines from the engine log (mark the line count
  first). NOTE: curl wall times include single-slot queueing — only the engine
  log lines are truth. The filler lands ≈210k tokens, not 100k.

- Production-side decomposition: mlab `/mlab/usage/stats?min_tok=&min_ctx=&max_ctx=`
  (per-dimension min-tokens filter + ctx bounds, 2026-10-04) and the
  `prompt_ms` column / `median_prompt_ms` API field.
- Bin analysis snippet lives in session history; key numbers preserved in §3.

## 10. House-patch port queue (fork = primary engine, 2026-10-04)

1. **#214 deferred tool-call rescue + finish gate** — IN PROGRESS (the
   silent-stop class; port from house stratas 0016-series to the fork's
   serve/server.py, upstream 0.1.38 lineage).
2. Engine log key for all presets — DONE 2026-10-04 (this doc exists because
   of it).
3. Conv-id stamps on park/restore/prompt lines (needs a serve→engine protocol
   field; enables per-lineage attribution, cache_report.py-style tooling).
4. Expectancy eviction policy (house 0014 port): tiered eviction instead of
   LRU; complements §8.

## 11. Open questions

- Why is the first-ever park 583 ms vs 79–90 ms steady? (allocation warmup —
  cosmetic, but confirms a fixed setup cost in the image path.)
- Does the expert cache travel with the parked image at all? (snapshot_bytes
  suggests KV + fixed state only; routing/expert warmth apparently not.)
- Is the ~200 ms fixed per-request overhead queue/scheduling or engine-side?
- Why did one early probe read show 3,657 tok/s cold on 11k but 4,584 on
  another 11k? (PLE/page-cache state? worth one look during E3.)
