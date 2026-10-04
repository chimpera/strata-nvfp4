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

## 5. The anomaly — RESOLVED as a warm-up ramp (2026-10-04, E1+E3)

The "post-restore slow tail" is not post-restore at all. It is a per-read
warm-up ramp: **the first ~1–2k tokens of any fresh read run at ~700–850
tok/s; the rate climbs to ~6.2k tok/s by ~10–16k tokens.** A conversational
turn's fresh tail (a few hundred to ~1.5k tokens) lives entirely inside the
ramp, so every deep turn pays ~0.6–1.5 s of prompt phase.

Evidence:

**E1 — hot vs restored tail (same conversation, same tail size):**
```
req2 HOT tail (no switch between requests):    read 1437 tokens (batched) in 1766 ms → 814 tok/s
req3 RESTORED tail (switch away and back):     read 1443 tokens (batched) in 1748 ms → 825 tok/s
```
Identical → restore/park exonerated. Slot lend/refill measured 0.1 ms +
68 ms — also exonerated. Both paths use the same `batched` read mode.

**E3 — tail-length sweep (extensions of one conversation, each request's
fresh read = only the new part):**
```
  1,009 tokens → 1,194 ms   845 tok/s
  3,809 tokens → 1,824 ms → 2,088 tok/s
 16,102 tokens → 2,590 ms → 6,217 tok/s
 49,652 tokens → 8,028 ms → 6,183 tok/s
107,787 tokens (base bulk) → 15,892 ms → 6,784 tok/s
```

This also re-explains §3's production stats: the "~1.2–1.5 s fixed" in the
turn decomposition IS the ramp's near-fixed cost for sub-2k tails, and the
production bin climb (235→589→1,107→4,695→6,347 tok/s with fresh size) was
the same ramp seen through noisy aggregates.

**E4 — CPU signature (thread utime sampling around requests):**
```
bulk 105k fresh read:  1.1 cores of engine thread CPU
cold ~1k tail:         2.3 cores   (reproducible, 2nd tail same)
```
The slow path does MORE host work while the fast bulk does almost none; the
pool (12 workers) is nowhere near saturated → not bandwidth-bound;
serialized host work per token is the shape.

## 6. Mechanism — current hypotheses (post-E1/E3/E4)

- **H-A (leading): PLE n-gram host path on novel text.** The 48 GB PLE table
  is in the prefill path (the `--ple-io` cliff history is a prefill
  phenomenon). Novel n-grams pay per-token host work (row insert/probe);
  REPETITIVE text finds matches and skips compute — which is why bulk reads
  of the probe filler run at 6.2–6.8k tok/s on only 1.1 cores. Fits the CPU
  signature (~2.3 cores), fits the ramp (table locality builds as the read
  proceeds), fits the fork's own `--ple-io` cliff lore.
- **H-B: serialized expert-miss latency.** Novel routing misses the resident
  expert set; misses serialize on pool round-trips (~ms each) rather than
  saturate bandwidth. Not excluded; the ~2.3-core signature is compatible
  with a small degree of this too.
- **Retired: restore/park/swap causes (E1), slot lend/refill (trace),
  attention depth (E3's 16k-at-depth-107k reaches 6.2k), chunk mode
  (trace: both `batched`).**

**Filler-bias caveat (important for reading every number in this doc):** the
probe filler is maximally repetitive synthetic text, which the PLE loves.
The "steady state" 6.2–6.8k may be PLE-inflated; real novel text's steady
state is probably lower (production real-text fresh-heavy reads: 4.7–6.3k).
The RAMP shape is the finding, not the exact asymptote.

## 7. Mechanism CONFIRMED (E5, 2026-10-04 night): expert-cache misses

`STRATA_PREFILL_TIMING=1` (already built into the engine — prefill.cpp's
PfTimer) prints a per-run phase breakdown. Cold tail vs warm, same
conversation:

| phase | bulk 100k read | cold 905-tok tail | 14.9k extension (warm) |
|---|---|---|---|
| wait copy | 7.4% | **42.6%** | 1.1% |
| dequant | 8.3% | **31.1%** | 12.1% |
| gemm gate/up + down | 19.7% | 12.1% | 20.6% |
| ple | 0.6% | 0.1% | 0.5% |
| everything else | ~64% | ~14% | ~66% |

**The cold tail spends 73.7% of the GPU timeline in `wait copy` + `dequant`**
— expert-cache misses. Novel tokens route to experts absent from the
8,651-slot / 22.8 GB VRAM cache; each miss pays a host→VRAM copy plus FP4
dequant, serialized against compute (the profiler's own note: "a gap where
the GPU waits (for the host's expert grouping, or for an expert's copy) lands
on the phase that was waiting"). As the read proceeds the working set becomes
resident — wait copy collapses to 1.1% — and the rate reaches compute-bound
~6.2k tok/s. PLE is exonerated (≤0.6% everywhere; its ~130 ms host block
overlaps). H-A retired; H-B confirmed; E4's ~2.3 cores = the host staging
feeding those serialized copies.

The whole chain, resolved: mlab's 530 tok/s → small fresh tails → tails live
in the cold-miss regime → misses are the cost of NOVEL ROUTING, not of
parking, restoring, swapping, chunk size, or PLE.

**Code-level shape (prefill.cpp read, same night):** two expert paths per
chunk. Chunks ≥ `stream_all_min()` = 1024 tokens take the "streamed walk" —
every non-resident expert of every layer streams through a 384-slot ring,
overlapped with the attention halves; but a ≥1k chunk touches ~10k
(token,expert) pairs → nearly all 512 experts per layer, each streaming its
whole ~2 MB blob for possibly few rows — a roughly FIXED ~1–1.5 s of PCIe
traffic per chunk that compute hides only at large chunk sizes. Chunks
< 1024 take the older path: host grouping with a `cudaStreamSynchronize`
PER LAYER plus serialized expert copies (E5's 905-token tail: 42.6 %
wait-copy). So the E3 "ramp" is mostly amortization of the per-chunk
streaming volume, not cross-chunk cache warming — which is exactly why it
was indifferent to hot-vs-restored (E1).

**E6 — tested and rejected (2026-10-04):** `STRATA_PREFILL_STREAM_MIN=256`
(lowers the streamed-walk threshold so small tails take the ring path):
the 905-token tail got WORSE — 1,750 ms vs 1,148 ms default (517 vs
814 tok/s). The 1024 floor is real (matches the source comment: "below
~1,000 tokens the output changed on Q2_0"; here even the timing regresses).
Reverted; the preset is back to defaults.

**Filler caveat, sharpened:** the probe filler routes near-uniformly (touches
~every expert per layer per chunk). Real text routes far more selectively
(the expert-profile exists because real routing is skewed), so production
tails miss a smaller fraction — production's 400–900 tok/s sits between the
probe's 517 and the bulk rate. The probe numbers bound the phenomenon; they
exaggerate its per-token cost for natural text.

## 7b. E7 (2026-10-04 night): anticipation is measured, and the cache ignores it

The user's question: can we anticipate needed experts — track hot experts
along with prefixes? **Yes, and the measured opportunity is large.**

Method: `--dump-routing` (engine flag, per-token routing records) on a
code-agent-shaped conversation — base + six turns, each ~1.8–2.8k tokens of
REAL text (the investigation doc, frontend.py, pool.cpp, prefill.cpp,
AGENTS.md, server.py). Scripts preserved in `notes/e7_probe.py` +
`notes/e7_analyze.py`; raw trace at /tmp/e7-routing.bin (wiped on reboot).

Findings:

1. **Real text routes selectively:** ~90–110 distinct experts per layer per
   turn (of 512) — vs the probe filler's ~all 512. So real tails have a
   small working set.
2. **Routing persists across turns:**

   | predictor of turn N's (layer,expert) pairs | coverage |
   |---|---|
   | turn N−1 alone | **87.3%** |
   | union of the conversation so far | **94.3%** |

3. **But the cache never captures it.** EVERY turn still paid
   wait copy 700–812 ms + dequant 355–415 ms (60–70% of its ~1.75 s read) —
   turn 6 as slow as turn 1 despite 87% overlap. Code-level cause, verified:
   **prefill misses never admit into the resident cache.** `admit()` is
   called only from `expert_hit_run` (the DECODE dispatch,
   src/core/expert_source.cpp:2126), the peer/remote tiers, and the boot-time
   profile load (generate.cpp:3227). Prefill streams its misses through the
   ring and gives the slots back — the conversation's recurring prefill
   working set re-streams EVERY turn, forever. Meanwhile the cache is
   deliberately no-eviction once full ("full: no eviction, deliberately",
   ExpertCache::admit), so decode admissions + the boot profile own the
   8,651 slots and prefill's set never enters.
4. Half the machinery for the fix already exists: the serve loop keeps a
   routing **heat** map (feeds `--expert-profile-save`, #477) — the counting
   side. It just isn't wired to prefill admission or to any prefetch.

Implication: with 87–94% of a turn's experts prefetched, wait copy drops
toward the warm 1.1% (E5) → turns ~1.75 s → ~0.6 s. This is the single
highest-leverage fix found.

## 7c. E8–E10 (2026-10-04 night): the gather is unconditional — the full cost tree

Three probes closed the remaining questions:

- **E8 — conversation-profile residency:** booted the cache with a profile
  built from E7's own routing (`tools/make_profile.py --no-base`, 7,878
  conversation pairs ranked first, all 8,651 slots pre-filled — verified in
  the boot log). Result: turns UNCHANGED (wait copy 44–47%). Residency alone
  does not fix the tails.
- **E9 — routed-only staging everywhere** (`STRATA_PREFILL_STREAM_MIN=999999`):
  forces every chunk onto the per-layer path that stages only ROUTED experts
  (prefill.cpp:2170 builds `order` from `cnt > 0`; resident ones skip
  staging, line 2388/2404). Result: 5–16% faster turns; wait copy still
  ~470–595 ms + dequant ~480–530 ms (63–67%).
- **E10 — routing transfers across cache states:** reran the same
  conversation under a different profile + staging path with a fresh routing
  trace: **88.1% of the rerun's routing was covered by E7's trace** (7,878 vs
  8,706 pairs, union 8,916). Numerics drift ≈ 12% — the routing signature is
  robust across cache states, so anticipation from yesterday's traffic is
  valid.

**The complete cost tree of a real-text turn (~2.4k fresh tokens):**

1. **The gather is unconditional (the dominant fixed cost).** `compute()`
   (prefill.cpp:2478) marks `kPfDequant` and runs `mmq::gather_native` for
   EVERY routed expert — resident or streamed — every chunk: ~4.4k distinct
   pairs × ~2.6 MB ≈ **~11 GB of device-to-device gather per turn** ≈ the
   measured ~500 ms, regardless of residency. This is why E8 changed
   nothing.
2. **Wait-copy = pipeline stalls, not bandwidth.** With ~1-in-8 streamed
   experts scattered through the MMQ gather groups, nearly every group's
   flush (line 2452) waits on a copy event — high wait-copy %, modest actual
   bytes.
3. Compute itself is ~250 ms (gemms + attention) — the floor.

Also from E8's mechanics: the ≥1024 streamed walk streams EVERY non-resident
expert of every layer — routed or not (plan at prefill.cpp:1537) — because
host-side routing isn't known at plan time. For real text that over-streams
~5×; it hides only at large chunk sizes (E5's warm 14.9k: wait copy 1.1%).

## 7d. E11–E13 (2026-10-04, late): the wall is the walk's over-issue

**E13 — the streamed-bytes telemetry (commit d7e6a50).** One log line after
each timing block ("streamed N experts, X GiB host->device"), counting both
staging paths. On the c524 preset, the interleaved probe (§9's shape), per
~2-4k-fresh turn:

| turn | fresh tok | streamed experts | GiB H2D | wait copy |
|------|-----------|------------------|---------|-----------|
| A1 | 4,321 | 16,971 | 43.8 | 549 ms (31%) |
| B1 | 2,205 | 16,706 | 43.0 | 791 ms (46%) |
| A2 | 2,822 | 16,817 | 43.3 | 697 ms (40%) |
| B2 | 2,196 | 16,706 | 43.0 | 802 ms (47%) |

The cache holds 8,650 slots of 24,576 pairs, so ~17.9k experts are
non-resident — **each turn streams essentially the ENTIRE non-resident set**,
while the routing picks ~4.8k pairs (~6-8 GiB of true misses). At the probed
26.3 GB/s, 43 GiB is ~1.65 s of DMA against a 1.75 s GPU timeline: the copy
stream is saturated wall-to-wall and "wait copy" is the compute stream
catching up to it. The walk's plan is built in expert-id order BEFORE the
host routing is consulted (the code says so at the plan builder), so the
issuer walks over - and DMAs - every non-resident expert up to each layer's
highest routed id. **The over-issue, not the misses, is the deep-turn wall.**
This subsumes §7c's cost tree: the D2D gathers (§8.1) and the misses were
both riding under this stream.

**E11 — forcing the small-chunk path (`STRATA_PREFILL_STREAM_MIN=999999`):
REJECTED.** The idea was to land every chunk on `stage_one`, which stages
`order` (routed experts). Measured: the first read collapsed to 61 s
(47.8 s GPU timeline, 174k experts staged - a cold-cache pathology in that
path), and the turns were at best ~5% better while STILL streaming 13-15k
experts each. The path is not routed-filtered either; it just pays per-expert
waits on top.

**E12 — the idle 3090 as a remote expert tier (`CUDA_VISIBLE_DEVICES=g1,0` +
`--expert-cache-device1 6000`): REJECTED for this preset.** The card worked
(1.9-3.1k expert entries per request, packed returns 23-30 MiB vs 143-159
full-row) but the primary waited 46-186 ms per request on it, wait copy ROSE
to 46-53% (the tier's input staging competes with the miss stream on the same
PCIe), the 5090's own cache shrank 8,650 -> 6,146 slots for the tier's
contexts, and decode slowed (24 tok in 295-340 ms vs 197-250). Net ~-250 ms
per turn. Answers §11's peer prediction: coverage does not pay while the
stream itself is 5-7x over-issued - fix the over-issue first, then re-measure
the tier against a right-sized stream.

**E14 — fix §8.0 BUILT AND MEASURED (STRATA_PF_ROUTED_STREAM=1, same night).**
Each layer's stream segment is filtered to the experts its routing actually
picks, at the moment the layer's host grouping makes the routing known
(single-threaded issuing while on - the filter rewrites the seq in place).
Same probe, read walls: 1,983/1,924/1,956/1,922 ms -> 1,708/1,511/1,609/
1,545 ms (-14 to -21%); wait copy 548-804 -> 405-472 ms; streamed 43.0 ->
33.3-34.2 GiB/turn; parity gate PASS byte-exact default-off. Left ON in the
c524 preset.

**The corrected physics.** E13's "~4.8k routed pairs" undercounted for
multi-k-token turns: T x K routing instances over 512 experts per layer means
a 2-4k-token chunk routes NEARLY EVERY expert, so the filter's headroom is
the ~20% of non-resident experts the chunk happens not to touch - the rest
must cross PCIe once per turn regardless. The deep-turn wall is now cleanly
`(non-resident experts touched) x 2.7 MB / 26.3 GB/s ~= 1.3 s of DMA`,
overlapped against ~1.0 s of compute. The levers that remain, in order:
cache coverage (more slots), link bandwidth (the 5090 idles at Gen1 and
probes 26.3 GB/s - Gen4 x16), or shrinking the touched set per chunk.

**E15 — the 3090 tier ON TOP of the routed stream: still REJECTED.**
2,159/2,061/2,086/1,996 ms - worse than E14 alone. Even against a
right-sized stream, the tier's per-layer sync waits and input staging cost
more than the 6,000 experts x 2.7 MB it removes from the miss stream.

**E16 — the warm-up-after-switch hypothesis: REFUTED (2026-10-04).**
The felt cost of a side call was assumed to be the ~0.3-0.6 s switch PLUS a
warm-up the main agent pays on its next turn (elastic K/V re-grow, expert
churn). Synthetic A/B/C on the live c524 preset (notes/warmup_ab_probe.py at
~26k depth, notes/warmup_ab_probe_deep.py at ~144k; a deep main conversation,
interrupted between turns by a 5-token unrelated task, round-rotated: U =
quiet gap, I-side = task on the side instance, I-main = task on the MAIN
engine; the next main append (427 fresh tokens) is the measurement):

| depth | U | I-side | I-main |
| --- | --- | --- | --- |
| ~26k | 1.45 s | 1.49 s | 1.43 s |
| ~144k | 2.09 s | 2.19 s | 2.13 s |

No warm-up term at either depth: the append after a main-engine interrupt
equals the uninterrupted append (bounded <= ~0.1 s at 144k). The engine log
shows the predicted machinery all firing per I-main interrupt at 144k -
park 142k tokens in 295 ms; K/V trimmed to 8192, 691 slots back, refilled
from the profile; re-grown to 147456 cells, 94-282 hotter experts moved
(~0.3-0.8 GB of D2D, tens of ms); restore 142k tokens (checkpoint) in
112 ms - and the refill+regrow land INSIDE the interrupt's own wall, not in
the next append. The restore is RAM-resident and checkpoint-based, so it
stays ~O(delta), not O(image).

What the switch DOES cost is the interrupt's own serialized wall, and that
scales with the parked conversation: 0.31 s at 26k -> 0.62 s at 144k (park
121->295 ms, restore 26->112 ms, and the trim/refill churn slows the tiny
read itself 24 -> 10 tok/s). So every side call answered by the MAIN engine
gets linearly more expensive as the main conversation deepens, while the
side instance answered the same calls in a flat 0.15 s and left the main
engine's log empty (I-side ~= U). That is the measured case for the side
instance: not eviction protection but latency isolation.

**E17 — v2, the SUBSTANTIVE interrupt: still no warm-up, and the mechanism
is closed (2026-10-04).** E16's interrupt was 5 tokens - too thin to test
the hypothesis (prefill misses stream without admitting; only decode
admits). v2 (notes/warmup_ab_probe2.py) interrupts with a NEW ~480-token
unrelated-domain conversation (cooking / marine biology / contract law /
prosody, no prefix reuse) that DECODES 250 tokens - dispatch admits its
experts and evicts colder (the main's). Proof the pollution is real: the
post-interrupt K/V re-grow relocated 656-696 experts, vs 94-282 in E16.

Client walls suggested +0.7 s for I-main (U 2.95 / I-side 3.06 / I-main
3.67 s) - a confound: I-main's appends happened to draw the 250-token
answers (walls include generation). The engine log splits prompt from
decode and kills it - next main-append READ walls: U 1552 / I-side 1644 /
I-main 1633 ms (the ~90 ms spread between arms is rotation drift; I-side,
whose interrupts ran on another GPU, sits ABOVE I-main); decode tok/s: U
98 / I-side 112 / I-main 119 - no cold-decode ramp. Streamed per append:
U 31.9 / I-side 23.5 / I-main 30.5 GiB - no arm pattern.

Why displacement can't hurt here: a ~430-token chunk routes essentially
every expert per layer (3440 draws over 512 bins misses <1 per layer), so
the append streams the touched-minus-resident complement regardless - a
wall set by cache CAPACITY (~7,959 of 31,744 experts resident), not by
which 9% of slots the interrupt swapped. Marginal cost = displaced-and-
touched-and-was-a-hit x 2.7 MB / 26.3 GB/s ~= 20-80 ms. And the trim-time
refill re-seeds the cache from the LEARNED profile (the main conversation's
own hot set) before the interrupt decodes, shrinking that intersection
further - the #477 profile is the warm-up fix, already deployed.

What would make it bite: minutes of main-engine decode on unrelated
traffic (a sub-agent), admitting thousands of its experts. That is
precisely the workload the side instance exists to absorb - and even then
the append wall is capacity-bound, so the damage is bounded to the moved
set's share of residency.

Caveats: the interrupt generated 8 tokens (real tool calls generate more,
which adds to the interrupt's wall but not to any warm-up); the append's
~2.1 s small-tail floor at 144k could mask a sub-0.1 s term; no cache-slot
pressure (18 of 48 slots, 0 evictions).

## 8. Fix directions (ranked, fork-side, upstream-able)

*(2026-10-04 postscript: 1 and 2 are BUILT (commits cee2509, 2566a60,
0b591ef) and the live A/B measured them — neither moves the wall on the
c524 preset, and the mechanism is now understood: the D2D gather was
overlapped with the H2D miss stream (full-duplex PCIe), and the profile
already covers repo-text conversations so the slice's thrash guard
correctly skips. Full numbers:
bench/results/2026-10-04-expert-residency/README.md.)*

**0. Route the stream - BUILT, MEASURED, ON in production (E14 above;
STRATA_PF_ROUTED_STREAM, commit follows).)** Build the
streamed walk's per-layer seq FROM THE CHUNK'S ROUTING (the host grouping's
ids/cnt tables, already computed before the walk runs) instead of the
id-ordered non-resident list. Measured size: 43 GiB -> ~6-8 GiB per turn,
1.65 s of DMA -> ~0.3 s; the waits should collapse toward the necessary
transfer and the turn wall plausibly drops 1.9 s -> ~1.1-1.2 s (compute's own
~1.0-1.2 s becomes the floor). The code notes the plan is "built before host
routing known" — that ordering is the bug's address. Verification is already
standing: the streamed-GiB telemetry line (d7e6a50) plus the phase blocks,
the parity gate, and §9's probe. After it lands, RE-RUN E12 (the 3090 tier
against a right-sized stream) and re-check the anticipation slice's thrash
guard - both were measured against the over-issuing walk and may flip
positive once the stream is honest.

1. **Pre-gathered resident slots (new top pick — mechanism-independent).**
   Store resident experts in MMQ group layout (gather ONCE at fill/admission
   instead of per use). Kills the ~500 ms/turn unconditional gather for every
   resident expert and removes most group-flush waits. Needs nothing else to
   be true. Cost: fill-time conversion (one extra pass per admission), and
   the slots stop being byte-identical to the pack blobs (fill_slot semantics
   change; peer/streamed paths unaffected).
2. **Prefix-coupled expert working set (E7/E10 — coverage measured 87–94%).**
   Track the conversation's routed-expert set (heat-map infra exists;
   prefill routing currently doesn't feed it) and make it resident per
   conversation. With fix 1 in place this is what makes coverage PAY: the
   88%-resident turn computes straight from pre-gathered slots, and only the
   ~12% streamed tail pays gather+copy. Combined ceiling: turn read
   ~1.7 s → ~0.5–0.7 s.
3. **Wire prefill routing into admission** (the enabler for 2): count
   prefill routing into the existing heat map and admit prefill misses. The
   cache is no-eviction once full, so this needs the conversation-tier shape
   of fix 2 (wholesale swap) or a rebuild-from-ranked path
   (`rank_learned_profile` exists) rather than plain admission.
4. **More resident slots** — 8,651 now (VRAM-bound on the 32 GB 5090). The
   2×3090 peer preset holds 13,056 — prediction: cheaper tails; one A/B
   when convenient (note: without fix 1, extra slots only trim the wait-copy
   stalls, not the gather).
5. Not fixes: hotter conversation cache (E1 — buys only the 0.12–0.25 s
   park+restore), smaller checkpoint chunks (§4), PLE tuning (exonerated),
   lowering `stream_all_min` (E6 — regresses), residency alone (E8 — the
   gather dominates), routed-only staging alone (E9 — 5–16%).
6. NOTE on "route-ahead" in the strong sense (compute routing for incoming
   tokens before the read): not possible for deep layers — routing needs
   the hidden states of all previous layers. The practical form of
   anticipation IS the prefix-coupled set (fix 2): yesterday's routing
   predicts today's at 87–88%, robust across cache states (E10).

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
5. **The fork's first vendored llama.cpp edit — DONE 2026-10-04 (B1)**:
   `third_party/llama-mmq-ptr-table.patch` (40 insertions in
   `ggml/src/ggml-cuda/mmq.cuh`), applied to the FetchContent checkout at
   configure time (CMakeLists.txt, reverse-check first so re-runs are no-ops).
   Two parts: (a) STRATA_MMQ_PTR_TABLE — one weight base pointer per expert
   for `src/prefill/mmq_direct.cu`, ifdef-guarded so every other TU compiles
   the gathered-image kernel of today (mmq_args gains one unconditional
   trailing `x_ptrs` field so the struct is one type everywhere); (b) an ids
   overread guard on the three `ids_dst_shared` fills — a partial J-tile at
   the last expert read up to J-1 entries past `col_high` (compute-sanitizer
   on the new test caught it in the STOCK kernel; values were unused, but
   production's `ids_identity` arena read into neighbors every launch).
   Upstreamable as-is. Regenerate against a moved GIT_TAG with:
   edit `third_party/llama.cpp` (the pinned reference checkout), `git diff >
   third_party/llama-mmq-ptr-table.patch`, `git checkout --` it back.

## 11. Open questions

- Why is the first-ever park 583 ms vs 79–90 ms steady? (allocation warmup —
  cosmetic, but confirms a fixed setup cost in the image path.)
- Does the expert cache travel with the parked image at all? (snapshot_bytes
  suggests KV + fixed state only; routing/expert warmth apparently not.)
- `/metrics` per-request `hit_rate` measures the DECODE phase only ("while
  writing the answer") — it read 0.96 on a crawling tail. A prefill-phase
  miss counter would make the E5 conclusion continuously watchable; add one
  in the telemetry port (§10.3).
- Real-text steady-state rate (the filler-bias caveat, §6).
- ~~Is the copy serialized per-expert with a sync each, or batched?~~
  (Answered by E5/E13: the D2D gathers are kernels on the compute stream,
  the H2D misses are async DMAs on the copy stream — genuinely parallel,
  full duplex.)
- ~~The 2×3090 peer preset should show proportionally cheaper cold tails.~~
  (Answered by E12: measured NEGATIVE on c524 — see §7d; revisit only after
  fix §8.0.)
- The `stage_one` cold-cache pathology (E11's 61 s first read): why does the
  small-chunk path stage 174k experts for a 932-token read? Worth one look
  if that path ever matters again — it also streamed 13-15k experts/turn
  where its `order` implies ~4.8k.

## 12. Maintenance-interval work log (2026-10-04, evening)

- Engine-stderr `log` key added to all three desktop presets (§9).
- `STRATA_TRACE=1` enabled permanently in the c524 preset (per-request read
  telemetry: mode, ms, slot lend/refill, window counts).
- `STRATA_PREFILL_TIMING=1` enabled in the c524 preset (one phase block per
  prefill run — E5's instrument, kept on as ongoing telemetry).
- **#214 port LANDED** (fork commit 4916d7c): deferred tool-call rescue +
  natural-stop finish gate from upstream PR #525; 10 rescue tests + full
  serve suite 145/145; instance reloaded, patch live.
- E1/E3/E4/E5 run and recorded above; probe scripts preserved in /tmp
  (ab_probe.py, e1_probe.py, e3_probe.py, e4_probe.py, e5_probe.py — /tmp is
  wiped on reboot; the important one, ab_probe.py, is reproduced in §9).
- E8/E9/E10 (later same night, doc §7c): conversation-profile residency
  (no effect — the gather is unconditional), routed-only staging
  (`STRATA_PREFILL_STREAM_MIN=999999`, +5–16%), routing-transfer check
  (88.1%). All probe config reverted; preset back to defaults
  (`--expert-profile data/expert-profile.bin`, default stream threshold);
  e5/e7 probe scripts also copied to notes/.
- **A3 landed in the working tree** (stages A1/A2 committed as 9ba1c06 /
  21b63cf): the anticipation slice reserves the arena's last N slots
  (`alloc_top()`), the serve loop swaps the restored conversation's ranked
  routing into it on a dedicated stream hidden in the park/restore window,
  and lands it (sync + res_upload) before the K/V/loan touch host_res.
- **A3's missed-site landmine, caught by the parity gate run WITH the slice
  on** (`STRATA_PF_ANTICIPATE=512`): the serve path's `fits_one` lend check
  still compared `k + 128 > slots()` — the slice let chunk 32768 pass, the
  loan bottomed at slot 105 < the K/V's 128-slot floor, `kvg.on` went false,
  and the fallback tried to map the whole 524288-cell window → fatal "the K/V
  cannot hold 524288 cells". Baseline parity passed because anticipation was
  off (N=0 ⇒ alloc_top() == slots() everywhere). Fix: `fits_one`, the
  interactive path's belt-and-braces re-check, and three loan-size log
  arithmetic sites → `alloc_top()`. Lesson recorded: the parity gate must be
  run once with the feature's env override set, not only default-off.
- **The live A/B on c524 (the maintenance interval's measurement)**: 4 arms
  over two interleaved conversations (notes/ab_interleave_probe.py, the only
  shape that parks/restores per request). Baseline turns 1.92-1.98 s;
  anticipation 2048 → 2.14-2.19 s (no swap ever fired - the 6,602-slot
  profile covers ≥90% of repo-text conversations, so the thrash guard skips;
  the regression is the 2,048 profile slots it cost); resident-direct → wall
  unchanged (the ~370 ms/turn of gathers were overlapped with the ~550-800 ms
  H2D miss stream; phases move, gemms +8-17% from the indirect bases).
  Preset restored to defaults; both flags stay opt-in. The features are
  correct (bit-exact through every gate) - the expected win was wrong, not
  the code. bench/results/2026-10-04-expert-residency/README.md.
- **The strict parity check vs residency changes, classified** (control
  experiment): with the slice on, the gate's A/B/A continuation emits
  IDENTICAL tokens (initial request bit-identical too), but
  `restored main-model state differs` in exactly `gdn, tail, pooled,
  pooled_full, kv` — the recomputed fields.  Control: stock engine, NO
  anticipation, `--adapt-swaps 4` (the shipped adaptive tier) fails the
  same check with the same five fields and identical outputs.  So the
  check fails whenever mid-run residency changes (GPU-computed experts
  round differently — the startup banner's own caveat), which is why the
  tool pins `--adapt-swaps 0`.  The feature gate is the default-off run
  (byte-exact, PASS); the slice-on run is the smoke: swap fired
  ("388 pairs into 512 slots, 124 already resident, 95.0 ms, hidden in
  the park/restore"), outputs correct.
- **E13's telemetry + the two negative experiments (late evening)**: the
  streamed-experts/GiB line landed (d7e6a50; the threaded issuer's byte
  count was missing — its counters fold at chunk end, so the first cut froze
  once a run left the small-chunk path). The finding: ~16.8k experts / ~43
  GiB streamed per turn vs ~4.8k routed pairs — the walk re-streams the
  whole non-resident set every turn. E11 (STREAM_MIN=999999) and E12
  (3090 remote tier, --expert-cache-device1 6000) both measured negative
  and were reverted; the preset is back on defaults and serving. §7d has
  the numbers, §8.0 is the fix they point at.
- **Fix §8.0 landed same night (E14)**: STRATA_PF_ROUTED_STREAM (default
  off in code, ON in the c524 preset) - each layer's stream segment is
  filtered to its routing's picks at host-grouping time. Read walls
  -14 to -21% (1.98/1.92/1.96/1.92 s -> 1.71/1.51/1.61/1.55 s), wait copy
  548-804 -> 405-472 ms, streamed 43 -> ~34 GiB/turn. E15 (the 3090 tier
  stacked on it) still lost; the tier stays rejected on this box. The
  corrected physics - big chunks route nearly every expert, so the wall is
  now the honest miss-stream floor - is recorded in §7e.
- **Upstream ports #699 and #693 (late night)**: read-ahead at startup
  (madvise/fadvise WILLNEED walks over weights, dense, profile fill, RAM
  copy, MTP drafts; STRATA_READ_AHEAD=0 off) and the finer auto-chunk list
  + equal prompt chunks - #693's stated physics is this doc's §7d/§7e,
  arrived at independently upstream. Both parity-gated byte-exact, both
  live in production (fa005ec, 71dc3cb). The startup phases now log their
  durations; the c524 profile fill reads 0.9 s / 25.7 GB/s warm. The
  fill-ahead advise line does not print on the c524 preset (its source
  wiring takes a different srcp); the cold-start benefit shows after the
  next reboot.
