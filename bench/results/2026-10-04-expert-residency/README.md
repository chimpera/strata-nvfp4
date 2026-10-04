# Expert residency: the anticipation slice + direct-from-slot MMQ (2026-10-04)

Two opt-in features, built and gated correct, measured on the workload they were
designed for — and the measurement says neither moves the wall on THIS preset.
Recorded because a negative result with the mechanism understood beats an
unmeasured expectation.

## The workload

`notes/ab_interleave_probe.py` on the c524 preset (RTX 5090, 8,650-slot expert
cache, 6,602 slots filled from the shipped profile): two conversations
alternating request-by-request (each switch parks one and restores the other —
the only shape where the anticipation slice can act), six turns appending
~1.5-4.3k tokens of real repo text each.

## The arms (per-turn read wall, ms)

| arm | A-turn1 (4.3k fresh) | B-turn1 (2.2k) | A-turn2 (2.8k) | B-turn2 (2.2k) |
|-----|-----|-----|-----|-----|
| baseline (both off) | 1,983 | 1,924 | 1,956 | 1,922 |
| `--expert-anticipation 2048` | 2,194 | 2,139 | 2,165 | 2,135 |
| `STRATA_PF_RESIDENT_DIRECT=1` | 1,985 | 1,929 | 1,954 | 1,920 |

Phase blocks of A-turn1, baseline → direct (the other turns agree):

- wait copy 548 ms (30.7%) → 384 ms (21.5%) — the H2D expert streams
- dequant 370 ms (20.8%) → 520 ms (29.1%) — the D2D gathers (baseline) vs the
  streamed experts' exposed gathers (direct)
- gemm gate/up 193 → 208 ms, down 118 → 138 ms (+8-17%: an indirect base
  pointer per expert, no cross-expert L2 locality)

## What the numbers say

**The D2D gather was overlapped, not exposed.** Baseline's ~370 ms of per-turn
gathers (≈11 GB device-to-device, every routed expert copied into a contiguous
group buffer — E5's "unconditional gather") ran concurrently with the ~550-800
ms of H2D streaming of the MISSED experts: full-duplex PCIe hides one behind
the other. Removing the gathers entirely (direct) moves the phases around and
adds ~35 ms of kernel time — the wall stays. The deep-turn wall on this preset
is bounded by the miss stream itself (host staging 96-297 ms visible in the
same blocks), which is what E9's routed-only staging already measured at
+5-16%.

**The anticipation slice never fired — correctly.** The thrash guard skips a
swap when ≥90% of the incoming conversation's experts are already resident,
and the 6,602-slot profile covers that much of repo-text conversations (E7
measured 87-94% coverage by the conversation's own prior routing; the global
profile is wider). Zero swap lines across the whole arm; the only effect was
2,048 fewer profile slots, hence the ~200 ms regression. The feature pays
where the profile does NOT cover the conversation: small caches, or a
conversation whose domain is idiosyncratic against the corpus the profile was
built from.

## Where each feature should pay

- `STRATA_PF_RESIDENT_DIRECT=1` (pointer-table MMQ, NVFP4 W4A8, single GPU):
  removes ~370 ms/turn of D2D traffic (2k-fresh turns on this preset). Worth
  it when wait-copy is small (few misses) or the copy engine is contended —
  multiple engines sharing a card, or K/V restore traffic in the window — at
  a ~35 ms/turn kernel cost. Bit-exact: the pointer path computed
  byte-identical main-model state through the conversation-cache parity gate.
- `--expert-anticipation N`: fills the last N slots with the restored
  conversation's parked routing at each restore (measured 95 ms for 388
  experts, hidden in the park/restore window on the parity gate's 3090).
  Pays only when the conversation's experts are mostly profile MISSES.

## Correctness (the part that must never regress)

- `prefill_mmq_direct_test`: 8 product shapes bit-equal (mixed/empty/one-row
  groups, fallback tiles, 16-expert groups, the full
  gate/up→swiglu→q8→down→scales chain with alternating residency);
  `compute-sanitizer --tool memcheck` 0 errors (the vendored patch also fixed
  a stock ids overread the sanitizer caught — see
  `third_party/llama-mmq-ptr-table.patch`).
- Conversation-cache parity gate (`tools/conversation_cache_parity.py`):
  PASS byte-exact default-off AND with each feature's env override set.
