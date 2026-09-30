# Upstream PR draft — for review, nothing posted

- **target:** `Niko1221/Strata`, base `main`, head `chimpera/vmm-peer-series`
  (branch `offer-peer-series` locally: upstream `82f46a8` + 4 commits — the
  gate change + three VMM fixes; `vmm_test` and the cache suites green)
- **title:** `generate: allow the elastic K/V with --peer-device (VMM ranges keep their memory on the owning GPU)`

---

## PR body

`--kv-grow` turns itself off whenever a second GPU is present — the gate reads
`o.peer_device < 0` — so adding a peer tier costs the primary its elastic K/V.
Nothing in the machinery requires the exclusion; what it protected against is
fixed here directly, and the two now run together.

**Why the gate was there.** `set_vmm` was static: enabling it mapped *every*
expert cache's arena through a `vmm.cpp` range — including the peer tier's.
And a range's physical chunks were created with the device id captured at the
once-only `api()` init (the primary's device), not the device that owns the
range, so a second card's mapping failed while that card sat nearly empty.
Reproduced on this PR's base with only the gate line removed: with `--kv-grow
--peer-device 1`, startup ends in

    strata generate: peer experts: ExpertCache: mapping 18.05 GiB of VRAM failed: out of memory

on a card with ~20 GiB free — the peer's chunks were being allocated on the
primary. (First seen on the strata-nvfp4 fork's 2× RTX 3090 pair; the same
`vmm.cpp` ships here with the elastic K/V.)

**The three fixes:**

1. `vmm_chunk_new` takes the device; a range creates its chunks on the device
   current at `reserve()`, not the one captured at init. Device 0 is granted
   access to a peer range only when `cudaDeviceCanAccessPeer` says a path
   exists, with an owner-only fallback (`cuMemSetAccess` rejects an unreachable
   remote location outright, failing the whole map).
2. `set_vmm` is per-instance: only the cache the K/V borrows from maps through
   VMM; a peer tier's cache is one plain `cudaMalloc`. This is what makes the
   gate unnecessary.
3. The peer device's context is created before the arena's whole-range pin
   (`cuMemMap`/`cuMemSetAccess` on a peer range need it to already exist).

**Measured on mainline** (2× RTX 3090, `01:00.0` + `61:00.0`, no P2P, IQ3_S,
262K context, 16,399-token prompt, both cards capped at 19 GiB; the second
card also carries desktop processes):

| | resident experts | elastic K/V | prefill 16k | decode 600 tok at 16k |
| --- | --- | --- | ---: | ---: |
| base + gate relaxed | — | — | fails to start (the line above) | — |
| this PR, one card | 4761 of 24576 | off | 2599 tok/s (n=6) | 56.4 tok/s median (n=6) |
| this PR + peer | 15903 of 24576 | on | 2611 tok/s (n=5 warm) | 65.0 tok/s median (n=6) |

Peer attach: 9488 experts, 18.06 GiB, filled in 0.8 s. Prefill is parity — at
this shape the prompt path is bound on the primary either way. Decode medians
15% apart with overlapping spreads (pair 41.7–93.0, solo 30.8–71.7): the
second card's 3.3× residency removes per-token misses, but the spread is
dominated by the MTP draft-acceptance swing of greedy prose and this
machine's other tenants, so the honest claim is at-or-above, not a speedup.
The elastic K/V is the part the gate was blocking, working under load,
quoted from the run log:

    strata: K/V grown to 24576 cells (0.36 GiB); the expert cache gave 56 slots for it, 21 hotter experts moved to colder ones' slots (6359 of 6415 slots hold experts)

The single-GPU path is byte-identical: a range on the current device creates
its chunks exactly as before, and `vmm_test` and the conversation-cache suites
pass unchanged. Layer splits, `--batch` and `--vram-elastic` still turn the
elastic K/V off, as before. Layer-split smoke, same two cards through the
serve (`--gpu 0,2 --layer-split auto`): this PR and its base pick the same
K=30 split (layers 0–29 / 30–47, session [30, 48)), print no elastic-K/V and
no peer lines, and answer normally — the only startup differences are the
free-VRAM readings (the second card carries desktop processes), which size
its cache 3733 vs 3764 slots run to run.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
