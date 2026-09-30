# The VMM / peer series, explained

Three commits on top of the new `main`, fixing the VMM range machinery that
`main` itself now ships. Upstream took elastic K/V (`9e3dbfe`) with the
`vmm.cpp` range allocator the fork built — and with it, two multi-GPU bugs and
one ordering bug that only show up when a VMM range belongs to a card other
than the first.

## 1. Chunks are created on the device that is current at `reserve()` (`ef75143`)

`vmm.cpp`'s `api()` initializes once and captures the current device id. Every
physical chunk (`cuMemCreate`) then used that captured id — so a range that
belongs to a second card had its memory allocated **on the first card**, and
the second card's mapping failed with a misleading `mapping X GiB of VRAM
failed: out of memory` while the second card sat nearly empty and the first
was full.

Measured on 2× RTX 3090 without P2P (`01:00.0` + `61:00.0`): any peer-tier
map of 18–22 GiB failed at any size. After the fix — chunks on the device
current at `reserve()`, and device-0 access to a peer range granted only when
`cudaDeviceCanAccessPeer` says a path exists (with an owner-only fallback,
because `cuMemSetAccess` rejects an unreachable remote location outright) —
the same pair maps 7000 slots (18.03 GiB, filled in 2.6 s) beside the
primary's 6056: 13056 of 24576 experts resident, decode at a measured 124.7
tok/s median against 85–89 tok/s for the same cards on `stratas` IQ3_S.

This is not peer-specific: **any** VMM range whose owning card is not the one
that happened to be current at first `api()` use lands its memory on the
wrong device. Elastic K/V on the primary happens to be safe today only
because device 0 is current first.

## 2. Only the cache the K/V borrows from maps through VMM (`cf43c30`)

`set_vmm` was static: enabling it turned *every* expert cache's arena into a
VMM range — including a peer tier's, which has no elastic K/V of its own and
gains nothing from VMM, while paying for it in chunk-map machinery (and, on
a second card, in bug 1). Now per-instance: the primary enables it, a peer
tier's cache is one plain `cudaMalloc` again.

## 3. The peer device's context exists before the arena's whole-range pin (`55284ba`)

`cuMemMap`/`cuMemSetAccess` on a peer range need the peer device's CUDA
context to already exist. The peer context used to be created after the
arena's whole-range pin; on some driver/device combinations the pin itself
needs the peer context, and the map faults or fails. Ordering: context
first, pin second.

## Why this matters on `main` now

Elastic K/V is `main`'s feature, and its VMM range is the same code. A
single-GPU user never notices; a multi-GPU user who maps anything on a
second card through it gets the empty-card "out of memory". The series is
measured on real hardware, is a net +56/−23 lines, and changes nothing on
the single-GPU path (chunk creation for a range on the current device is
byte-identical to before).
