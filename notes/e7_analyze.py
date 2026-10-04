#!/usr/bin/env python3
"""E7 analysis: parse the routing trace, measure anticipation coverage.

Metrics per turn n (pair = (layer, expert)):
  prev   : coverage of turn n's pairs by turn n-1's pairs      (short memory)
  conv   : coverage by the union of turns 1..n-1               (prefix-coupled tracker)
  global : coverage by top-K pairs by frequency over turns 1..n-1, K = |conv set|
           (a same-budget global profile ~ what the adaptive tier approximates)
Also: distinct experts per layer per turn (real-text selectivity).
"""
import json, struct
from collections import Counter

data = open("/tmp/e7-routing.bin", "rb").read()
offs = json.load(open("/tmp/e7-offsets.json"))

def parse(seg):
    pairs, per_layer = set(), {}
    i = 0
    while i + 8 <= len(seg):
        layer, k = struct.unpack_from("<ii", seg, i)
        i += 8
        if k <= 0 or k > 64 or i + k * 8 > len(seg):
            break
        ids = struct.unpack_from(f"<{k}i", seg, i)
        i += k * 8   # skip ids + weights
        s = per_layer.setdefault(layer, set())
        for e in ids:
            s.add(e)
            pairs.add((layer, e))
    return pairs, per_layer

segs = [parse(data[offs[j]:offs[j + 1]]) for j in range(len(offs) - 1)]
names = ["base", "t1-doc", "t2-frontend", "t3-pool", "t4-prefill", "t5-agents", "t6-server"]

print(f"{'turn':<12} {'pairs':>6} {'experts/layer (avg)':>20}")
for nm, (p, pl) in zip(names, segs):
    avg = sum(len(s) for s in pl.values()) / max(len(pl), 1)
    print(f"{nm:<12} {len(p):>6} {avg:>20.1f}")

print()
print(f"{'turn':<12} {'prev':>7} {'conv':>7} {'global(same budget)':>20}")
cov_prev, cov_conv, cov_glob = [], [], []
for n in range(2, len(segs)):
    cur = segs[n][0]
    prev = segs[n - 1][0]
    union = set().union(*[segs[j][0] for j in range(n)])
    cnt = Counter()
    for j in range(n):
        cnt.update(segs[j][0])
    glob = set(dict(cnt.most_common(len(union))))
    cp, cc, cg = len(cur & prev) / len(cur), len(cur & union) / len(cur), len(cur & glob) / len(cur)
    cov_prev.append(cp); cov_conv.append(cc); cov_glob.append(cg)
    print(f"{names[n]:<12} {cp:>7.1%} {cc:>7.1%} {cg:>20.1%}")

m = lambda xs: sum(xs) / len(xs)
print(f"\nmean over turns:  prev={m(cov_prev):.1%}  conv(prefix-coupled)={m(cov_conv):.1%}  global={m(cov_glob):.1%}")
