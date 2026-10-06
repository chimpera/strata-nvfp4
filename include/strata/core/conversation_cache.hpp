// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // Ordinary layer-split checkpoints retain each device's running state.
    // Whole-session parking is currently single-GPU and rejects these parts.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
    std::vector<ConversationKvReuse> stages;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
        for (const auto& layer : kv) n += layer.bytes();
        for (const auto& s : stages) n += s.bytes();
        return n;
    }
};

// Observed return behavior of one conversation lineage, carried by the parked image
// through park -> restore -> re-park (drop_superseded drops only the stale copy; the
// re-parked image keeps the accumulated record).  Bursts counts RESTORES, so a
// conversation served once and never asked again - a permission classifier, a title
// generator, any one-shot - stays at 0 forever.  Fixed-size, so bytes() ignores it.
struct ConversationTraffic {
    uint32_t bursts = 0;                  // park -> restore cycles of this lineage
    uint64_t first_ms = 0, last_ms = 0;   // first park, last park-or-restore (steady clock)
    std::array<uint64_t, 8> gaps{};       // the most recent park -> restore gaps
    uint32_t gaps_n = 0, gaps_w = 0;   // recorded count (capped) and ring write cursor

    void note_gap(uint64_t now_ms) {
        if (last_ms != 0 && now_ms > last_ms) {
            gaps[gaps_w] = now_ms - last_ms;
            gaps_w = (gaps_w + 1) % gaps.size();
            if (gaps_n < gaps.size()) ++gaps_n;        // capped, so gap_median_ms's copy is in bounds
        }
    }
    // Median of the recorded return gaps: the lineage's own expected pace.  A median, not
    // a mean, so one long tool-call silence inside a session does not stretch the budget.
    uint64_t gap_median_ms() const {
        if (gaps_n == 0) return 0;
        std::vector<uint64_t> sorted(gaps.begin(), gaps.begin() + (ptrdiff_t) gaps_n);   // cold path: eviction scan only
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        return (n & 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }
};

/// Layer-split parking: each running checkpoint holds the first stage's state and, in `stage_parts`, one part per
/// later stage; a parked conversation keeps them in one image per stage.  `split` MOVES the checkpoints apart (no
/// running state is copied): the complete ones (a part for each of `stages` stages) into `stage0` and `parts[k]`,
/// every part carrying the checkpoint's ids / images / LRU stamp, the others aside; `merge` puts them back in their
/// original order.  merge(split(x)) == x.
struct ConversationCheckpointSplit {
    std::vector<ConversationCheckpoint> stage0;
    std::vector<std::vector<ConversationCheckpoint>> parts;   ///< [stage][checkpoint]
    std::vector<ConversationCheckpoint> rest;                 ///< not split (no complete set of stage parts)
    std::vector<bool> complete;                               ///< per original position: in stage0/parts or in rest
};

inline ConversationCheckpointSplit conversation_checkpoints_split(std::vector<ConversationCheckpoint>&& checks,
                                                                  size_t stages) {
    ConversationCheckpointSplit out;
    out.parts.resize(stages);
    out.complete.reserve(checks.size());
    size_t whole = 0;
    for (const auto& c : checks) {
        const bool complete = c.stage_parts.size() == stages;
        out.complete.push_back(complete);
        whole += complete ? 1 : 0;
    }
    out.stage0.reserve(whole);
    out.rest.reserve(checks.size() - whole);
    for (auto& parts : out.parts) parts.reserve(whole);
    // Allocate and copy identity metadata before moving any running state.
    // If allocation fails, every input checkpoint remains usable by the caller.
    for (const auto& c : checks) {
        if (c.stage_parts.size() != stages) continue;
        for (size_t k = 0; k < stages; ++k) {
            ConversationCheckpoint part;
            part.ids = c.ids; part.imgs = c.imgs; part.used = c.used;
            out.parts[k].push_back(std::move(part));
        }
    }
    size_t w = 0;
    for (auto& c : checks) {
        if (c.stage_parts.size() != stages) { out.rest.push_back(std::move(c)); continue; }
        for (size_t k = 0; k < stages; ++k) {
            auto& dst = out.parts[k][w];
            auto& src = c.stage_parts[k];
            dst.gdn = std::move(src.gdn); dst.ple = std::move(src.ple);
            dst.tails = std::move(src.tails); dst.dead = std::move(src.dead);
            dst.block_pos = std::move(src.block_pos);
        }
        c.stage_parts.clear();
        out.stage0.push_back(std::move(c));
        ++w;
    }
    checks.clear();
    return out;
}

/// false (and `out` empty) when the parts do not line up: a stage with another number of checkpoints, or a part
/// whose ids differ from its checkpoint's.
inline bool conversation_checkpoints_merge(ConversationCheckpointSplit&& split, std::vector<ConversationCheckpoint>& out) {
    out.clear();
    for (const auto& p : split.parts)
        if (p.size() != split.stage0.size()) return false;
    size_t whole = 0, other = 0;
    for (bool c : split.complete) (c ? whole : other) += 1;
    if (!split.complete.empty() && (whole != split.stage0.size() || other != split.rest.size())) return false;
    if (split.complete.empty()) split.complete.assign(split.stage0.size(), true);   // a restored image: all whole
    for (size_t j = 0; j < split.stage0.size(); ++j)
        for (const auto& p : split.parts)
            if (p[j].ids != split.stage0[j].ids) return false;
    size_t w = 0, r = 0;
    for (bool c : split.complete) {
        if (!c) { out.push_back(std::move(split.rest[r++])); continue; }
        ConversationCheckpoint cp = std::move(split.stage0[w]);
        for (auto& p : split.parts) cp.stage_parts.push_back(std::move(p[w]));
        ++w;
        out.push_back(std::move(cp));
    }
    return true;
}

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    bool cvec = true;
    // with a layer split, the later stages' own images, one per stage, in stage order
    std::vector<SavedConversation> stage_images;
    // House patch (T1 conversation identity): the client-supplied
    // x-conversation-id this conversation was served under, and the
    // ephemeral flag (ephemeral conversations are never parked at all, so
    // an entry carrying it can only come from an older engine). Both are
    // exact-drop metadata: matching for RESTORE still runs on token
    // prefixes, which is what reuse correctness needs.
    std::string conv_id;
    bool ephemeral = false;
    ConversationTraffic traffic;

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv) + conv_id.capacity();
        for (const auto& s : stage_images) n += s.bytes();
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

// Parking-victim policy.  `lru` evicts the least recently active entry (a restore or a
// re-park refreshes position): today's behavior, the default.  `expectancy` promotes two
// statistically-dead classes ahead of it - conversations that never came back (a one-shot:
// classifier, title, any single request; zero restores and quiet past a grace) and lineages
// silent far past their own observed return pace - then falls back to LRU order.  Built for
// agent-harness traffic, where one long-lived conversation interleaves with bursts of tiny
// one-shots: under LRU a freshly parked one-shot is the MOST protected entry at exactly the
// moment it becomes dead forever.  Statistical by construction: whether a silent conversation
// ended or is merely idle is unknowable engine-side, so nothing is dropped eagerly - the
// tiers only reorder victims under pressure.
enum class EvictionPolicy { lru, expectancy };

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    ConversationCache(size_t budget, size_t slots, EvictionPolicy policy = EvictionPolicy::lru)
        : budget_(budget), slots_(slots), policy_(policy) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    size_t evictions_dead() const { return evictions_dead_; }
    size_t evictions_stale() const { return evictions_stale_; }
    // Tests drive the tiers with a fake clock; the engine uses the steady clock.
    void set_clock_for_testing(uint64_t (*fn)()) { now_ms_ = fn; }
    size_t dropped() const { return dropped_; }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        for (auto& s : reuse_.stages) s.unchanged_tokens = std::min(s.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    // Purge hint: the caller certifies this prefix is dead, so its parked
    // images leave now instead of waiting for LRU. Deliberately looser than
    // best(): either side may prefix the other, and EQUAL length matches -
    // the probe replays a finished request, whose parked entry is the same
    // length, which conversation_prefix() would (correctly, for reuse) reject.
    // Text probes carry empty images, so multimodal entries never match one.
    template<class Token>
    size_t drop(const std::vector<Token>& probe, const std::vector<ConversationImageKey>& images, bool cvec) {
        size_t removed = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->cvec != cvec) { ++it; continue; }
            auto hit = [&](const ConversationCheckpoint& c) {
                const size_t m = std::min(c.ids.size(), probe.size());
                if (m == 0 || !std::equal(c.ids.begin(), c.ids.begin() + (ptrdiff_t) m, probe.begin())) return false;
                // the overlapping window must agree on image identity too: same
                // image keys positioned before m on both sides (empty for text)
                auto within = [&](std::vector<ConversationImageKey> k) {
                    k.erase(std::remove_if(k.begin(), k.end(),
                          [&](const ConversationImageKey& x) { return x.start >= (int64_t) m; }), k.end());
                    return k;
                };
                return within(c.imgs) == within(images);
            };
            bool match = hit(it->live);
            for (const auto& c : it->checkpoints) match = match || hit(c);
            if (match) {
                bytes_ -= it->bytes();
                ++dropped_;
                ++removed;
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        return removed;
    }

    // House patch (T1): drop every entry parked under this exact
    // conversation id (x-conversation-id). No tokenization needed on
    // either side; idempotent on a miss. cvec-agnostic: an id names a
    // conversation, not a steering mode.
    size_t drop_id(const std::string& conv_id) {
        if (conv_id.empty()) return 0;
        size_t removed = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->conv_id == conv_id) {
                bytes_ -= it->bytes();
                ++dropped_;
                ++removed;
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        return removed;
    }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        // A restore is the lineage returning: count the burst and record how long it
        // stayed away (the park -> restore gap the expectancy policy learns from).
        out.traffic.note_gap(now_ms_());
        out.traffic.bursts++;
        out.traffic.last_ms = now_ms_();
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            size_t victim = 0;
            int tier = 0;                               // 0 = LRU front, 1 = dead one-shot, 2 = gap-stale
            if (policy_ == EvictionPolicy::expectancy) std::tie(victim, tier) = expectant_victim();
            if (tier == 1) ++evictions_dead_;
            else if (tier == 2) ++evictions_stale_;
            bytes_ -= entries_[victim].bytes();
            entries_.erase(entries_.begin() + (std::ptrdiff_t) victim);
            ++evictions_;
        }
        return true;
    }

    // #342: drop the parked entries an outgoing conversation (its live tokens and checkpoint chain) supersedes:
    // the same conversation a turn back, whose DEEPEST checkpoint the outgoing chain still holds, so all it adds
    // is the tail the client rewrote (the reply as it was generated, before the next request re-rendered it) and
    // checkpoints older than that one.  A subagent's successive turns parked one such copy each, and make_room's
    // oldest-first eviction then pushed the parent conversation out after `slots` turns.  An entry without
    // checkpoints, or whose deepest checkpoint the outgoing chain does not hold (another conversation that only
    // shares the system prompt's root with it), is kept.  Returns how many were dropped.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        auto held = [&](const ConversationCheckpoint& c) {
            if (c.ids == ids && c.imgs == images) return true;
            for (const auto& k : checkpoints)
                if (k.ids == c.ids && k.imgs == c.imgs) return true;
            return false;
        };
        size_t dropped = 0;
        for (size_t i = 0; i < entries_.size();) {
            const auto& e = entries_[i];
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
                bytes_ -= e.bytes();
                entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
                ++dropped;
                continue;
            }
            ++i;
        }
        superseded_ += dropped;
        return dropped;
    }
    size_t superseded() const { return superseded_; }

    bool put(SavedConversation&& image, size_t held = 0) {
        const size_t n = image.bytes();
        if (!enabled() || held > budget_ || n > budget_ - held) return false;   // make_room's refusal, first
        image.traffic.last_ms = now_ms_();            // a park is activity; first_ms survives restores
        if (image.traffic.first_ms == 0) image.traffic.first_ms = image.traffic.last_ms;
        drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
        if (!make_room(n, held)) return false;
        entries_.push_back(std::move(image));
        bytes_ += n;
        return true;
    }

private:
    // A conversation that returned zero times and has been quiet this long is dead with
    // near-certainty (agent-harness one-shots never return at all), and a lineage silent
    // past its own observed pace (k * its median park->restore gap, floored) has almost
    // certainly ended.  Both are statistical: nothing is dropped eagerly, victims are only
    // reordered under pressure, and tier 2 never fires without a recorded return gap.
    static constexpr uint64_t DEAD_GRACE_MS = 60'000;
    static constexpr uint64_t STALE_K = 8;
    static constexpr uint64_t STALE_FLOOR_MS = 300'000;

    // The expectancy victim and its tier: entries_ is LRU order, so each tier scan takes the
    // oldest qualifying entry first.  An entry without traffic stamps (parked by an older
    // engine) is never tier-1/2; it falls through to the LRU front like today.
    std::pair<size_t, int> expectant_victim() const {
        const uint64_t now = now_ms_();
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& tr = entries_[i].traffic;
            if (tr.bursts == 0 && tr.last_ms != 0 && now - tr.last_ms > DEAD_GRACE_MS) return {i, 1};
        }
        for (size_t i = 0; i < entries_.size(); ++i) {
            const auto& tr = entries_[i].traffic;
            const uint64_t stale = std::max<uint64_t>(STALE_K * tr.gap_median_ms(), STALE_FLOOR_MS);
            if (tr.bursts > 0 && tr.last_ms != 0 && now - tr.last_ms > stale) return {i, 2};
        }
        return {0, 0};
    }
    static uint64_t steady_now_ms() {
        return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0, dropped_ = 0;
    size_t evictions_dead_ = 0, evictions_stale_ = 0;
    EvictionPolicy policy_ = EvictionPolicy::lru;
    uint64_t (*now_ms_)() = &steady_now_ms;
    std::deque<SavedConversation> entries_; // least recently active first
    ConversationKvReuse reuse_;
};

} // namespace strata::core
