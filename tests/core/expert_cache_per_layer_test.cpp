// #369: per-layer admission on a sized cache (open_sized, every native pack) - each layer admits into its own slot
// range, as on the uniform cache, so no two layers write the same slot.  Needs a CUDA device (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    std::string err;
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache cache;
        const bool ok = sized ? cache.open_sized(std::vector<int64_t>(6, 1024), 3, 4, err)
                              : cache.open(6, 3, 4, 1024, err);
        if (!ok) {
            std::fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        cache.set_per_layer_admission(true);
        for (int64_t layer = 0; layer < 3; ++layer) {
            int64_t lo = 0, hi = 0;
            cache.layer_slot_range(layer, lo, hi);
            for (int64_t e = 0; e < 3; ++e) {
                const int32_t slot = cache.admit(layer, e);
                const bool want_room = lo + e < hi;
                if (want_room ? slot != (int32_t) (lo + e) : slot != strata::core::kNotResident) {
                    std::fprintf(stderr, "%s cache: layer %lld expert %lld got slot %d (range %lld..%lld)\n",
                                 sized ? "sized" : "uniform", (long long) layer, (long long) e, slot, (long long) lo,
                                 (long long) hi - 1);
                    return 1;
                }
            }
        }
    }
    // The working-set feature: the LAST `reserved` slots are the anticipation slice, and `alloc_top()` is the
    // exclusive ceiling EVERY slot consumer must compute from - admission (both forms), the per-layer ranges,
    // the prefill loan (which counts down from the top) and the elastic K/V's bare `kvg_start` sites.  The
    // check below is the regression shape for the last of those: a scan seeded at `slots()` instead of
    // `alloc_top()` would walk straight into the slice, and only the arena's contents would differ.
    for (int sized = 0; sized < 2; ++sized) {
        strata::core::ExpertCache cache;
        const bool ok = sized ? cache.open_sized(std::vector<int64_t>(8, 1024), 3, 4, err)
                              : cache.open(8, 3, 4, 1024, err);
        if (!ok) {
            std::fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        cache.set_reserved(2);
        if (cache.reserved() != 2 || cache.alloc_top() != 6 || cache.slots() != 8) {
            std::fprintf(stderr, "%s cache: reserved %lld, alloc_top %lld, slots %lld - want 2/6/8\n",
                         sized ? "sized" : "uniform", (long long) cache.reserved(),
                         (long long) cache.alloc_top(), (long long) cache.slots());
            return 1;
        }
        // the per-layer ranges divide alloc_top(), not slots(): 8 slots / 3 layers hands the last layer
        // [4,8), whose upper half is the slice; 6/3 hands it [4,6) and the slice belongs to nobody.
        int64_t lo = 0, hi = 0;
        cache.layer_slot_range(2, lo, hi);
        if (lo != 4 || hi != 6) {
            std::fprintf(stderr, "%s cache: layer 2 range %lld..%lld, want 4..6 (alloc_top, not slots)\n",
                         sized ? "sized" : "uniform", (long long) lo, (long long) hi);
            return 1;
        }
        // global admission stops at the slice: 3x4 distinct pairs, only 6 admissible slots
        for (int64_t l = 0; l < 3; ++l)
            for (int64_t e = 0; e < 4; ++e) {
                const int32_t slot = cache.admit(l, e);
                const int64_t nth = l * 4 + e;
                if (nth < 6 ? slot != (int32_t) nth : slot != strata::core::kNotResident) {
                    std::fprintf(stderr, "%s cache: admit(%lld,%lld) = %d, want %lld\n", sized ? "sized" : "uniform",
                                 (long long) l, (long long) e, slot, nth < 6 ? nth : (int64_t) strata::core::kNotResident);
                    return 1;
                }
            }
        // the slice itself is still fillable storage - the serve loop's wholesale swap writes exactly there
        uint8_t blob[1024] = {0};
        blob[0] = 7;
        if (!cache.fill_slot_blocking(6, blob, err) || !cache.verify_slot(6, blob, err)) {
            std::fprintf(stderr, "%s cache: slice slot 6 not fillable/verifiable: %s\n",
                         sized ? "sized" : "uniform", err.c_str());
            return 1;
        }
    }
    // an over-large reservation clamps to the arena rather than making alloc_top() negative
    {
        strata::core::ExpertCache cache;
        if (!cache.open(4, 2, 2, 1024, err)) {
            std::fprintf(stderr, "open: %s\n", err.c_str());
            return 1;
        }
        cache.set_reserved(9);
        if (cache.alloc_top() != 0 || cache.reserved() != 9 || cache.admit(0, 0) != strata::core::kNotResident) {
            std::fprintf(stderr, "clamped reservation: alloc_top %lld, admit %d - want 0 / not resident\n",
                         (long long) cache.alloc_top(), cache.admit(0, 0));
            return 1;
        }
    }
    std::puts("expert cache per-layer admission: uniform and sized caches passed");
    return 0;
}
