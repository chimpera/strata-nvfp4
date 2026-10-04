// prefill_mmq_direct_test - the pointer-table MMQ (src/prefill/mmq_direct.cu, STRATA_MMQ_PTR_TABLE) against the
// gathered-image path of the same int8 kernels, BIT FOR BIT: the same weight bytes, the same q8_1 activations,
// the same bounds/ids - one product reads experts from a contiguous gathered buffer (today's mul_mat_q), the
// other from one base pointer per expert at unrelated 256-byte-aligned addresses with gaps between them (the
// expert cache's slots).  Any difference is the pointer-table kernel addressing weights differently, which is
// exactly the bug this test exists to catch; the numbers themselves are meaningless (random bytes through the
// NVFP4 decoders, masked to 0x3F so every e4m3/fp16 block scale stays finite).
// Cases: a mixed group with an empty expert and a one-row expert, a single-expert product, weight rows that are
// not a multiple of 128 (the fallback tile config), and a full 16-expert group.
// Exit 77 without a CUDA device, or when STRATA_PREFILL_NVFP4 is not w4a8 (the direct path is W4A8-only in v1).
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

int checks = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t round256(uint64_t n) { return (n + 255) / 256 * 256; }

/// One product shape through both paths.  `rows[e]`: expert e's activation rows; the product is
/// dst[row, 0..w_rows) = W_e * x[row], weights a fresh random image per expert.
void run_case(const char* name, int64_t w_rows, int64_t w_cols, const std::vector<int64_t>& rows, std::mt19937& rng) {
    const int n = (int) rows.size();
    int64_t total = 0, max_rows = 0;
    std::vector<int32_t> bounds = {0};
    for (int64_t r : rows) {
        total += r;
        max_rows = max_rows > r ? max_rows : r;
        bounds.push_back((int32_t) total);
    }
    const size_t image = mmq::matrix_bytes(GGML_TYPE_NVFP4, w_rows, w_cols);
    const uint64_t slot = round256(image) + 512;   // a gap after every expert: no contiguity to lean on

    // the weights, once on the host: the gathered image gets them back to back, the "slots" with gaps
    std::vector<uint8_t> host((size_t) n * image);
    for (auto& b : host) b = (uint8_t) (rng() & 0x3F);
    void *gather = nullptr, *arena = nullptr;
    ck(cudaMalloc(&gather, (size_t) n * image), "cudaMalloc gather");
    ck(cudaMalloc(&arena, (size_t) n * slot), "cudaMalloc arena");
    ck(cudaMemcpy(gather, host.data(), host.size(), cudaMemcpyHostToDevice), "fill gather");
    std::vector<const void*> ptrs;
    for (int e = 0; e < n; ++e) ptrs.push_back((const uint8_t*) arena + (size_t) e * slot);
    for (int e = 0; e < n; ++e)
        ck(cudaMemcpy((void*) ptrs[e], host.data() + (size_t) e * image, image, cudaMemcpyHostToDevice), "fill slot");
    void* d_ptrs = nullptr;
    ck(cudaMalloc(&d_ptrs, ptrs.size() * sizeof(void*)), "cudaMalloc pointer table");
    ck(cudaMemcpy(d_ptrs, ptrs.data(), ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice), "fill pointer table");

    // the activations and their q8_1 image, shared by both products (only the weights' addressing differs)
    std::vector<float> hx((size_t) total * w_cols);
    for (auto& v : hx) v = (float) ((int64_t) rng() % 2001 - 1000) / 1000.0f;
    void *x = nullptr, *xq = nullptr;
    const size_t xq_bytes = mmq::q8_bytes(total, w_cols);
    ck(cudaMalloc(&x, hx.size() * sizeof(float)), "cudaMalloc x");
    ck(cudaMalloc(&xq, xq_bytes), "cudaMalloc xq");
    ck(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice), "fill x");
    std::vector<int32_t> ids(total);
    for (int64_t i = 0; i < total; ++i) ids[i] = (int32_t) i;
    for (int64_t i = total - 1; i > 0; --i) std::swap(ids[i], ids[rng() % (uint32_t) (i + 1)]);   // a fixed shuffle
    void *d_ids = nullptr, *d_bounds = nullptr;
    ck(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)), "cudaMalloc ids");
    ck(cudaMalloc(&d_bounds, bounds.size() * sizeof(int32_t)), "cudaMalloc bounds");
    ck(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "fill ids");
    ck(cudaMemcpy(d_bounds, bounds.data(), bounds.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "fill bounds");
    mmq::quantize((const float*) x, (const int32_t*) d_ids, xq, GGML_TYPE_NVFP4, w_cols, w_cols, total, nullptr);

    float *dst[2] = {nullptr, nullptr};
    for (float*& d : dst) {
        ck(cudaMalloc(&d, (size_t) total * w_rows * sizeof(float)), "cudaMalloc dst");
        ck(cudaMemset(d, 0, (size_t) total * w_rows * sizeof(float)), "clear dst");
    }
    mmq::Context ctx;
    const mmq::Product base = {.type = GGML_TYPE_NVFP4,
                               .w_rows = w_rows,
                               .w_cols = w_cols,
                               .expert_bytes = image,
                               .n = n,
                               .xq = xq,
                               .bounds = (const int32_t*) d_bounds,
                               .ids = (const int32_t*) d_ids,
                               .total_rows = total,
                               .max_rows = max_rows,
                               .dst = nullptr,
                               .ld_dst = w_rows};
    mmq::Product p = base;
    p.w = gather;   // today's path: one image, experts `image` bytes apart
    p.dst = dst[0];
    ctx.run(p, nullptr);
    p = base;
    p.w = nullptr;   // the pointer table: experts at their own (gapped, 256-aligned) addresses
    p.w_ptrs = (const void* const*) d_ptrs;
    p.dst = dst[1];
    ctx.run(p, nullptr);
    ck(cudaDeviceSynchronize(), "run");

    std::vector<float> out[2];
    for (int i = 0; i < 2; ++i) {
        out[i].resize((size_t) total * w_rows);
        ck(cudaMemcpy(out[i].data(), dst[i], out[i].size() * sizeof(float), cudaMemcpyDeviceToHost), "read dst");
    }
    size_t first = SIZE_MAX;
    for (size_t i = 0; i < out[0].size(); ++i)
        if (std::memcmp(&out[0][i], &out[1][i], sizeof(float)) != 0) { first = i; break; }
    ++checks;
    if (first != SIZE_MAX)
        std::fprintf(stderr, "FAIL: %s: dst[%zu] (row %lld, col %lld) = %f gathered vs %f pointer table\n", name,
                     first, (long long) (first / w_rows), (long long) (first % w_rows), out[0][first], out[1][first]);
    else if (total > 0 && w_rows > 0 && out[0][0] == 0.0f && out[0].back() == 0.0f)
        std::fprintf(stderr, "FAIL: %s: both outputs are zero - the comparison proved nothing\n", name);
    else
        std::printf("PASS: %s: %d experts, %lld rows of %lld, %lld weights each - bit-equal\n", name, n,
                    (long long) total, (long long) w_rows, (long long) w_rows);
    if (first != SIZE_MAX) std::exit(1);

    for (float* d : dst) cudaFree(d);
    cudaFree(x);
    cudaFree(xq);
    cudaFree(d_ids);
    cudaFree(d_bounds);
    cudaFree(d_ptrs);
    cudaFree(gather);
    cudaFree(arena);
}
/// The full chain prefill.cpp runs per MMQ group, gathered vs pointer-table, bit for bit at every stage:
/// gate/up product -> swiglu_scaled (the group's tails) -> q8_1 of H -> down product -> down_row_scales.
/// The blob carries the pack's eligible layout: [gu image][down image][16-byte tail], up_off = mmq_gub/2,
/// down_off = mmq_gub.  The gathered path copies every expert into its group slot (tails next to it); the
/// direct path gathers only the "streamed" experts (compacted) and points the resident ones at their arena
/// slots, tails by mmq::gather_tails.  Same random bytes everywhere, masked to 0x3F for finite scales.
void run_chain_case(const char* name, const std::vector<int64_t>& rows, std::mt19937& rng) {
    constexpr int64_t GU_R = 1280, DN_R = 2560, NC = 2560, NF = 640;
    const int n = (int) rows.size();
    int64_t total = 0, max_rows = 0;
    std::vector<int32_t> bounds = {0};
    for (int64_t r : rows) {
        total += r;
        max_rows = max_rows > r ? max_rows : r;
        bounds.push_back((int32_t) total);
    }
    const size_t gu_b = mmq::matrix_bytes(GGML_TYPE_NVFP4, GU_R, NC), dn_b = mmq::matrix_bytes(GGML_TYPE_NVFP4, DN_R, NF);
    const size_t blob = gu_b + dn_b + 16;
    const uint64_t slot = round256(blob) + 512;

    std::vector<uint8_t> host((size_t) n * blob);
    for (auto& b : host) b = (uint8_t) (rng() & 0x3F);
    void* arena = nullptr;
    ck(cudaMalloc(&arena, (size_t) n * slot), "cudaMalloc arena");
    std::vector<const uint8_t*> slotp;
    for (int e = 0; e < n; ++e) {
        slotp.push_back((const uint8_t*) arena + (size_t) e * slot);
        ck(cudaMemcpy((void*) slotp[e], host.data() + (size_t) e * blob, blob, cudaMemcpyHostToDevice), "fill slot");
    }
    // the group buffers of the two paths, each with room for the MMQ tail after the last slot
    void *g_gu = nullptr, *g_dn = nullptr, *d_gu = nullptr, *d_dn = nullptr;
    ck(cudaMalloc(&g_gu, (size_t) n * gu_b + 4096), "cudaMalloc g_gu");
    ck(cudaMalloc(&g_dn, (size_t) n * dn_b + 4096), "cudaMalloc g_dn");
    ck(cudaMalloc(&d_gu, (size_t) n * gu_b + 4096), "cudaMalloc d_gu");
    ck(cudaMalloc(&d_dn, (size_t) n * dn_b + 4096), "cudaMalloc d_dn");
    float *g_tail = nullptr, *d_tail = nullptr;
    ck(cudaMalloc(&g_tail, (size_t) n * 4 * 4), "cudaMalloc g_tail");
    ck(cudaMalloc(&d_tail, (size_t) n * 4 * 4), "cudaMalloc d_tail");

    // activations, once: the same q8_1 rows feed both paths' gate/up products
    std::vector<float> hx((size_t) total * NC);
    for (auto& v : hx) v = (float) ((int64_t) rng() % 2001 - 1000) / 1000.0f;
    std::vector<int32_t> ids(total);
    for (int64_t i = 0; i < total; ++i) ids[i] = (int32_t) i;
    for (int64_t i = total - 1; i > 0; --i) std::swap(ids[i], ids[rng() % (uint32_t) (i + 1)]);
    void *x = nullptr, *xq = nullptr, *d_ids = nullptr, *d_bounds = nullptr;
    ck(cudaMalloc(&x, hx.size() * sizeof(float)), "cudaMalloc x");
    ck(cudaMalloc(&xq, mmq::q8_bytes(total, NC)), "cudaMalloc xq");
    ck(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)), "cudaMalloc ids");
    ck(cudaMalloc(&d_bounds, bounds.size() * sizeof(int32_t)), "cudaMalloc bounds");
    ck(cudaMemcpy(x, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice), "fill x");
    ck(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "fill ids");
    ck(cudaMemcpy(d_bounds, bounds.data(), bounds.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "fill bounds");
    mmq::quantize((const float*) x, (const int32_t*) d_ids, xq, GGML_TYPE_NVFP4, NC, NC, total, nullptr);

    // the gathered path: every expert into its own slot, tails next to them
    for (int e = 0; e < n; ++e) {
        const uint8_t* b = slotp[e];
        mmq::gather_native(b, b + gu_b / 2, gu_b / 2, b + gu_b, dn_b, (uint8_t*) g_gu + (size_t) e * gu_b,
                           (uint8_t*) g_dn + (size_t) e * dn_b, nullptr, b + gu_b + dn_b, (uint8_t*) g_tail + (size_t) e * 16);
    }
    ck(cudaMemsetAsync((uint8_t*) g_gu + (size_t) n * gu_b, 0, 4096), "gathered gu tail");
    ck(cudaMemsetAsync((uint8_t*) g_dn + (size_t) n * dn_b, 0, 4096), "gathered dn tail");
    // the direct path: the EVEN experts are resident (arena slots), the odd ones stream into compacted slots
    std::vector<const void*> tab_gu(n), tab_dn(n), tab_tail(n, nullptr);
    int streamed = 0;
    for (int e = 0; e < n; ++e) {
        if (e % 2 == 0) {
            tab_gu[e] = slotp[e];
            tab_dn[e] = slotp[e] + gu_b;
            tab_tail[e] = slotp[e];
        } else {
            const uint8_t* b = slotp[e];   // a staging slot, for the test's purposes
            mmq::gather_native(b, b + gu_b / 2, gu_b / 2, b + gu_b, dn_b, (uint8_t*) d_gu + (size_t) streamed * gu_b,
                               (uint8_t*) d_dn + (size_t) streamed * dn_b, nullptr, b + gu_b + dn_b,
                               (uint8_t*) d_tail + (size_t) e * 16);
            tab_gu[e] = (const uint8_t*) d_gu + (size_t) streamed * gu_b;
            tab_dn[e] = (const uint8_t*) d_dn + (size_t) streamed * dn_b;
            ++streamed;
        }
    }
    ck(cudaMemsetAsync((uint8_t*) d_gu + (size_t) streamed * gu_b, 0, 4096), "direct gu tail");
    ck(cudaMemsetAsync((uint8_t*) d_dn + (size_t) streamed * dn_b, 0, 4096), "direct dn tail");
    void* d_tab = nullptr;
    ck(cudaMalloc(&d_tab, (size_t) 3 * n * sizeof(void*)), "cudaMalloc chain table");
    std::vector<const void*> table;
    table.reserve((size_t) 3 * n);
    table.insert(table.end(), tab_gu.begin(), tab_gu.end());
    table.insert(table.end(), tab_dn.begin(), tab_dn.end());
    table.insert(table.end(), tab_tail.begin(), tab_tail.end());
    ck(cudaMemcpy(d_tab, table.data(), table.size() * sizeof(void*), cudaMemcpyHostToDevice), "fill chain table");
    mmq::gather_tails((const void* const*) d_tab + 2 * (size_t) n, gu_b + dn_b, d_tail, n, nullptr);

    auto alloc3 = [&](float** p, int64_t els) { ck(cudaMalloc(p, (size_t) els * sizeof(float)), "cudaMalloc chain"); };
    float *GU[2] = {nullptr, nullptr}, *H[2] = {nullptr, nullptr}, *Dm[2] = {nullptr, nullptr}, *sd[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i) {
        alloc3(&GU[i], total * GU_R);
        alloc3(&H[i], total * NF);
        alloc3(&Dm[i], total * DN_R);
        alloc3(&sd[i], total);
    }
    mmq::Context ctx;
    for (int i = 0; i < 2; ++i) {   // i = 0: the gathered path; 1: the pointer table
        mmq::Product gu;
        gu.w = i ? d_gu : g_gu;
        gu.w_ptrs = i ? (const void* const*) d_tab : nullptr;
        gu.type = GGML_TYPE_NVFP4;
        gu.w_rows = GU_R;
        gu.w_cols = NC;
        gu.expert_bytes = gu_b;
        gu.n = n;
        gu.xq = xq;
        gu.bounds = (const int32_t*) d_bounds;
        gu.ids = (const int32_t*) d_ids;
        gu.total_rows = total;
        gu.max_rows = max_rows;
        gu.dst = GU[i];
        gu.ld_dst = GU_R;
        ctx.run(gu, nullptr);
        mmq::swiglu_scaled(GU[i], H[i], total, NF, (const int32_t*) d_bounds, n, i ? d_tail : g_tail, 0, nullptr);
    }
    int bad = 0;
    for (int64_t r = 0; r < total * GU_R && !bad; ++r) {
        float a = 0, b = 0;
        ck(cudaMemcpy(&a, GU[0] + r, 4, cudaMemcpyDeviceToHost), "read GU");
        ck(cudaMemcpy(&b, GU[1] + r, 4, cudaMemcpyDeviceToHost), "read GU2");
        if (std::memcmp(&a, &b, 4) != 0) bad = 1;
    }
    for (int64_t r = 0; r < total * NF && !bad; ++r) {
        float a = 0, b = 0;
        ck(cudaMemcpy(&a, H[0] + r, 4, cudaMemcpyDeviceToHost), "read H");
        ck(cudaMemcpy(&b, H[1] + r, 4, cudaMemcpyDeviceToHost), "read H2");
        if (std::memcmp(&a, &b, 4) != 0) bad = 2;
    }
    ++checks;
    if (bad) {
        std::fprintf(stderr, "FAIL: %s: the %s stage of the chain differs between the paths\n", name,
                     bad == 1 ? "gate/up product" : "swiglu_scaled");
        std::exit(1);
    }
    // the down products from the gathered path's H (bit-equal already, so one quantization serves both)
    void* hq = nullptr;
    ck(cudaMalloc(&hq, mmq::q8_bytes(total, NF)), "cudaMalloc hq");
    mmq::quantize(H[0], nullptr, hq, GGML_TYPE_NVFP4, NF, NF, total, nullptr);
    for (int i = 0; i < 2; ++i) {
        mmq::Product dn;
        dn.w = i ? d_dn : g_dn;
        dn.w_ptrs = i ? (const void* const*) d_tab + n : nullptr;
        dn.type = GGML_TYPE_NVFP4;
        dn.w_rows = DN_R;
        dn.w_cols = NF;
        dn.expert_bytes = dn_b;
        dn.n = n;
        dn.xq = hq;
        dn.bounds = (const int32_t*) d_bounds;
        dn.ids = (const int32_t*) d_ids;
        dn.total_rows = total;
        dn.max_rows = max_rows;
        dn.dst = Dm[i];
        dn.ld_dst = DN_R;
        ctx.run(dn, nullptr);
        mmq::down_row_scales(sd[i], (const int32_t*) d_bounds, n, i ? d_tail : g_tail, total, nullptr);
    }
    bad = 0;
    for (int64_t r = 0; r < total * DN_R && !bad; ++r) {
        float a = 0, b = 0;
        ck(cudaMemcpy(&a, Dm[0] + r, 4, cudaMemcpyDeviceToHost), "read Dm");
        ck(cudaMemcpy(&b, Dm[1] + r, 4, cudaMemcpyDeviceToHost), "read Dm2");
        if (std::memcmp(&a, &b, 4) != 0) bad = 1;
    }
    for (int64_t r = 0; r < total && !bad; ++r) {
        float a = 0, b = 0;
        ck(cudaMemcpy(&a, sd[0] + r, 4, cudaMemcpyDeviceToHost), "read sd");
        ck(cudaMemcpy(&b, sd[1] + r, 4, cudaMemcpyDeviceToHost), "read sd2");
        if (std::memcmp(&a, &b, 4) != 0) bad = 2;
    }
    ++checks;
    if (bad) {
        std::fprintf(stderr, "FAIL: %s: the %s stage of the chain differs between the paths\n", name,
                     bad == 1 ? "down product" : "down_row_scales");
        std::exit(1);
    }
    std::printf("PASS: %s: %d experts (%d resident, %d streamed), %lld rows - the chain bit-equal\n", name, n,
                n - (n + 1) / 2, (n + 1) / 2, (long long) total);

    for (int i = 0; i < 2; ++i) {
        cudaFree(GU[i]);
        cudaFree(H[i]);
        cudaFree(Dm[i]);
        cudaFree(sd[i]);
    }
    cudaFree(hq);
    cudaFree(d_tab);
    cudaFree(d_tail);
    cudaFree(g_tail);
    cudaFree(d_dn);
    cudaFree(d_gu);
    cudaFree(g_dn);
    cudaFree(g_gu);
    cudaFree(arena);
    cudaFree(x);
    cudaFree(xq);
    cudaFree(d_ids);
    cudaFree(d_bounds);
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    if (mmq::nvfp4_mode() != mmq::Nvfp4Mode::W4A8) {
        std::puts("STRATA_PREFILL_NVFP4 is not w4a8: the pointer-table path is W4A8-only, skipped");
        return 77;
    }
    if (!mmq::supported(GGML_TYPE_NVFP4)) {
        std::puts("NVFP4 MMQ not built: skipped");
        return 77;
    }
    std::mt19937 rng(12345);
    run_case("mixed group (an empty expert, a one-row expert)", 1024, 2560, {37, 0, 24}, rng);
    run_case("single expert", 1024, 2560, {29}, rng);
    run_case("fallback tiles (weights not a multiple of 128 rows)", 100, 2560, {5, 128, 1, 0}, rng);
    run_case("full 16-expert group", 512, 2560, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 130}, rng);
    run_chain_case("chain: mixed group with an empty expert", {37, 0, 24}, rng);
    run_chain_case("chain: full 16-expert group, alternating residency",
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 130}, rng);
    std::printf("prefill_mmq_direct_test: %d product shapes bit-equal\n", checks);
    return 0;
}
