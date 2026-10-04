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
    std::printf("prefill_mmq_direct_test: %d product shapes bit-equal\n", checks);
    return 0;
}
