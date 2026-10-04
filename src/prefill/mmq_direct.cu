// src/prefill/mmq_direct.cu - the resident experts' weights read straight from their cache slots.
//
// mmq.cuh's kernel reads one gathered image: every expert the product touches is copied into a contiguous buffer
// first (gather_native_group), which for the deep-context turns measured in post-restore-slow-tail.md is ~11 GB of
// D2D per turn - most of it experts the expert cache ALREADY holds.  This TU compiles the same W4A8 NVFP4 path as
// mmq_nvfp4_w4a8.cu but with STRATA_MMQ_PTR_TABLE: the kernel takes one base pointer per expert (the slot's, or a
// streamed expert's group slot) and reads the weights where they live.  The rename trick is the w4a8 shim's: the
// two non-static templates get their own names so the linker keeps the three NVFP4 instantiations apart.
#include "common.cuh"

#define STRATA_MMQ_PTR_TABLE
#undef BLACKWELL_MMA_AVAILABLE
#define blackwell_mma_available(cc) false
#define mul_mat_q_switch_J strata_direct_mul_mat_q_switch_J
#define mul_mat_q_case strata_direct_mul_mat_q_case
#include "mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace strata::prefill::mmq {
void run_direct(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_direct_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}
}  // namespace strata::prefill::mmq
