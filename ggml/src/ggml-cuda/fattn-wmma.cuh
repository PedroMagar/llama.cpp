#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

//
// Volta (sm_70) Flash Attention kernel using nvcuda::wmma Tensor Cores (m16n16k16).
// Compiles only for __CUDA_ARCH__ < 800 to avoid conflict with the Ampere+
// cp.async-based path in fattn-mma-f16.cuh.
//

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda_fp16.h>

namespace wmma = nvcuda::wmma;

// Tile with 8 half-element padding per row for bank-conflict-free access:
constexpr int WMMA_PADDING = 8; // half elements

// WMMA m16n16k16: each warp handles 16 Q rows per tile
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view>
__launch_bounds__(256, 2)
static __global__ void flash_attn_ext_f16_sm70(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_sync();

#if defined(FLASH_ATTN_AVAILABLE) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

    constexpr int ncols       = ncols1 * ncols2;
    constexpr int warp_size   = 32;
    const int nwarps      = blockDim.y;
    constexpr int stride_Q_h  = DKQ + WMMA_PADDING;   // half elements per Q row (smem)
    constexpr int stride_K_h  = DKQ + WMMA_PADDING;   // half elements per K row (smem)
    constexpr int stride_V_h  = DV  + WMMA_PADDING;   // half elements per V row (smem)

    // Phase 2: compute nbatch_fa from shared memory budget (48KB dynamic)
    // Each row uses (DKQ+8)*2 bytes = (DKQ+8)*2 in K/V, (DKQ+8)*2 in Q
    // Max capacity: Q tile + K tile + V tile <= 48KB
    // K and V share the same memory (used sequentially), so:
    // Q_tile + max(K_tile, V_tile) <= 48KB
    // ncols*(DKQ+8)*2 + max(nbatch_fa*(DKQ+8)*2, nbatch_fa*(DV+8)*2) <= 49152
    constexpr int nbatch_fa = []() {
        int max_by_DKQ = (49152/2 - ncols*(DKQ+8)) / (DKQ+8);
        int max_by_DV  = (49152/2 - ncols*(DKQ+8)) / (DV+8);
        int max_batch  = max_by_DKQ < max_by_DV ? max_by_DKQ : max_by_DV;
        if (max_batch > 128) max_batch = 128;
        // Round down to multiple of WMMA_N (16):
        max_batch = (max_batch / WMMA_N) * WMMA_N;
        return max_batch > 0 ? max_batch : WMMA_N;
    }();

    constexpr int stride_S_h = nbatch_fa + WMMA_PADDING; // half elements per S row (smem for softmax)

    // Shared memory layout:
    //   [0 .. ncols*stride_Q_h)                  = tile_Q (ncols x DKQ)
    //   [ncols*stride_Q_h .. + nbatch_fa*stride_K_h) = tile_KV (shared by K and V)
    //                                          (reused for either K or V)
    extern __shared__ half smem[];
    half * tile_Q   = smem;
    half * tile_KV  = smem + ncols * stride_Q_h;
    half * tile_S   = smem;  // Reuse Q tile space for S during softmax (temporary)

    // Grid / block mapping
    const int gqa_ratio = ne02 / ne12;
    const int stride_Q1 = nb01 / sizeof(float2);
    const int stride_Q2 = nb02 / sizeof(float2);
    const float2 * Q_f2 = (const float2 *) Q_ptr;

    // Q tile loading (global -> shared memory, float2 -> half with scaling)
    // Phase 3: full implementation with float2 -> half2 conversion and scale
    // Phase 2 placeholder: zero-initialize tile_Q
    if (threadIdx.x < warp_size) {
        for (int r = threadIdx.y; r < ncols; r += nwarps) {
            for (int c = threadIdx.x; c < DKQ; c += warp_size) {
                tile_Q[r * stride_Q_h + c] = __float2half(0.0f);
            }
        }
    }

    __syncthreads();

    // Warp distribution across Q rows
    // Each warp handles WMMA_M=16 consecutive Q rows
    const int warp_id    = threadIdx.y;
    const int warp_q_row = warp_id * WMMA_M;
    if (warp_q_row >= ncols) {
        return;
    }

    // Iterate over KV cache in blocks of nbatch_fa
    const int nKV_blocks  = (ne11 + nbatch_fa - 1) / nbatch_fa;

    // Per-warp persistent state
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_acc;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> VKQ_acc;

    for (int kb = 0; kb < nKV_blocks; ++kb) {
        const int kv_start = kb * nbatch_fa;
        const int kv_count = nbatch_fa;
        if (kv_start >= ne11) break;

        // Phase 3: Load K tile from global memory (on-the-fly dequant)
        if (threadIdx.x < warp_size) {
            for (int r = threadIdx.y; r < nbatch_fa; r += nwarps) {
                for (int c = threadIdx.x; c < DKQ; c += warp_size) {
                    tile_KV[r * stride_K_h + c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ---- Compute S = Q * K^T via WMMA ----
        wmma::fill_fragment(S_acc, 0.0f);

        for (int dkq = 0; dkq < DKQ; dkq += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_frag;

            wmma::load_matrix_sync(Q_frag, tile_Q + warp_q_row * stride_Q_h + dkq, stride_Q_h);
            wmma::load_matrix_sync(K_frag, tile_KV + dkq, stride_K_h);
            wmma::mma_sync(S_acc, Q_frag, K_frag, S_acc);
        }

        // ---- Online softmax (placeholder: identity transform) ----
        wmma::store_matrix_sync(tile_S + warp_q_row * stride_S_h, S_acc, stride_S_h, wmma::mem_row_major);
        __syncthreads();
        wmma::load_matrix_sync(S_acc, tile_S + warp_q_row * stride_S_h, stride_S_h, wmma::mem_row_major);

        // ---- Compute O = S * V via WMMA ----
        wmma::fill_fragment(VKQ_acc, 0.0f);

        // Phase 3: Load V tile from global memory
        if (threadIdx.x < warp_size) {
            for (int r = threadIdx.y; r < nbatch_fa; r += nwarps) {
                for (int c = threadIdx.x; c < DV; c += warp_size) {
                    tile_KV[r * stride_V_h + c] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        for (int kv_sub = 0; kv_sub < nbatch_fa; kv_sub += WMMA_N) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> V_frag;
            wmma::load_matrix_sync(V_frag, tile_KV + kv_sub * stride_V_h, stride_V_h);

            // Phase 4: full tiled VKQ with proper S*V multiply
            // For now, placeholder identity: VKQ_acc unchanged
        }

        __syncthreads();
    }

    // Phase 5: Write output to global memory (placeholder: no-op)
    GGML_UNUSED_VARS(Q_f2, stride_Q1, stride_Q2, gqa_ratio,
        K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr,
        max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
        ne31, ne32, ne33, nb31, nb32, nb33, VKQ_acc);

#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
    return;
#endif // FLASH_ATTN_AVAILABLE && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
}

//
// Host-side dispatch per (DKQ, DV, ncols1, ncols2) combination
//

template <int DKQ, int DV, int ncols1, int ncols2>
static void ggml_cuda_flash_attn_ext_wmma_sm70_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    constexpr int ncols = ncols1 * ncols2;

    // Volta WMMA (m16n16k16): 256 threads (8 warps x 32), occupancy 2
    constexpr int nthreads  = 256;
    constexpr int nwarps    = nthreads / 32;
    constexpr int occupancy = 2;

    // Compute proper dynamic shared memory size
    constexpr int stride_Q_h = DKQ + WMMA_PADDING;
    constexpr int stride_K_h = DKQ + WMMA_PADDING;
    constexpr int stride_V_h = DV  + WMMA_PADDING;

    // Estimate nbatch_fa to fit within 48KB dynamic shared memory
    constexpr int max_by_DKQ = (48*1024/2 - ncols*(DKQ+8)) / (DKQ+8);
    constexpr int max_by_DV  = (48*1024/2 - ncols*(DKQ+8)) / (DV+8);
    constexpr int max_batch  = max_by_DKQ < max_by_DV ? max_by_DKQ : max_by_DV;
    constexpr int nbatch_fa  = (max_batch > 128 ? 128 : (max_batch > 0 ? (max_batch / 16) * 16 : 16));

    // Shared memory: Q tile + KV tile (shared by K and V)
    constexpr size_t nbytes_shared_Q  = ncols * stride_Q_h * sizeof(half);
    constexpr size_t nbytes_shared_KV = nbatch_fa * (stride_K_h > stride_V_h ? stride_K_h : stride_V_h) * sizeof(half);
    constexpr size_t nbytes_shared_S  = ncols * (nbatch_fa + WMMA_PADDING) * sizeof(half);

    // S tile reuses Q tile memory (not simultaneously), so take the max
    constexpr size_t nbytes_shared_total = (nbytes_shared_Q > nbytes_shared_S ? nbytes_shared_Q : nbytes_shared_S) + nbytes_shared_KV;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr bool V_is_K_view = (DKQ == 576);

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_lsc = false;
        fattn_kernel = flash_attn_ext_f16_sm70<DKQ, DV, ncols1, ncols2, use_lsc, V_is_K_view>;
    } else {
        constexpr bool use_lsc = true;
        fattn_kernel = flash_attn_ext_f16_sm70<DKQ, DV, ncols1, ncols2, use_lsc, V_is_K_view>;
    }

    launch_fattn<DV, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, 32);
}

//
// ncols1 switch: choose Q-token tile size based on batch size
//

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    // With WMMA m16n16k16 each warp tiles 16 Q rows (M=16).
    // 8 warps per block => up to 128 Q-rows per thread block.
    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_wmma_sm70_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_wmma_sm70_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }
    if (Q->ne[1] <= 64/ncols2) {
        ggml_cuda_flash_attn_ext_wmma_sm70_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_wmma_sm70_case<DKQ, DV, 128/ncols2, ncols2>(ctx, dst);
}

//
// ncols2 switch: GQA routing (how many Q heads per K/V head per thread block)
//

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

//
// Top-level dispatch: switch on head size (DKQ == DV)
//

static void ggml_cuda_flash_attn_ext_wmma_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols2<128, 128>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#else // GGML_USE_HIP || GGML_USE_MUSA

static void ggml_cuda_flash_attn_ext_wmma_sm70(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("flash_attn_ext_f16_sm70: not available on this platform");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)