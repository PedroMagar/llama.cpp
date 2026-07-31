#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

//
// Volta (sm_70) Flash Attention kernel using nvcuda::wmma Tensor Cores (m16n16k16).
// Compiles only for __CUDA_ARCH__ < 800 to avoid conflict with the Ampere+
// cp.async-based path in fattn-mma-f16.cuh.
//

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

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
    // =========================================================================
    // Phase 2+: WMMA kernel body with nvcuda::wmma fragments, online softmax,
    //           on-the-fly dequantization of Q8_0/Q4_0 KV cache, GQA, and RoPE.
    // =========================================================================
    NO_DEVICE_CODE;
#else
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
    GGML_UNUSED_VARS(ncols, occupancy);

    // Phase 2 will compute the exact size; 48KB is a safe placeholder
    // covering all reasonable tile configurations for D <= 128, nbatch_fa <= 128.
    const size_t nbytes_shared = 48 * 1024;

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

    launch_fattn<DV, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, 128, false, false, true, 32);
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