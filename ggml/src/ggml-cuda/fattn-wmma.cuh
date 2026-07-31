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
constexpr int WMMA_PADDING = 8;

// WMMA m16n16k16: each warp handles 16 Q rows per tile
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// -----------------------------------------------------------------------
// On-the-fly dequantization helpers: quantized block -> half elements
// -----------------------------------------------------------------------

__device__ __forceinline__ void dequant_q4_0_to_half(const block_q4_0 & blk, half * out) {
    const float df = __half2float(blk.d);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint8_t qs = blk.qs[i];
        out[2*i+0] = __float2half(((qs & 0x0F) - 8) * df);
        out[2*i+1] = __float2half((((qs >> 4) & 0x0F) - 8) * df);
    }
}

__device__ __forceinline__ void dequant_q8_0_to_half(const block_q8_0 & blk, half * out) {
    const float df = __half2float(blk.d);
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[i] = __float2half(blk.qs[i] * df);
    }
}

// -----------------------------------------------------------------------
// KV tile loader: global memory -> shared memory with on-the-fly dequant
// -----------------------------------------------------------------------

enum class KVType : int { F16, Q4_0, Q8_0 };

// Detect KV type from byte stride nb1 and column count ncols
__device__ __forceinline__ KVType detect_kv_type(int64_t nb1, int ncols) {
    if (nb1 == (int64_t)ncols * (int64_t)sizeof(half)) {
        return KVType::F16;
    }
    if (nb1 == (int64_t)(ncols / QK4_0) * (int64_t)sizeof(block_q4_0)) {
        return KVType::Q4_0;
    }
    // Q8_0: sizeof(block_q8_0) = 34, QK8_0 = 32
    return KVType::Q8_0;
}

// Load a tile of K or V from global memory into shared memory.
// nrows = nbatch_fa, ncols = DKQ (or DV for V).
// Global memory layout is row-major with stride nb1 (bytes between rows).
// Shared memory layout is row-major with stride smem_stride (half elements).
template<int smem_stride>
__device__ void load_KV_tile(
    const char * src, half * dst, int nrows, int ncols,
    int64_t nb1, KVType type)
{
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nt  = blockDim.x * blockDim.y;

    switch (type) {
    case KVType::F16: {
        const half * src_h = (const half *)src;
        const int src_stride = (int)(nb1 / sizeof(half));
        for (int idx = tid; idx < nrows * ncols; idx += nt) {
            const int r = idx / ncols;
            const int c = idx % ncols;
            dst[r * smem_stride + c] = src_h[r * src_stride + c];
        }
        break;
    }
    case KVType::Q4_0: {
        const int blocks_per_row = ncols / QK4_0;
        const int total_blocks = nrows * blocks_per_row;
        for (int b = tid; b < total_blocks; b += nt) {
            const int r = b / blocks_per_row;
            const int blk = b % blocks_per_row;
            const block_q4_0 * blk_ptr = (const block_q4_0 *)(src + r * nb1) + blk;
            half tmp[QK4_0];
            dequant_q4_0_to_half(*blk_ptr, tmp);
            const int col_base = blk * QK4_0;
            #pragma unroll
            for (int i = 0; i < QK4_0; ++i) {
                dst[r * smem_stride + col_base + i] = tmp[i];
            }
        }
        break;
    }
    case KVType::Q8_0: {
        const int blocks_per_row = ncols / QK8_0;
        const int total_blocks = nrows * blocks_per_row;
        for (int b = tid; b < total_blocks; b += nt) {
            const int r = b / blocks_per_row;
            const int blk = b % blocks_per_row;
            const block_q8_0 * blk_ptr = (const block_q8_0 *)(src + r * nb1) + blk;
            half tmp[QK8_0];
            dequant_q8_0_to_half(*blk_ptr, tmp);
            const int col_base = blk * QK8_0;
            #pragma unroll
            for (int i = 0; i < QK8_0; ++i) {
                dst[r * smem_stride + col_base + i] = tmp[i];
            }
        }
        break;
    }
    }
}

// -----------------------------------------------------------------------
// Q tile loader: F32 global memory -> half shared memory with scale
// -----------------------------------------------------------------------

// Load Q tile from F32 global memory into half shared memory with scale.
// Q global layout: [seq][head][token][dkq/2] as float2, stride_Q1 = tokens, stride_Q2 = heads.
// Shared memory: [ncols x DKQ] half elements with stride smem_stride.
template<int smem_stride>
__device__ void load_Q_tile(
    const float2 * src, half * dst, int nrows, int ncols,
    int stride_Q1, int stride_Q2, int jt, int ncols2, half scale_h)
{
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nt  = blockDim.x * blockDim.y;
    const float sf = __half2float(scale_h);

    // Each thread loads one or more float2 pairs and expands to 2 half elements.
    const int total_f2 = nrows * (DKQ/2);
    for (int idx = tid; idx < total_f2; idx += nt) {
        const int jc = idx / (DKQ/2);
        const int k  = idx % (DKQ/2);
        const int j  = (jt * ncols1) + (jc / ncols2);
        const int c  = jc % ncols2;

        const float2 tmp = src[j * stride_Q1 + c * stride_Q2 + k];
        dst[jc * smem_stride + 2*k + 0] = __float2half(tmp.x * sf);
        dst[jc * smem_stride + 2*k + 1] = __float2half(tmp.y * sf);
    }
}

// =========================================================================
// Main WMMA Flash Attention kernel
// =========================================================================

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
    const int nwarps          = blockDim.y;
    constexpr int stride_Q_h  = DKQ + WMMA_PADDING;
    constexpr int stride_K_h  = DKQ + WMMA_PADDING;
    constexpr int stride_V_h  = DV  + WMMA_PADDING;

    // Compute nbatch_fa from shared memory budget (48KB dynamic)
    constexpr int nbatch_fa = []() {
        int max_by_DKQ = (48*1024/2 - ncols*(DKQ+8)) / (DKQ+8);
        int max_by_DV  = (48*1024/2 - ncols*(DKQ+8)) / (DV+8);
        int max_batch  = max_by_DKQ < max_by_DV ? max_by_DKQ : max_by_DV;
        if (max_batch > 128) max_batch = 128;
        max_batch = (max_batch / WMMA_N) * WMMA_N;
        return max_batch > 0 ? max_batch : WMMA_N;
    }();

    constexpr int stride_S_h   = nbatch_fa + WMMA_PADDING;
    constexpr int nKV_blocks   = (ne11 + nbatch_fa - 1) / nbatch_fa;

    constexpr int smem_QKV_offset = ncols * (stride_Q_h > stride_S_h ? stride_Q_h : stride_S_h);

    // Shared memory layout:
    extern __shared__ half smem[];
    half * tile_Q   = smem;
    half * tile_KV  = smem + smem_QKV_offset;
    half * tile_S   = smem;

    // Grid/block tile mapping: blockIdx.x encodes the output tile index
    const int gqa_ratio  = ne02 / ne12;
    const int stride_Q1  = nb01 / sizeof(float2);
    const int stride_Q2  = nb02 / sizeof(float2);
    const float2 * Q_f2  = (const float2 *) Q_ptr;
    const char   * K_raw = K_ptr;
    const char   * V_raw = V_ptr;

    // Detect K/V types at runtime from byte strides
    const KVType kv_K_type = detect_kv_type(nb11, DKQ);
    const KVType kv_V_type = detect_kv_type(nb21, DV);

    // Unpack tile index: sequence->K/V head->GQA head->token position
    const int iter_j      = (ne01.z + ncols1 - 1) / ncols1;
    const int iter_z_gqa  = (gqa_ratio + ncols2 - 1) / ncols2;
    const int tiles_per_seq = iter_j * iter_z_gqa * ne12;

    const int tile_global = blockIdx.x;
    const int seq         = tile_global / tiles_per_seq;
    const int tile_in_seq = tile_global % tiles_per_seq;
    const int z_KV        = tile_in_seq / (iter_j * iter_z_gqa);
    const int tile_in_KV  = tile_in_seq % (iter_j * iter_z_gqa);
    const int z_gqa       = tile_in_KV / iter_j;
    const int jt          = tile_in_KV % iter_j;
    const int zt_Q        = z_KV * gqa_ratio + z_gqa * ncols2;

    // ---- Q load: F32 -> half with scale ----
    {
        const half scale_h = __float2half(scale);
        const float2 * Q_src = Q_f2 + (seq * ne02 + zt_Q) * stride_Q2;

        load_Q_tile<stride_Q_h>(
            Q_src, tile_Q, ncols, DKQ, stride_Q1, stride_Q2, jt, ncols2, scale_h);
    }
    __syncthreads();

    // Warp distribution across Q rows
    const int warp_id    = threadIdx.y;
    const int warp_q_row = warp_id * WMMA_M;
    if (warp_q_row >= ncols) {
        return;
    }

    // Number of DV tiles per VKQ accumulator
    constexpr int DV_tiles = DV / WMMA_K;

    // Per-warp online softmax state
    float row_max[WMMA_M];
    float row_sum[WMMA_M];

    // VKQ accumulator: one WMMA fragment per DV tile
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> VKQ_acc[DV_tiles];

    // Online softmax rescaling buffer (16x16 float = 1KB)
    // Located at the end of tile_KV, reusing shared memory after V is loaded
    float * tile_rescale = (float *)(tile_KV + nbatch_fa * (stride_K_h > stride_V_h ? stride_K_h : stride_V_h));

    // Initialize online softmax state
    #pragma unroll
    for (int i = 0; i < WMMA_M; ++i) {
        row_max[i] = -FLT_MAX;
        row_sum[i] = 0.0f;
    }
    #pragma unroll
    for (int t = 0; t < DV_tiles; ++t) {
        wmma::fill_fragment(VKQ_acc[t], 0.0f);
    }

    for (int kb = 0; kb < nKV_blocks; ++kb) {
        const int kv_start = kb * nbatch_fa;
        if (kv_start >= ne11) break;
        const int kv_count = min(nbatch_fa, ne11 - kv_start);

        // Zero-initialize unused rows of tile_KV for the partial last block
        if (kv_count < nbatch_fa) {
            const int tid = threadIdx.x + threadIdx.y * warp_size;
            const int nt  = blockDim.x * blockDim.y;
            for (int idx = tid; idx < nbatch_fa * stride_K_h; idx += nt) {
                tile_KV[idx] = __float2half(0.0f);
            }
            __syncthreads();
        }

        // ---- Phase A: Load K tile ----
        load_KV_tile<stride_K_h>(K_raw + kv_start * nb11, tile_KV, kv_count, DKQ, nb11, kv_K_type);
        __syncthreads();

        // ---- Phase A: Compute S = Q * K^T and online softmax, store P to tile_S ----
        for (int kv_sub = 0; kv_sub < kv_count; kv_sub += WMMA_N) {
            const int kv_sub_count = min(WMMA_N, kv_count - kv_sub);

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
            wmma::fill_fragment(S_frag, 0.0f);

            for (int dkq = 0; dkq < DKQ; dkq += WMMA_K) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Q_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> K_frag;

                wmma::load_matrix_sync(Q_frag, tile_Q + warp_q_row * stride_Q_h + dkq, stride_Q_h);
                wmma::load_matrix_sync(K_frag, tile_KV + kv_sub * stride_K_h + dkq, stride_K_h);
                wmma::mma_sync(S_frag, Q_frag, K_frag, S_frag);
            }

            wmma::store_matrix_sync(
                tile_S + warp_q_row * stride_S_h + kv_sub,
                S_frag, stride_S_h, wmma::mem_row_major);
            __syncthreads();

            // Step 1: compute new max for all 16 Q rows (batched)
            float new_max_vals[WMMA_M];
            #pragma unroll
            for (int r = 0; r < WMMA_M; ++r) {
                float local_max = -FLT_MAX;
                #pragma unroll
                for (int c = threadIdx.x; c < WMMA_N && c < kv_sub_count; c += warp_size) {
                    local_max = fmaxf(local_max, __half2float(tile_S[(warp_q_row + r) * stride_S_h + kv_sub + c]));
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset, warp_size));
                }
                new_max_vals[r] = fmaxf(row_max[r], local_max);
            }

            // Step 2: batch rescale VKQ accumulators (once per tile, not per row)
            {
                bool need_rescale = false;
                float scale_vals[WMMA_M];
                #pragma unroll
                for (int r = 0; r < WMMA_M; ++r) {
                    float max_diff = row_max[r] - new_max_vals[r];
                    float scale = expf(max_diff);
                    *((uint32_t *)&scale) *= max_diff >= SOFTMAX_FTZ_THRESHOLD;
                    scale_vals[r] = scale;
                    if (row_max[r] < new_max_vals[r]) {
                        need_rescale = true;
                    }
                }

                if (need_rescale) {
                    #pragma unroll
                    for (int t = 0; t < DV_tiles; ++t) {
                        wmma::store_matrix_sync(
                            (half *)tile_rescale, VKQ_acc[t], WMMA_N, wmma::mem_row_major);
                        __syncthreads();

                        if (threadIdx.x < WMMA_N) {
                            #pragma unroll
                            for (int r = 0; r < WMMA_M; ++r) {
                                if (row_max[r] < new_max_vals[r]) {
                                    tile_rescale[r * WMMA_N + threadIdx.x] *= scale_vals[r];
                                }
                            }
                        }
                        __syncthreads();

                        wmma::load_matrix_sync(
                            VKQ_acc[t], (half *)tile_rescale, WMMA_N, wmma::mem_row_major);
                    }
                }
            }

            // Step 3: compute exp, update state, write P back
            #pragma unroll
            for (int r = 0; r < WMMA_M; ++r) {
                row_max[r] = new_max_vals[r];

                float local_sum = 0.0f;
                #pragma unroll
                for (int c = threadIdx.x; c < WMMA_N && c < kv_sub_count; c += warp_size) {
                    const int offset = (warp_q_row + r) * stride_S_h + kv_sub + c;
                    const float diff = __half2float(tile_S[offset]) - new_max_vals[r];
                    const float val = diff < SOFTMAX_FTZ_THRESHOLD ? 0.0f : expf(diff);
                    tile_S[offset] = __float2half(val);
                    local_sum += val;
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset, warp_size);
                }
                row_sum[r] = row_sum[r] * scale_vals[r] + local_sum;
            }
            __syncthreads();
        }

        // ---- Phase B: Load V tile ----
        if (kv_count < nbatch_fa) {
            const int tid = threadIdx.x + threadIdx.y * warp_size;
            const int nt  = blockDim.x * blockDim.y;
            for (int idx = tid; idx < nbatch_fa * stride_V_h; idx += nt) {
                tile_KV[idx] = __float2half(0.0f);
            }
            __syncthreads();
        }
        load_KV_tile<stride_V_h>(V_raw + kv_start * nb21, tile_KV, kv_count, DV, nb21, kv_V_type);
        __syncthreads();

        // ---- Phase B: Compute O = P * V via WMMA ----
        #pragma unroll
        for (int kv_sub = 0; kv_sub < kv_count; kv_sub += WMMA_N) {
            #pragma unroll
            for (int t = 0; t < DV_tiles; ++t) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> P_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> V_frag;

                wmma::load_matrix_sync(
                    P_frag,
                    tile_S + warp_q_row * stride_S_h + kv_sub,
                    stride_S_h);

                wmma::load_matrix_sync(
                    V_frag,
                    tile_KV + kv_sub * stride_V_h + t * WMMA_K,
                    stride_V_h);

                wmma::mma_sync(VKQ_acc[t], P_frag, V_frag, VKQ_acc[t]);
            }
        }
    }

    // ---- Final normalization and output ----
    {
        float2 * dst_f2 = ((float2 *)dst_ptr) + (seq * ne01.z * ne02 + zt_Q) * (DV/2);

        #pragma unroll
        for (int t = 0; t < DV_tiles; ++t) {
            // Store VKQ tile to rescale buffer
            wmma::store_matrix_sync(
                (half *)tile_rescale, VKQ_acc[t], WMMA_N, wmma::mem_row_major);
            __syncthreads();

            const int dv_base = t * WMMA_K;
            const int tid = threadIdx.x + threadIdx.y * warp_size;
            const int nt  = blockDim.x * blockDim.y;

            // Each thread writes one float2 (2 consecutive DV elements)
            const int total_pairs = WMMA_M * (WMMA_N / 2);
            for (int idx = tid; idx < total_pairs; idx += nt) {
                const int r      = idx / (WMMA_N / 2);
                const int c_pair = idx % (WMMA_N / 2);
                const int c0     = c_pair * 2;
                const int c1     = c0 + 1;
                const int dv0    = dv_base + c0;
                const int dv1    = dv_base + c1;

                if (dv1 >= DV) continue;

                const float inv_sum = 1.0f / fmaxf(row_sum[r], 1e-10f);
                const float v0 = tile_rescale[r * WMMA_N + c0] * inv_sum;
                const float v1 = tile_rescale[r * WMMA_N + c1] * inv_sum;

                const int j      = (warp_q_row + r) / ncols2;
                const int c_head = (warp_q_row + r) % ncols2;
                const int64_t token = (int64_t)jt * ncols1 + j;
                const int64_t head  = zt_Q + c_head;
                const int64_t f2_idx = (token * ne02 + head) * (DV/2) + dv0/2;

                dst_f2[f2_idx] = make_float2(v0, v1);
            }
            __syncthreads();
        }
    }

    GGML_UNUSED_VARS(Q_f2, stride_Q1, stride_Q2, gqa_ratio,
        mask_ptr, sinks_ptr, KV_max_ptr, dst_meta_ptr,
        max_bias, m0, m1, n_head_log2, ne00, ne01, ne02, ne03,
        ne11, ne12, ne13, nb12, nb13, nb22, nb23,
        ne31, ne32, ne33, nb31, nb32, nb33);

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
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

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

    // Shared memory: Q tile + KV tile (shared by K and V) + rescale buffer (16x16 float)
    constexpr size_t nbytes_shared_Q  = ncols * stride_Q_h * sizeof(half);
    constexpr size_t nbytes_shared_KV = nbatch_fa * (stride_K_h > stride_V_h ? stride_K_h : stride_V_h) * sizeof(half);
    constexpr size_t nbytes_shared_S  = ncols * (nbatch_fa + WMMA_PADDING) * sizeof(half);
    constexpr size_t nbytes_shared_Q_or_S = nbytes_shared_Q > nbytes_shared_S ? nbytes_shared_Q : nbytes_shared_S;
    constexpr size_t nbytes_shared_rescale = WMMA_M * WMMA_N * sizeof(float); // 1024 bytes
    constexpr size_t nbytes_shared_total = nbytes_shared_Q_or_S + nbytes_shared_KV + nbytes_shared_rescale;

    // Compute grid dimensions: one block per output tile
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int iter_j       = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int iter_z_gqa   = (gqa_ratio + ncols2 - 1) / ncols2;
    const int total_tiles  = iter_j * iter_z_gqa * K->ne[2] * Q->ne[3];

    const dim3 block_dim(32, nwarps, 1);
    const dim3 grid_dim(total_tiles, 1, 1);
    const uint3 ne01_fd = init_fastdiv_values(Q->ne[1]);

    float scale_val = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale_val,      (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&logit_softcap,  (const float *) KQV->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale_val /= logit_softcap;
    }

    constexpr bool V_is_K_view = (DKQ == 576);

    const char * K_data = (const char *) K->data;
    const char * V_data = (const char *) V->data;

    auto launch_kernel = [&](auto kernel_ptr) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nbytes_shared_total));
        kernel_ptr<<<grid_dim, block_dim, nbytes_shared_total, ctx.stream()>>>(
            (const char *)Q->data, K_data, V_data,
            (const char *)nullptr,
            (const char *)nullptr,
            (const int   *)nullptr,
            (float       *)dst->data,
            (float2      *)nullptr,
            scale_val, 0.0f, 0.0f, 0.0f, (uint32_t)0, logit_softcap,
            Q->ne[0], ne01_fd, Q->ne[2], Q->ne[3],
            (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int32_t)Q->nb[3],
            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
            (int32_t)K->nb[1], (int32_t)K->nb[2], (int64_t)K->nb[3],
            (int32_t)V->nb[1], (int32_t)V->nb[2], (int64_t)V->nb[3],
            0, 0, 0, 0, 0, 0, 0);
    };

    if (logit_softcap == 0.0f) {
        launch_kernel(flash_attn_ext_f16_sm70<DKQ, DV, ncols1, ncols2, false, V_is_K_view>);
    } else {
        launch_kernel(flash_attn_ext_f16_sm70<DKQ, DV, ncols1, ncols2, true, V_is_K_view>);
    }
}

//
// ncols1 switch: choose Q-token tile size based on batch size
//

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_wmma_sm70_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

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
// ncols2 switch: GQA routing
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