#pragma once

#include "../../FiberJobSystem/JobSystem.h"
#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).
#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <print>       // Formatting
#include <cstdint>
#include <array>
#include <mdspan>
#include <cstddef>
#include <span>
#include <vector>
#include <algorithm>   // Required for std::min, std::copy, std::swap
#include <utility>     // Required for std::swap (in some compilers)
#include <numeric>     // Standard header for std::reduce algorithms
#include <bit>         // Required for std::bit_cast

// ================================================================================
// AVX-512: This represents 16 vectors. It doesn't own memory; it just processes it.
// ================================================================================

// --- 1. THE MATH LAYER (Intel-specific Intrinsics SIMD) ---
struct SIMDVector16 {
    // --- AVX-512: 16-Wide Batch ---
    __m512 x, y, z;

    void add(const __m512& bx, const __m512& by, const __m512& bz) {
        x = _mm512_add_ps(x, bx);
        y = _mm512_add_ps(y, by);
        z = _mm512_add_ps(z, bz);
    }

    __m512 dot_fma(const __m512& bx, const __m512& by, const __m512& bz) const {
        __m512 res = _mm512_mul_ps(x, bx);
        res = _mm512_fmadd_ps(y, by, res);
        res = _mm512_fmadd_ps(z, bz, res);
        return res;
    }

    __m512 dot(const __m512& bx, const __m512& by, const __m512& bz) const {
        __m512 mx = _mm512_mul_ps(x, bx);
        __m512 my = _mm512_mul_ps(y, by);
        __m512 mz = _mm512_mul_ps(z, bz);
        return _mm512_add_ps(_mm512_add_ps(mx, my), mz);
    }

    void cross(const __m512& bx, const __m512& by, const __m512& bz) {
        // OPTIMIZATION: Fused Multiply-Subtract reduces 6 instructions to 3.
        // Formula: (y * bz) - (z * by)
        __m512 rx = _mm512_fmsub_ps(y, bz, _mm512_mul_ps(z, by));
        __m512 ry = _mm512_fmsub_ps(z, bx, _mm512_mul_ps(x, bz));
        __m512 rz = _mm512_fmsub_ps(x, by, _mm512_mul_ps(y, bx));
        x = rx; y = ry; z = rz;
    }

    void sub(const __m512& bx, const __m512& by, const __m512& bz) {
        x = _mm512_sub_ps(x, bx);
        y = _mm512_sub_ps(y, by);
        z = _mm512_sub_ps(z, bz);
    }

    __m512 length_sq() const {
        // OPTIMIZATION: Fused Multiply-Add
        __m512 xx = _mm512_mul_ps(x, x);
        __m512 xx_yy = _mm512_fmadd_ps(y, y, xx);
        return _mm512_fmadd_ps(z, z, xx_yy);
        // __m512 xx = _mm512_mul_ps(x, x);
        // __m512 yy = _mm512_mul_ps(y, y);
        // __m512 zz = _mm512_mul_ps(z, z);
        // return _mm512_add_ps(_mm512_add_ps(xx, yy), zz);
    }

    __m512 length() const {
        return _mm512_sqrt_ps(length_sq());
    }

    // --- AVX-512 EXCLUSIVE OPMASK LOGIC ---
    void normalize() {
        __m512 sqLen = length_sq();
        __mmask16 mask = _mm512_cmp_ps_mask(sqLen, _mm512_set1_ps(1e-8f), _CMP_GT_OQ);
        
        // 1. Generate an Opmask (16 bits) instead of a 512-bit float mask
        // __mmask16 mask = _mm512_cmp_ps_mask(len, epsilon, _CMP_GT_OQ); 
        
        // OPTIMIZATION: Masked Approximate Reciprocal Square Root
        // _mm512_maskz_rsqrt14_ps returns ~1.0/sqrt(sqLen). 
        // Crucially, if the mask bit is 0, it places 0.0f in that lane automatically!
        __m512 invLen = _mm512_maskz_rsqrt14_ps(mask, sqLen);
        
        // 2. Masked Multiplication. 
        // Because invLen is 0.0f in invalid lanes, standard multiplication zeroes out x/y/z safely.
        x = _mm512_mul_ps(x, invLen);
        y = _mm512_mul_ps(y, invLen);
        z = _mm512_mul_ps(z, invLen);
    }
};

// --- 2. THE SYSTEM LAYER (16-Wide Alignment) ---
class VectorManagerSOA_V3_AVX512 {
public:
    // Using our 64-byte aligned vectors
    AlignedVector64<float> xs, ys, zs;

    VectorManagerSOA_V3_AVX512(size_t count) {
        // Pad to nearest multiple of 16 for AVX-512 boundaries
        size_t paddedCount = (count + 15) & ~15;
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        __m512 sX = _mm512_set1_ps(stepX);
        __m512 sY = _mm512_set1_ps(stepY);
        __m512 sZ = _mm512_set1_ps(stepZ);
        __m512 smallVal = _mm512_set1_ps(0.00001f);

        uint32_t dataCount = static_cast<uint32_t>(xs.size());

        // 1. Calculate a cache-friendly chunk size
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t targetChunksPerThread = 16;
        uint32_t CHUNK_SIZE = dataCount / (threadCount * targetChunksPerThread);
        
        // Clamp to L1/L2 friendly sizes
        CHUNK_SIZE = std::clamp(CHUNK_SIZE, 8192u, 32768u);
        
        // CRITICAL: Ensure AVX-512 16-float (64-byte) alignment
        CHUNK_SIZE = (CHUNK_SIZE + 15) & ~15;

        // 2. Dispatch to the Coroutine Work-Stealing Queue
        g_JobSystem.DispatchAndWait(dataCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Ensure boundaries are strictly aligned to prevent AVX-512 load crashes
            uint32_t alignedStart = start & ~15;
            uint32_t alignedEnd = (end + 15) & ~15;

            for(uint32_t i = alignedStart; i < alignedEnd; i += 16) {
                // ALIGNED LOAD: We are guaranteed 64-byte alignment (512-bit memory)
                // Custom Allocator: We can now safely use the faster aligned loads (_mm512_load_ps) and stores!
                SIMDVector16 batch = { 
                    _mm512_load_ps(&xs[i]), 
                    _mm512_load_ps(&ys[i]), 
                    _mm512_load_ps(&zs[i]) 
                };

                // MATH
                batch.add(sX, sY, sZ);
                __m512 d = batch.dot_fma(sX, sY, sZ);
                batch.x = _mm512_add_ps(batch.x, _mm512_mul_ps(d, smallVal));
                batch.cross(sX, sY, sZ);

                // ALIGNED STORE: Direct write-back
                _mm512_store_ps(&xs[i], batch.x);
                _mm512_store_ps(&ys[i], batch.y);
                _mm512_store_ps(&zs[i], batch.z);
            }

            // Clean the CPU registers before handing the thread back to the scheduler
            _mm256_zeroupper();
        });
        // Clean the Caller Thread before returning
        _mm256_zeroupper();
    }
};
