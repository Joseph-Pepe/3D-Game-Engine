#pragma once

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
