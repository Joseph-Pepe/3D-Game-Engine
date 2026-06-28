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

// ==================================================================================
// AVX2 BARE-METAL SIMD STRUCTURES (8-Wide)
// ==================================================================================

// --- THE MATH LAYER (Intel-specific Intrinsics SIMD) ---
// AVX-256: This represents 8 vectors. It doesn't own memory; it just processes it (used for bulk data processing).
// This is a custom SIMD wrapper that bypasses standard C++ compilers to explicitly command the CPU's execution ports.
// SIMD Intrinsics: Cannot be evaluated at compile-time, only run-time.
struct SIMDVector8 {
    // --- AVX-256: 8-Wide Batch ---
    __m256 x, y, z;

    // Standard Addition
    FORCE_INLINE void add(const __m256& bx, const __m256& by, const __m256& bz) {
        x = _mm256_add_ps(x, bx);
        y = _mm256_add_ps(y, by);
        z = _mm256_add_ps(z, bz);
    }

    // Subtraction: A - B
    FORCE_INLINE void sub(const __m256& bx, const __m256& by, const __m256& bz) {
        x = _mm256_sub_ps(x, bx);
        y = _mm256_sub_ps(y, by);
        z = _mm256_sub_ps(z, bz);
    }

    // Scalar Multiplication (Vector * Float)
    FORCE_INLINE void mul(const __m256& scalar) {
        x = _mm256_mul_ps(x, scalar);
        y = _mm256_mul_ps(y, scalar);
        z = _mm256_mul_ps(z, scalar);
    }

    // SOA Dot Product with FMA (if supported) to perform multiplication and addition in a single clock cycle (i.e., shrinks instruction footprint).
    FORCE_INLINE __m256 dot_fma(const __m256& bx, const __m256& by, const __m256& bz) const {
        __m256 res = _mm256_mul_ps(x, bx); // res = x*bx
        res = _mm256_fmadd_ps(y, by, res); // res = (y*by) + res
        res = _mm256_fmadd_ps(z, bz, res); // res = (z*bz) + res
        return res;
    }

    // SOA Dot Product
    FORCE_INLINE __m256 dot(const __m256& bx, const __m256& by, const __m256& bz) const {
        __m256 mx = _mm256_mul_ps(x, bx);
        __m256 my = _mm256_mul_ps(y, by);
        __m256 mz = _mm256_mul_ps(z, bz);
        return _mm256_add_ps(_mm256_add_ps(mx, my), mz);
    }
    

    // SOA Cross Product
    FORCE_INLINE void cross(const __m256& bx, const __m256& by, const __m256& bz) {
        __m256 rx = _mm256_sub_ps(_mm256_mul_ps(y, bz), _mm256_mul_ps(z, by));
        __m256 ry = _mm256_sub_ps(_mm256_mul_ps(z, bx), _mm256_mul_ps(x, bz));
        __m256 rz = _mm256_sub_ps(_mm256_mul_ps(x, by), _mm256_mul_ps(y, bx));
        x = rx; y = ry; z = rz;
    }

    // Magnitude Squared (Length Squared) is faster than magnitude because it skips the square root (sqrt_ps).
    FORCE_INLINE __m256 length_sq() const {
        __m256 xx = _mm256_mul_ps(x, x);
        __m256 yy = _mm256_mul_ps(y, y);
        __m256 zz = _mm256_mul_ps(z, z);
        return _mm256_add_ps(_mm256_add_ps(xx, yy), zz);
    }

    // Magnitude (Length)
    FORCE_INLINE __m256 length() const {
        return _mm256_sqrt_ps(length_sq());
    }

    // Normalization (Set length to 1.0)
    FORCE_INLINE void normalize() {
        __m256 len = length();
        // Use a small epsilon to prevent division by zero
        __m256 epsilon = _mm256_set1_ps(1e-8f);
        __m256 mask = _mm256_cmp_ps(len, epsilon, _CMP_GT_OQ); 
        
        // rcpps (Reciprocal) is even faster, but divps is more precise.
        __m256 invLen = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
        
        // Only multiply if length > epsilon, otherwise set to 0
        invLen = _mm256_and_ps(invLen, mask);

        x = _mm256_mul_ps(x, invLen);
        y = _mm256_mul_ps(y, invLen);
        z = _mm256_mul_ps(z, invLen);
    }
};
