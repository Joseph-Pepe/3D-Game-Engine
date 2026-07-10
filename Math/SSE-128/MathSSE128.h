#pragma once

#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).
#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <print>       // Formatting
#include <cstdint>
#include <array>
#include <cstddef>
#include <algorithm>   // Required for std::min, std::copy, std::swap
#include <bit>         // Required for std::bit_cast

// --- COMPILER MACROS ---
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

class Vector3DStack {
public:
    alignas(16) float data[4]; 

    // Constructor: Default initializes to {0, 0, 0, 0}
    Vector3DStack(float x = 0.0f, float y = 0.0f, float z = 0.0f) : data{x, y, z, 0.0f} {}

    // Cross Product (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack cross(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 a = _mm_load_ps(this->data);
        __m128 b = _mm_load_ps(other.data);

        __m128 tmp0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 tmp1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp2 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp3 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));

        __m128 res = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
        _mm_store_ps(result.data, res);
        
        return result;
    }
};

// ==================================================================================
// SIMD QUATERNION (128-bit)
// ==================================================================================
struct alignas(16) Quaternion {
    __m128 reg;
};

// ==================================================================================
// SIMD 4x4 MATRIX (COLUMN-MAJOR)
// ==================================================================================

struct alignas(64) SIMD_SSE128_Matrix4x4 {
    __m128 col[4];
};
