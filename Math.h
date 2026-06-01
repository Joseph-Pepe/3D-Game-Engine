#pragma once

#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).
#include <cmath>       // Trigonometry
#include <print>       // Formatting
#include <cstdint>

// --- COMPILER MACROS ---
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

// ==================================================================================
// 1. FAST MATH UTILITIES
// ==================================================================================

// --- AVX2 FAST HORIZONTAL SUM ---
// Folds an 8-wide __m256 register down to a single scalar float inside the silicon registers without touching memory.
FORCE_INLINE float hsum_avx2(__m256 v) {
    // 1. Split the 256-bit register into two 128-bit halves and add them
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);

    // 2. Collapse the remaining 4 floats down to 1 (Standard SSE shuffle reduction)
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 1, 1, 1));
    lo = _mm_add_ps(lo, hi);

    // 3. Extract the final single float
    return _mm_cvtss_f32(lo);
}

// ===============================================
// CRYPTOGRAPHIC RANDOM NUMBER GENERATOR & MAPPING
// ===============================================

// --- FAST STATELESS PRNG ---
// Executes in ~1-2 clock cycles entirely inside the ALU registers.
FORCE_INLINE uint32_t XorShift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// --- DIVISION-LESS RANGE MAPPING ---
// Maps a random 32-bit integer to [0, range) using multiplication instead of division.
FORCE_INLINE uint32_t MapToRange(uint32_t randomVal, uint32_t range) {
    return static_cast<uint32_t>((static_cast<uint64_t>(randomVal) * static_cast<uint64_t>(range)) >> 32);
}

// ==================================================================================
// 2. AVX2 BARE-METAL SIMD STRUCTURES (8-Wide)
// ==================================================================================

// --- THE MATH LAYER (Intel-specific Intrinsics SIMD) ---
// AVX-256: This represents 8 vectors. It doesn't own memory; it just processes it.
// This is a custom SIMD wrapper that bypasses standard C++ compilers to explicitly command the CPU's execution ports.
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

// ==================================================================================
// 3. SSE / SCALAR VECTORS
// ==================================================================================

// SSE Accelerated Stack Vector (Use for 99% of general game logic).
class Vector3DStack {
public:
    // This tells the compiler: "Every instance of this class must start at a 16-byte boundary in memory."
    // Forces every instance of this class to align to 16 bytes.
    // Memory is allocated on the stack automatically!
    alignas(16) float data[4]; 

    // Constructor: Default initializes to {0, 0, 0, 0}
    Vector3DStack(float x = 0.0f, float y = 0.0f, float z = 0.0f) : data{x, y, z, 0.0f} {}
    
    // ---------------------------------------------------------
    // No Destructor, Copy Constructor, or Assignment Operator 
    // are needed! The compiler handles the 16 bytes trivially.
    // ---------------------------------------------------------

    // Addition: C = A + B
    FORCE_INLINE Vector3DStack operator+(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);
        __m128 res = _mm_add_ps(v1, v2);
        
        _mm_store_ps(result.data, res);
        return result;
    }

    // Subtraction: C = A - B
    FORCE_INLINE Vector3DStack operator-(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);
        __m128 res = _mm_sub_ps(v1, v2);
        
        _mm_store_ps(result.data, res);
        return result;
    }

    // Scalar Multiplication: B = A * scalar
    FORCE_INLINE Vector3DStack operator*(float scalar) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar); // Broadcasts scalar to all 4 slots
        __m128 res = _mm_mul_ps(v1, s);
        
        _mm_store_ps(result.data, res);
        return result;
    }

    // In-place Scalar Multiplication: A *= scalar
    FORCE_INLINE void operator*=(float scalar) {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar);
        
        __m128 res = _mm_mul_ps(v1, s);
        _mm_store_ps(this->data, res); // Store directly back into itself
    }

    // Dot Product
    FORCE_INLINE float dot(const Vector3DStack& other) const {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);

        // 0x71 mask: calculates dots for first 3 elements, stores in first element
        __m128 res = _mm_dp_ps(v1, v2, 0x71);

        float result;
        _mm_store_ss(&result, res); 
        return result;
    }

    // Cross Product
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

    // Utility to easily print the vector
    void print() const {
        std::println("[{}, {}, {}, {}]", data[0], data[1], data[2], data[3]);
    }
};

// Pure Scalar Fallback
class Vector3DScalar {
public:
    float x, y, z, w; // Same 16-byte memory footprint for fairness

    Vector3DScalar(float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : x(x), y(y), z(z), w(0.0f) {}

    // Traditional element-by-element addition
    FORCE_INLINE Vector3DScalar operator+(const Vector3DScalar& other) const {
        return Vector3DScalar(x + other.x, y + other.y, z + other.z);
    }

    // Traditional dot product
    FORCE_INLINE float dot(const Vector3DScalar& other) const {
        return (x * other.x) + (y * other.y) + (z * other.z);
    }

    // Traditional cross product
    FORCE_INLINE Vector3DScalar cross(const Vector3DScalar& other) const {
        return Vector3DScalar(
            (y * other.z) - (z * other.y),
            (z * other.x) - (x * other.z),
            (x * other.y) - (y * other.x)
        );
    }
};

// ==================================================================================
// 4. MATRICES & INTERPOLATION
// ==================================================================================

// --- 3D CAMERA & MATRIX MATH ---
// --- 4x4 MATRIX MATH (Stack Allocated, Column-Major for OpenGL) ---
struct Matrix4 {
    float m[16] = {0}; // Initializes to all zeros

    // Creates an Identity Matrix
    static Matrix4 Identity() {
        Matrix4 mat;
        mat.m[0] = 1.0f; mat.m[5] = 1.0f; mat.m[10] = 1.0f; mat.m[15] = 1.0f;
        return mat;
    }

    // Creates a Perspective Projection Matrix
    static Matrix4 Perspective(float fovY_degrees, float aspect, float nearZ, float farZ) {
        Matrix4 mat;
        float fovY_rad = fovY_degrees * (3.14159265359f / 180.0f);
        float tanHalfFovY = std::tan(fovY_rad / 2.0f);

        mat.m[0] = 1.0f / (aspect * tanHalfFovY);
        mat.m[5] = 1.0f / tanHalfFovY;
        mat.m[10] = -(farZ + nearZ) / (farZ - nearZ);
        mat.m[11] = -1.0f;
        mat.m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
        return mat;
    }

    // Creates a View Matrix (LookAt)
    static Matrix4 LookAt(const Vector3DStack& eye, const Vector3DStack& target, const Vector3DStack& upVec) {

        // 1. Forward Vector (Z)
        Vector3DStack f = target - eye;
        float fLen = std::sqrt(f.dot(f));
        f = f * (1.0f / fLen);

        // 2. Right Vector (X)
        Vector3DStack r = f.cross(upVec);
        float rLen = std::sqrt(r.dot(r));
        r = r * (1.0f / rLen);

        // 3. Up Vector (Y)
        Vector3DStack u = r.cross(f);

        // 4. Build Column-Major Matrix
        Matrix4 mat = Identity();
        mat.m[0] = r.data[0];  mat.m[4] = r.data[1];  mat.m[8] = r.data[2];
        mat.m[1] = u.data[0];  mat.m[5] = u.data[1];  mat.m[9] = u.data[2];
        mat.m[2] = -f.data[0]; mat.m[6] = -f.data[1]; mat.m[10] = -f.data[2];

        // Translation offsets
        mat.m[12] = -r.dot(eye);
        mat.m[13] = -u.dot(eye);
        mat.m[14] = f.dot(eye);
        return mat;
    }
};

// --- MATH UTILITIES FOR CAMERA PATHING (SPLINES & LINEAR ALGEBRA) ---
FORCE_INLINE Vector3DStack Lerp(const Vector3DStack& a, const Vector3DStack& b, float t) {
    // V = A + t * (B - A)
    return a + ((b - a) * t);
}

FORCE_INLINE Vector3DStack CatmullRom(const Vector3DStack& p0, const Vector3DStack& p1, 
                                      const Vector3DStack& p2, const Vector3DStack& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    // Evaluates in a few CPU cycles using standard ALUs
    Vector3DStack v0 = p1 * 2.0f;
    Vector3DStack v1 = (p2 - p0) * t;
    Vector3DStack v2 = (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2;
    Vector3DStack v3 = (p1 * 3.0f - p0 - p2 * 3.0f + p3) * t3;

    return (v0 + v1 + v2 + v3) * 0.5f;
}
