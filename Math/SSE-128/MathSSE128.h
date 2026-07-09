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

// Storing data in an array of floats inside a struct that is meant to be used for SIMD is an anti-pattern (i.e., Load-Hit-Store penalty).
class Vector3DStack {
public:
    alignas(16) float data[4]; 

    // Constructor: Default initializes to {0, 0, 0, 0}
    Vector3DStack(float x = 0.0f, float y = 0.0f, float z = 0.0f) : data{x, y, z, 0.0f} {}

    // Addition: C = A + B (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack operator+(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);
        _mm_store_ps(result.data, _mm_add_ps(v1, v2));

        return result;
    }

    // Subtraction: C = A - B (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack operator-(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);
        _mm_store_ps(result.data, _mm_sub_ps(v1, v2));

        return result;
    }

    // Scalar Multiplication: B = A * scalar (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack operator*(float scalar) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar); 
        _mm_store_ps(result.data, _mm_mul_ps(v1, s));

        return result;
    }

    // In-place Scalar Multiplication: A *= scalar
    FORCE_INLINE void operator*=(float scalar) {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar);
        _mm_store_ps(this->data, _mm_mul_ps(v1, s)); // Store directly back into itself
    }

    // Dot Product (Load-Hit-Store penalty)
    FORCE_INLINE float dot(const Vector3DStack& other) const {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);

        __m128 mul = _mm_mul_ps(v1, v2);

        // Since constructor guarantees w = 0.0f, we can safely horizontal sum all 4 lanes
        __m128 shuf = _mm_movehl_ps(mul, mul); 
        mul = _mm_add_ps(mul, shuf);           
        shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)); 
        mul = _mm_add_ss(mul, shuf);           
        
        return _mm_cvtss_f32(mul);
    }

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

    // (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack asDirection() const {
        Vector3DStack result;
        __m128 reg = _mm_load_ps(this->data);
        
        reg = _mm_blend_ps(reg, _mm_setzero_ps(), 0x08);
        
        _mm_store_ps(result.data, reg);
        return result;
    }

    // (Load-Hit-Store penalty)
    FORCE_INLINE Vector3DStack asPoint() const {
        Vector3DStack result;
        __m128 reg = _mm_load_ps(this->data);
        
        
        __m128 wOne = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); 
        reg = _mm_blend_ps(reg, wOne, 0x08);
        
        _mm_store_ps(result.data, reg);
        return result;
    }

    // Utility to easily print the vector
    void print() const {
        std::println("[{}, {}, {}, {}]", data[0], data[1], data[2], data[3]);
    }
};

// ==================================================================================
// SIMD QUATERNION (128-bit)
// ==================================================================================
struct alignas(16) Quaternion {
    __m128 reg;

    FORCE_INLINE Quaternion() : reg(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f)) {} 
    FORCE_INLINE Quaternion(__m128 m) : reg(m) {}
    FORCE_INLINE Quaternion(float _x, float _y, float _z, float _w) : reg(_mm_set_ps(_w, _z, _y, _x)) {}

    // --- HARDWARE GETTERS ---
    FORCE_INLINE float x() const { return _mm_cvtss_f32(reg); }
    FORCE_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(1, 1, 1, 1))); }
    FORCE_INLINE float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(2, 2, 2, 2))); }
    FORCE_INLINE float w() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3))); }

    FORCE_INLINE SSE128_SIMDVector3D GetForwardVector() const {
        return RotateVector(SSE128_SIMDVector3D(0.0f, 0.0f, -1.0f, 0.0f));
    }

    FORCE_INLINE SSE128_SIMDVector3D GetRightVector() const {
        return RotateVector(SSE128_SIMDVector3D(1.0f, 0.0f, 0.0f, 0.0f));
    }

    static FORCE_INLINE Quaternion AngleAxis(float angleDegrees, const SSE128_SIMDVector3D& axis) {
        float halfAngleRad = (angleDegrees * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        float s = std::sin(halfAngleRad);
        float c = std::cos(halfAngleRad);
        
        
        __m128 sinVec = _mm_set1_ps(s);
        __m128 axisScaled = _mm_mul_ps(axis.reg, sinVec);
        
        
        __m128 wCos = _mm_set_ps(c, 0.0f, 0.0f, 0.0f);
        return Quaternion(_mm_blend_ps(axisScaled, wCos, 0x08));
    }

    FORCE_INLINE Quaternion operator*(const Quaternion& rhs) const {
        
        __m128 q1 = reg;
        __m128 q2 = rhs.reg;

        
        __m128 w1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 x1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(2, 2, 2, 2));

        
        __m128 tmp0 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 0)); // w, z, y, x
        __m128 tmp1 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(2, 3, 0, 1)); // z, w, x, y
        __m128 tmp2 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z

        
        __m128 res = _mm_mul_ps(w1, q2);
        
        
        __m128 signX = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0, 0x80000000));
        __m128 signY = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0x80000000));
        __m128 signZ = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0));

        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(x1, tmp0), signX));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(y1, tmp1), signY));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(z1, tmp2), signZ));

        return Quaternion(res);
    }

    
    FORCE_INLINE void Normalize() {
        __m128 dot = _mm_dp_ps(reg, reg, 0xFF);
        __m128 invLen = _mm_rsqrt_ps(dot); 
        reg = _mm_mul_ps(reg, invLen);
    }

    
    FORCE_INLINE Quaternion Conjugate() const {
        __m128 signMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000));
        return Quaternion(_mm_xor_ps(reg, signMask));
    }
    
    
    FORCE_INLINE SSE128_SIMDVector3D RotateVector(const SSE128_SIMDVector3D& v) const {
        SSE128_SIMDVector3D qVec(_mm_blend_ps(reg, _mm_setzero_ps(), 0x08));
        __m128 wReg = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3));
        SSE128_SIMDVector3D t = qVec.cross(v) * 2.0f;
        SSE128_SIMDVector3D tw(_mm_mul_ps(t.reg, wReg)); 
        return v + tw + qVec.cross(t);
    }

    
    // Converts a normalized forward vector into a rotation without using Trigonometry.
    static FORCE_INLINE Quaternion FromDirection(const SSE128_SIMDVector3D& dir) {
        SSE128_SIMDVector3D baseForward(0.0f, 0.0f, -1.0f, 0.0f); 
        float dot = baseForward.dot(dir);
        
        if (dot < -0.9999f) {
            return Quaternion(0.0f, 1.0f, 0.0f, 0.0f); 
        }
        
        SSE128_SIMDVector3D axis = baseForward.cross(dir);
        Quaternion q(axis.x(), axis.y(), axis.z(), 1.0f + dot);
        q.Normalize();
        
        return q;
    }

    FORCE_INLINE float dot(const Quaternion& other) const {
        return _mm_cvtss_f32(_mm_dp_ps(reg, other.reg, 0xFF));
    }

    // --- SPHERICAL LINEAR INTERPOLATION (SLERP) ---
    static FORCE_INLINE Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        float cosOmega = q1.dot(q2);
        __m128 q2Reg = q2.reg;

        
        if (cosOmega < 0.0f) {
            cosOmega = -cosOmega;
            q2Reg = _mm_xor_ps(q2Reg, _mm_set1_ps(-0.0f));
        }

        
        if (cosOmega > 0.9999f) {
            __m128 tReg = _mm_set1_ps(t);
            __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tReg);
            
            Quaternion result(_mm_add_ps(_mm_mul_ps(q1.reg, oneMinusT), _mm_mul_ps(q2Reg, tReg)));
            result.Normalize();
            return result;
        }
        
        float omega = std::acos(cosOmega);
        float invSinOmega = 1.0f / std::sin(omega);

        float weight0 = std::sin((1.0f - t) * omega) * invSinOmega;
        float weight1 = std::sin(t * omega) * invSinOmega;

        __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, _mm_set1_ps(weight0)), _mm_mul_ps(q2Reg, _mm_set1_ps(weight1)));

        return Quaternion(res);
    }
};

// ==================================================================================
// SIMD 4x4 MATRIX (COLUMN-MAJOR)
// ==================================================================================

struct alignas(64) SIMD_SSE128_Matrix4x4 {
    __m128 col[4];

    static FORCE_INLINE SIMD_SSE128_Matrix4x4 Identity() {
        SIMD_SSE128_Matrix4x4 mat;
        mat.col[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f); 
        mat.col[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f); 
        mat.col[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f); 
        mat.col[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); 
        return mat;
    }
};

// --- SIMD MATRIX OPERATORS ---
FORCE_INLINE SSE128_SIMDVector3D operator*(const SIMD_SSE128_Matrix4x4& mat, const SSE128_SIMDVector3D& v) {
    __m128 vx = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(0, 0, 0, 0)); 
    __m128 vy = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(1, 1, 1, 1)); 
    __m128 vz = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(2, 2, 2, 2)); 
    __m128 vw = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(3, 3, 3, 3)); 
    
    __m128 res = _mm_mul_ps(vx, mat.col[0]);
    
    res = _mm_fmadd_ps(vy, mat.col[1], res);  
    res = _mm_fmadd_ps(vz, mat.col[2], res);
    res = _mm_fmadd_ps(vw, mat.col[3], res);

    return SSE128_SIMDVector3D(res);
}

// --- SIMD MATRIX MULTIPLICATION (C = A * B) ---
FORCE_INLINE SIMD_SSE128_Matrix4x4 operator*(const SIMD_SSE128_Matrix4x4& a, const SIMD_SSE128_Matrix4x4& b) {
    SIMD_SSE128_Matrix4x4 res;
    
    for (int i = 0; i < 4; ++i) {
        __m128 vx = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(0, 0, 0, 0));
        __m128 vy = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(1, 1, 1, 1));
        __m128 vz = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(2, 2, 2, 2));
        __m128 vw = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(3, 3, 3, 3));

        __m128 col = _mm_mul_ps(vx, a.col[0]);
        col = _mm_fmadd_ps(vy, a.col[1], col);
        col = _mm_fmadd_ps(vz, a.col[2], col);
        col = _mm_fmadd_ps(vw, a.col[3], col);
        
        res.col[i] = col;
    }
    return res;
}
