#pragma once

#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <cstdint>
#include <array>
#include <mdspan>
#include <cstddef>
#include <span>
#include <algorithm>   // Required for std::min, std::copy, std::swap

// --- COMPILER MACROS ---
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

// C++26 linear algebra (compiler dependent availability)
#if __has_include(<linear_algebra>)
    #include <linear_algebra>
#endif

// ==================================================================================
// 1. CROSS-PLATFORM HARDWARE ABSTRACTION LAYER (128-bit AoS)
// ==================================================================================
/*
    This layer completely hides the CPU vendor. 
    It maps mathematically identical operations to Intel SSE/AVX and ARM NEON.
*/
#if defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #define MATH_ISA_ARM       // Apple Silicon (M-series), Mobile (iPhone, Android) (ARM64)
#else
    #include <immintrin.h>
    #define MATH_ISA_X86       // Intel/AMD (PS5, Xbox, PC) (x86_x64)
#endif

namespace Engine::Math::SIMD {

    #ifdef MATH_ISA_ARM
        // --- ARM APPLE SILICON / MOBILE (NEON) ---
        using Float4 = float32x4_t;

        FORCE_INLINE Float4 Zero() { return vdupq_n_f32(0.0f); }
        FORCE_INLINE Float4 Set(float x, float y, float z, float w) { 
            float arr[4] = {x, y, z, w}; 
            return vld1q_f32(arr); 
        }
        FORCE_INLINE Float4 Set1(float v) { return vdupq_n_f32(v); }
        
        FORCE_INLINE Float4 Add(Float4 a, Float4 b) { return vaddq_f32(a, b); }
        FORCE_INLINE Float4 Sub(Float4 a, Float4 b) { return vsubq_f32(a, b); }
        FORCE_INLINE Float4 Mul(Float4 a, Float4 b) { return vmulq_f32(a, b); }
        FORCE_INLINE Float4 FMAdd(Float4 a, Float4 b, Float4 c) { return vfmaq_f32(c, a, b); } // c + (a * b)

        // Blend: Replace W in 'a' with W from 'b'
        FORCE_INLINE Float4 BlendMaskW(Float4 a, Float4 b) { return vsetq_lane_f32(vgetq_lane_f32(b, 3), a, 3); }
        
        FORCE_INLINE float ExtractX(Float4 v) { return vgetq_lane_f32(v, 0); }
        FORCE_INLINE float ExtractY(Float4 v) { return vgetq_lane_f32(v, 1); }
        FORCE_INLINE float ExtractZ(Float4 v) { return vgetq_lane_f32(v, 2); }
        FORCE_INLINE float ExtractW(Float4 v) { return vgetq_lane_f32(v, 3); }

        FORCE_INLINE float Dot4(Float4 a, Float4 b) { return vaddvq_f32(vmulq_f32(a, b)); }
        FORCE_INLINE float Dot3(Float4 a, Float4 b) { 
            Float4 a3 = vsetq_lane_f32(0.0f, a, 3);
            Float4 b3 = vsetq_lane_f32(0.0f, b, 3);
            return vaddvq_f32(vmulq_f32(a3, b3)); 
        }

        FORCE_INLINE Float4 Cross(Float4 a, Float4 b) {
            #if defined(__GNUC__) || defined(__clang__)
                // Clang perfectly maps vector shuffles directly to NEON registers
                Float4 a_yzx = __builtin_shufflevector(a, a, 1, 2, 0, 3);
                Float4 b_zxy = __builtin_shufflevector(b, b, 2, 0, 1, 3);
                Float4 a_zxy = __builtin_shufflevector(a, a, 2, 0, 1, 3);
                Float4 b_yzx = __builtin_shufflevector(b, b, 1, 2, 0, 3);
                return Sub(Mul(a_yzx, b_zxy), Mul(a_zxy, b_yzx));
            #else
                float ax = ExtractX(a), ay = ExtractY(a), az = ExtractZ(a);
                float bx = ExtractX(b), by = ExtractY(b), bz = ExtractZ(b);
                return Set((ay*bz)-(az*by), (az*bx)-(ax*bz), (ax*by)-(ay*bx), 0.0f);
            #endif
        }

        FORCE_INLINE void Transpose4(Float4& r0, Float4& r1, Float4& r2, Float4& r3) {
            float32x4x2_t t0 = vtrnq_f32(r0, r1);
            float32x4x2_t t1 = vtrnq_f32(r2, r3);
            r0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
            r1 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
            r2 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
            r3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
        }

        FORCE_INLINE Float4 FlipSignXYZ(Float4 v) {
            uint32x4_t mask = {0x80000000, 0x80000000, 0x80000000, 0};
            return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), mask));
        }

        FORCE_INLINE Float4 QuaternionMul(Float4 q1, Float4 q2) {
            // Scalar extraction on M-Series ARM is vastly faster than complex NEON bitmask shuffles
            float x1 = ExtractX(q1), y1 = ExtractY(q1), z1 = ExtractZ(q1), w1 = ExtractW(q1);
            float x2 = ExtractX(q2), y2 = ExtractY(q2), z2 = ExtractZ(q2), w2 = ExtractW(q2);
            return Set(
                (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
                (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
                (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2)
            );
        }

        FORCE_INLINE Float4 CmpGt(Float4 a, Float4 b) { return vreinterpretq_f32_u32(vcgtq_f32(a, b)); }
        FORCE_INLINE Float4 CmpLt(Float4 a, Float4 b) { return vreinterpretq_f32_u32(vcltq_f32(a, b)); }
        FORCE_INLINE Float4 BitwiseAnd(Float4 a, Float4 b) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
        FORCE_INLINE Float4 BlendVariable(Float4 a, Float4 b, Float4 mask) { return vbslq_f32(vreinterpretq_u32_f32(mask), b, a); }
        
        // Simulates Intel's _mm_movemask_ps by checking the high bit of each float
        FORCE_INLINE int MoveMask(Float4 v) {
            uint32x4_t mask = vreinterpretq_u32_f32(v);
            uint32x2_t tmp = vshrn_n_u64(vreinterpretq_u64_u32(mask), 16);
            uint64_t result = vget_lane_u64(vreinterpret_u64_u32(tmp), 0);
            return ((result >> 15) & 1) | ((result >> 30) & 2) | ((result >> 45) & 4) | ((result >> 60) & 8);
        }
    #else
        // --- INTEL / AMD PC (SSE / AVX2) ---
        using Float4 = __m128;

        FORCE_INLINE Float4 Zero() { return _mm_setzero_ps(); }
        FORCE_INLINE Float4 Set(float x, float y, float z, float w) { return _mm_set_ps(w, z, y, x); }
        FORCE_INLINE Float4 Set1(float v) { return _mm_set1_ps(v); }
        
        FORCE_INLINE Float4 Add(Float4 a, Float4 b) { return _mm_add_ps(a, b); }
        FORCE_INLINE Float4 Sub(Float4 a, Float4 b) { return _mm_sub_ps(a, b); }
        FORCE_INLINE Float4 Mul(Float4 a, Float4 b) { return _mm_mul_ps(a, b); }
        FORCE_INLINE Float4 FMAdd(Float4 a, Float4 b, Float4 c) { return _mm_fmadd_ps(a, b, c); } // (a * b) + c

        FORCE_INLINE Float4 BlendMaskW(Float4 a, Float4 b) { return _mm_blend_ps(a, b, 0x08); }

        FORCE_INLINE float ExtractX(Float4 v) { return _mm_cvtss_f32(v); }
        FORCE_INLINE float ExtractY(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))); }
        FORCE_INLINE float ExtractZ(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))); }
        FORCE_INLINE float ExtractW(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3))); }

        FORCE_INLINE float Dot4(Float4 a, Float4 b) { return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xFF)); }
        FORCE_INLINE float Dot3(Float4 a, Float4 b) { return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x7F)); }

        FORCE_INLINE Float4 Cross(Float4 a, Float4 b) {
            __m128 tmp0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 tmp1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 tmp2 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 tmp3 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
            return _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
        }

        FORCE_INLINE void Transpose4(Float4& r0, Float4& r1, Float4& r2, Float4& r3) {
            // _MM_TRANSPOSE4_PS(r0, r1, r2, r3);

            // Guarantees it modifies the passed in references, ensures compiler tracks the mutations correctly.
            Float4 tmp0 = _mm_unpacklo_ps(r0, r1);
            Float4 tmp2 = _mm_unpacklo_ps(r2, r3);
            Float4 tmp1 = _mm_unpackhi_ps(r0, r1);
            Float4 tmp3 = _mm_unpackhi_ps(r2, r3);

            r0 = _mm_movelh_ps(_mm_unpacklo_ps(tmp0, tmp2), tmp0);
            r1 = _mm_movehl_ps(_mm_unpacklo_ps(tmp0, tmp2), tmp0);
            r2 = _mm_movelh_ps(_mm_unpackhi_ps(tmp1, tmp3), tmp1);
            r3 = _mm_movehl_ps(_mm_unpackhi_ps(tmp1, tmp3), tmp1);
        }

        FORCE_INLINE Float4 FlipSignXYZ(Float4 v) {
            __m128 signMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000));
            return _mm_xor_ps(v, signMask);
        }

        FORCE_INLINE Float4 QuaternionMul(Float4 q1, Float4 q2) {
            __m128 w1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 3, 3, 3));
            __m128 x1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 y1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(2, 2, 2, 2));

            __m128 tmp0 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 0));
            __m128 tmp1 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 tmp2 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(1, 0, 3, 2));

            __m128 res = _mm_mul_ps(w1, q2);
            
            __m128 signX = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0, 0x80000000));
            __m128 signY = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0x80000000));
            __m128 signZ = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0));

            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(x1, tmp0), signX));
            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(y1, tmp1), signY));
            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(z1, tmp2), signZ));

            return res;
        }

        FORCE_INLINE Float4 CmpGt(Float4 a, Float4 b) { return _mm_cmpgt_ps(a, b); }
        FORCE_INLINE Float4 CmpLt(Float4 a, Float4 b) { return _mm_cmplt_ps(a, b); }
        FORCE_INLINE Float4 BitwiseAnd(Float4 a, Float4 b) { return _mm_and_ps(a, b); }
        FORCE_INLINE Float4 BlendVariable(Float4 a, Float4 b, Float4 mask) { return _mm_blendv_ps(a, b, mask); }
        FORCE_INLINE int MoveMask(Float4 v) { return _mm_movemask_ps(v); }
    #endif
} // namespace Engine::Math::SIMD

// ==================================================================================
// PURE SCALAR VECTORS (NON-SIMD)
// ==================================================================================
/*
    - Used for one-off calculations like the camera and compile time calculations.
    - SIMD Vectors must perform math operations on all its available lanes (4, 8, 16) simultaneously in hardware registers.
    - Pure Scalar Vectors execute math operations in one lane.
*/
class Vector3D {
public:
    // 16-bytes 
    float x, y, z, w;

    constexpr Vector3D(float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : x(x), y(y), z(z), w(0.0f) {}

    // Addition: C = A + B
    FORCE_INLINE constexpr Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction: C = A - B
    FORCE_INLINE constexpr Vector3D operator-(const Vector3D& other) const {
        return Vector3D(x - other.x, y - other.y, z - other.z);
    }

    // Scalar Multiplication: B = A * scalar
    FORCE_INLINE constexpr Vector3D operator*(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    // In-Place Scalar Multiplication: A *= scalar
    FORCE_INLINE constexpr void operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    // Dot product
    FORCE_INLINE constexpr float dot(const Vector3D& other) const {
        return (x * other.x) + (y * other.y) + (z * other.z);
    }

    // Cross product
    FORCE_INLINE constexpr Vector3D cross(const Vector3D& other) const {
        return Vector3D(
            (y * other.z) - (z * other.y),
            (z * other.x) - (x * other.z),
            (x * other.y) - (y * other.x)
        );
    }

    // --- MAGNITUDE & NORMALIZATION ---
    FORCE_INLINE float lengthSquared() const {
        return dot(*this);
    }

    FORCE_INLINE float length() const {
        return std::sqrt(lengthSquared());
    }

    FORCE_INLINE void Normalize() {
        float lenSq = lengthSquared();
        if (lenSq > 1e-8f) {
            *this = *this * (1.0f / std::sqrt(lenSq));
        }
    }

    FORCE_INLINE auto GetNormalized() const {
        auto copy = *this;
        copy.Normalize();
        return copy;
    }

    // Utility to easily print the vector
    void print() const {
        std::println("<{}, {}, {}, {}>", x, y, z, w);
    }
};

// ==================================================================================
// 128-BIT ACCELERATED VECTORS (CROSS-PLATFORM)
// ==================================================================================
class alignas(16) SIMDVector3D {
public:
    Engine::Math::SIMD::Float4 reg;

    FORCE_INLINE SIMDVector3D() : reg(Engine::Math::SIMD::Zero()) {}
    FORCE_INLINE SIMDVector3D(float _x, float _y, float _z, float _w = 0.0f) 
        : reg(Engine::Math::SIMD::Set(_x, _y, _z, _w)) {}
    FORCE_INLINE SIMDVector3D(Engine::Math::SIMD::Float4 m) : reg(m) {}

    // --- HARDWARE GETTERS ---
    FORCE_INLINE float x() const { return Engine::Math::SIMD::ExtractX(reg); }
    FORCE_INLINE float y() const { return Engine::Math::SIMD::ExtractY(reg); }
    FORCE_INLINE float z() const { return Engine::Math::SIMD::ExtractZ(reg); }
    FORCE_INLINE float w() const { return Engine::Math::SIMD::ExtractW(reg); }

    // --- C++20 BRIDGE ---
    FORCE_INLINE std::array<float, 4> asArray() const {
        return std::bit_cast<std::array<float, 4>>(reg);
    }

    FORCE_INLINE float operator[](int axis) const {
        auto arr = std::bit_cast<std::array<float, 4>>(reg);
        return arr[axis];
    }

    // --- MATH ---
    static FORCE_INLINE SIMDVector3D Lerp(const SIMDVector3D& a, const SIMDVector3D& b, float t) {
        Engine::Math::SIMD::Float4 tReg = Engine::Math::SIMD::Set1(t);
        Engine::Math::SIMD::Float4 diff = Engine::Math::SIMD::Sub(b.reg, a.reg);
        return SIMDVector3D(Engine::Math::SIMD::FMAdd(diff, tReg, a.reg));
    }

    FORCE_INLINE SIMDVector3D operator+(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Add(reg, other.reg)); }
    FORCE_INLINE SIMDVector3D operator-(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Sub(reg, other.reg)); }
    FORCE_INLINE SIMDVector3D operator*(float scalar) const { return SIMDVector3D(Engine::Math::SIMD::Mul(reg, Engine::Math::SIMD::Set1(scalar))); }
    
    FORCE_INLINE float dot(const SIMDVector3D& other) const { return Engine::Math::SIMD::Dot3(reg, other.reg); }
    FORCE_INLINE SIMDVector3D cross(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Cross(reg, other.reg)); }

    FORCE_INLINE SIMDVector3D asDirection() const { return SIMDVector3D(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::SIMD::Zero())); }
    FORCE_INLINE SIMDVector3D asPoint() const { return SIMDVector3D(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f))); }

    // --- MAGNITUDE & NORMALIZATION ---
    FORCE_INLINE float lengthSquared() const {
        return dot(*this);
    }

    FORCE_INLINE float length() const {
        return std::sqrt(lengthSquared());
    }

    FORCE_INLINE void Normalize() {
        float lenSq = lengthSquared();
        if (lenSq > 1e-8f) {
            *this = *this * (1.0f / std::sqrt(lenSq));
        }
    }

    FORCE_INLINE auto GetNormalized() const {
        auto copy = *this;
        copy.Normalize();
        return copy;
    }

    void print() const { std::println("[{}, {}, {}, {}]", x(), y(), z(), w()); }
};

// ==================================================================================
// SIMD QUATERNION (CROSS-PLATFORM)
// ==================================================================================
struct alignas(16) SIMDQuaternion {
    Engine::Math::SIMD::Float4 reg;

    FORCE_INLINE SIMDQuaternion() : reg(Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f)) {}
    FORCE_INLINE SIMDQuaternion(Engine::Math::SIMD::Float4 m) : reg(m) {}
    FORCE_INLINE SIMDQuaternion(float _x, float _y, float _z, float _w) : reg(Engine::Math::SIMD::Set(_x, _y, _z, _w)) {}

    FORCE_INLINE float x() const { return Engine::Math::SIMD::ExtractX(reg); }
    FORCE_INLINE float y() const { return Engine::Math::SIMD::ExtractY(reg); }
    FORCE_INLINE float z() const { return Engine::Math::SIMD::ExtractZ(reg); }
    FORCE_INLINE float w() const { return Engine::Math::SIMD::ExtractW(reg); }

    FORCE_INLINE SIMDVector3D GetForwardVector() const { return RotateVector(SIMDVector3D(0.0f, 0.0f, -1.0f, 0.0f)); }
    FORCE_INLINE SIMDVector3D GetRightVector() const { return RotateVector(SIMDVector3D(1.0f, 0.0f, 0.0f, 0.0f)); }
    FORCE_INLINE SIMDVector3D GetUpVector() const { return RotateVector(SIMDVector3D(0.0f, 1.0f, 0.0f, 0.0f)); }

    static FORCE_INLINE SIMDQuaternion AngleAxis(float angleDegrees, const SIMDVector3D& axis) {
        float halfAngleRad = (angleDegrees * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        
        Engine::Math::SIMD::Float4 sinVec = Engine::Math::SIMD::Set1(std::sin(halfAngleRad));
        Engine::Math::SIMD::Float4 axisScaled = Engine::Math::SIMD::Mul(axis.reg, sinVec);
        Engine::Math::SIMD::Float4 wCos = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, std::cos(halfAngleRad));
        
        return SIMDQuaternion(Engine::Math::SIMD::BlendMaskW(axisScaled, wCos));
    }

    FORCE_INLINE SIMDQuaternion operator*(const SIMDQuaternion& rhs) const {
        return SIMDQuaternion(Engine::Math::SIMD::QuaternionMul(reg, rhs.reg));
    }

    FORCE_INLINE void Normalize() {
        float dot = Engine::Math::SIMD::Dot4(reg, reg);
        float invLen = 1.0f / std::sqrt(dot); // Replaced _mm_rsqrt_ps to ensure exact ARM/x86 matching
        reg = Engine::Math::SIMD::Mul(reg, Engine::Math::SIMD::Set1(invLen));
    }

    FORCE_INLINE SIMDQuaternion Conjugate() const {
        return SIMDQuaternion(Engine::Math::SIMD::FlipSignXYZ(reg));
    }
    
    FORCE_INLINE SIMDVector3D RotateVector(const SIMDVector3D& v) const {
        SIMDVector3D qVec(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::SIMD::Zero()));
        Engine::Math::SIMD::Float4 wReg = Engine::Math::SIMD::Set1(w());
        
        SIMDVector3D t = qVec.cross(v) * 2.0f;
        SIMDVector3D tw(Engine::Math::SIMD::Mul(t.reg, wReg)); 
        return v + tw + qVec.cross(t);
    }

    FORCE_INLINE float dot(const SIMDQuaternion& other) const {
        return Engine::Math::SIMD::Dot4(reg, other.reg);
    }

    // Converts a normalized forward vector into a rotation without using Trigonometry.
    static FORCE_INLINE SIMDQuaternion FromDirection(const SIMDVector3D& dir) {
        SIMDVector3D baseForward(0.0f, 0.0f, -1.0f, 0.0f); 
        float d = baseForward.dot(dir);
        
        // Edge Case: The entity needs to perfectly turn around 180 degrees
        if (d < -0.9999f) {
            return SIMDQuaternion(0.0f, 1.0f, 0.0f, 0.0f); // 180-degree Yaw
        }
        
        // Build the Quaternion using the cross product axis and the half-way dot product
        SIMDVector3D axis = baseForward.cross(dir);
        SIMDQuaternion q(axis.x(), axis.y(), axis.z(), 1.0f + d);
        q.Normalize();
        return q;
    }

    // --- EULER TO QUATERNION ---
    // Converts human-readable Euler Angles [Pitch (X), Yaw (Y), Roll (Z) in degrees] into quaternions.
    static FORCE_INLINE SIMDQuaternion FromEuler(float pitch, float yaw, float roll) {
        // Level-editors, player controller, or camera script will provide rotations in standard euler angles (pitch, yaw, roll). 
        float p = (pitch * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        float y = (yaw * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        float r = (roll * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;

        float sp = std::sin(p); float cp = std::cos(p);
        float sy = std::sin(y); float cy = std::cos(y);
        float sr = std::sin(r); float cr = std::cos(r);

        return SIMDQuaternion(
            sr * cp * cy - cr * sp * sy, // X
            cr * sp * cy + sr * cp * sy, // Y
            cr * cp * sy - sr * sp * cy, // Z
            cr * cp * cy + sr * sp * sy  // W
        );
    }

    // Interpolate rotations: Used for implementing smooth camera controls, cinematic splines, or AI rotation targeting (making AI entities slowly turn towards the player).
    static FORCE_INLINE SIMDQuaternion Slerp(const SIMDQuaternion& q1, const SIMDQuaternion& q2, float t) {
        float cosOmega = q1.dot(q2);
        Engine::Math::SIMD::Float4 q2Reg = q2.reg;

        if (cosOmega < 0.0f) {
            cosOmega = -cosOmega;
            q2Reg = Engine::Math::SIMD::FlipSignXYZ(q2Reg); // We need a full negate here, not just XYZ
            // For a true negate in your cross platform layer:
            #ifdef MATH_ISA_ARM
                q2Reg = vnegq_f32(q2Reg);
            #else
                q2Reg = _mm_xor_ps(q2Reg, _mm_set1_ps(-0.0f));
            #endif
        }

        if (cosOmega > 0.9999f) {
            Engine::Math::SIMD::Float4 tReg = Engine::Math::SIMD::Set1(t);
            Engine::Math::SIMD::Float4 oneMinusT = Engine::Math::SIMD::Sub(Engine::Math::SIMD::Set1(1.0f), tReg);
            SIMDQuaternion result(Engine::Math::SIMD::Add(Engine::Math::SIMD::Mul(q1.reg, oneMinusT), Engine::Math::SIMD::Mul(q2Reg, tReg)));
            result.Normalize();
            return result;
        }

        float omega = std::acos(cosOmega);
        float invSinOmega = 1.0f / std::sin(omega);

        float weight0 = std::sin((1.0f - t) * omega) * invSinOmega;
        float weight1 = std::sin(t * omega) * invSinOmega;

        Engine::Math::SIMD::Float4 res = Engine::Math::SIMD::Add(
            Engine::Math::SIMD::Mul(q1.reg, Engine::Math::SIMD::Set1(weight0)), 
            Engine::Math::SIMD::Mul(q2Reg, Engine::Math::SIMD::Set1(weight1))
        );

        return SIMDQuaternion(res);
    }
};

// ==================================================================================
// SIMD 4x4 MATRIX (CROSS-PLATFORM)
// ==================================================================================
struct alignas(64) Matrix4x4_SIMD {
    Engine::Math::SIMD::Float4 col[4];

    static FORCE_INLINE Matrix4x4_SIMD Identity() {
        Matrix4x4_SIMD mat;
        mat.col[0] = Engine::Math::SIMD::Set(1.0f, 0.0f, 0.0f, 0.0f);
        mat.col[1] = Engine::Math::SIMD::Set(0.0f, 1.0f, 0.0f, 0.0f);
        mat.col[2] = Engine::Math::SIMD::Set(0.0f, 0.0f, 1.0f, 0.0f);
        mat.col[3] = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f);
        return mat;
    }

    // Creates a Perspective Projection Matrix
    static FORCE_INLINE Matrix4x4_SIMD Perspective_SIMD(float fovY_degrees, float aspect, float nearZ, float farZ) {
        Matrix4x4_SIMD mat;
        float fovY_rad = fovY_degrees * (std::numbers::pi_v<float> / 180.0f);
        float tanHalfFovY = std::tan(fovY_rad / 2.0f);

        float m00 = 1.0f / (aspect * tanHalfFovY);
        float m11 = 1.0f / tanHalfFovY;
        float m22 = -(farZ + nearZ) / (farZ - nearZ);
        float m32 = -(2.0f * farZ * nearZ) / (farZ - nearZ);

        using namespace Engine::Math::SIMD;
        mat.col[0] = Set(m00, 0.0f, 0.0f, 0.0f);
        mat.col[1] = Set(0.0f, m11, 0.0f, 0.0f);
        mat.col[2] = Set(0.0f, 0.0f, m22, -1.0f);
        mat.col[3] = Set(0.0f, 0.0f, m32, 0.0f);

        return mat;
    }

    static FORCE_INLINE Matrix4x4_SIMD LookAt_SIMD(const SIMDVector3D& eye, const SIMDVector3D& target, const SIMDVector3D& upVec) {
        SIMDVector3D f = (target - eye).asDirection();
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) f = f * (1.0f / std::sqrt(fLenSq));

        SIMDVector3D r = f.cross(upVec).asDirection();
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) r = r * (1.0f / std::sqrt(rLenSq));

        SIMDVector3D u = r.cross(f).asDirection();

        // Negate Forward for Right-Handed coordinates
        Engine::Math::SIMD::Float4 negF = Engine::Math::SIMD::Mul(f.reg, Engine::Math::SIMD::Set1(-1.0f));

        float tx = -r.dot(eye);
        float ty = -u.dot(eye);
        float tz = f.dot(eye); 
        Engine::Math::SIMD::Float4 translation = Engine::Math::SIMD::Set(tx, ty, tz, 1.0f);

        Engine::Math::SIMD::Float4 row0 = r.reg;
        Engine::Math::SIMD::Float4 row1 = u.reg;
        Engine::Math::SIMD::Float4 row2 = negF;
        Engine::Math::SIMD::Float4 row3 = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f);

        // Hardware Transpose
        Engine::Math::SIMD::Transpose4(row0, row1, row2, row3);

        Matrix4x4_SIMD mat;
        mat.col[0] = row0;
        mat.col[1] = row1;
        mat.col[2] = row2;
        mat.col[3] = translation; 
        return mat;
    }

    // Converts ECS TransformComponent vectors and quaternions into the 4x4 matrices required for GPU APIs (i.e., Vulkan, OpenGL, DirectX 12).
    static FORCE_INLINE Matrix4x4_SIMD TRS(const SIMDVector3D& translation, const SIMDQuaternion& rotation, const SIMDVector3D& scale) {
        Matrix4x4_SIMD mat;

        // 1. Convert Quaternion to a 3x3 Rotation Matrix
        float x2 = rotation.x() + rotation.x(), y2 = rotation.y() + rotation.y(), z2 = rotation.z() + rotation.z();
        float xx = rotation.x() * x2, xy = rotation.x() * y2, xz = rotation.x() * z2;
        float yy = rotation.y() * y2, yz = rotation.y() * z2, zz = rotation.z() * z2;
        float wx = rotation.w() * x2, wy = rotation.w() * y2, wz = rotation.w() * z2;

        // 2. Apply Scale directly to the rotation columns
        float sx = scale.x();
        float sy = scale.y();
        float sz = scale.z();

        // The arguments are now in standard (X, Y, Z, W) order!
        mat.col[0] = Engine::Math::SIMD::Set((1.0f - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0.0f);
        mat.col[1] = Engine::Math::SIMD::Set((xy - wz) * sy, (1.0f - (xx + zz)) * sy, (yz + wx) * sy, 0.0f);
        mat.col[2] = Engine::Math::SIMD::Set((xz + wy) * sz, (yz - wx) * sz, (1.0f - (xx + yy)) * sz, 0.0f);

        // 3. Inject Translation into the 4th Column (Ensure W = 1.0f)
        Engine::Math::SIMD::Float4 wOne = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f);
        mat.col[3] = Engine::Math::SIMD::BlendMaskW(translation.reg, wOne);

        return mat;
    }
};

// --- SIMD MATRIX OPERATORS ---
FORCE_INLINE SIMDVector3D operator*(const Matrix4x4_SIMD& mat, const SIMDVector3D& v) {
    Engine::Math::SIMD::Float4 vx = Engine::Math::SIMD::Set1(v.x());
    Engine::Math::SIMD::Float4 vy = Engine::Math::SIMD::Set1(v.y());
    Engine::Math::SIMD::Float4 vz = Engine::Math::SIMD::Set1(v.z());
    Engine::Math::SIMD::Float4 vw = Engine::Math::SIMD::Set1(v.w());

    Engine::Math::SIMD::Float4 res = Engine::Math::SIMD::Mul(vx, mat.col[0]);
    res = Engine::Math::SIMD::FMAdd(vy, mat.col[1], res);
    res = Engine::Math::SIMD::FMAdd(vz, mat.col[2], res);
    res = Engine::Math::SIMD::FMAdd(vw, mat.col[3], res);

    return SIMDVector3D(res);
}

FORCE_INLINE Matrix4x4_SIMD operator*(const Matrix4x4_SIMD& a, const Matrix4x4_SIMD& b) {
    Matrix4x4_SIMD res;
    for (int i = 0; i < 4; ++i) {
        Engine::Math::SIMD::Float4 vx = Engine::Math::SIMD::Set1(Engine::Math::SIMD::ExtractX(b.col[i]));
        Engine::Math::SIMD::Float4 vy = Engine::Math::SIMD::Set1(Engine::Math::SIMD::ExtractY(b.col[i]));
        Engine::Math::SIMD::Float4 vz = Engine::Math::SIMD::Set1(Engine::Math::SIMD::ExtractZ(b.col[i]));
        Engine::Math::SIMD::Float4 vw = Engine::Math::SIMD::Set1(Engine::Math::SIMD::ExtractW(b.col[i]));

        Engine::Math::SIMD::Float4 col = Engine::Math::SIMD::Mul(vx, a.col[0]);
        col = Engine::Math::SIMD::FMAdd(vy, a.col[1], col);
        col = Engine::Math::SIMD::FMAdd(vz, a.col[2], col);
        col = Engine::Math::SIMD::FMAdd(vw, a.col[3], col);
        
        res.col[i] = col;
    }
    return res;
}

// ==================================================================================
// LARGE WORLD COORDINATES (LWC)
// ==================================================================================
struct Vector3DWorld {
    double x, y, z;

    constexpr Vector3DWorld(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}

    FORCE_INLINE constexpr Vector3DWorld operator+(const Vector3DWorld& other) const { return Vector3DWorld(x + other.x, y + other.y, z + other.z); }
    FORCE_INLINE constexpr Vector3DWorld operator-(const Vector3DWorld& other) const { return Vector3DWorld(x - other.x, y - other.y, z - other.z); }

    // --- THE LWC BRIDGE ---
    // Safely casts a 64-bit world difference down to your ultra-fast 32-bit SIMD vector.
    // By returning a Vector3D (your SIMD wrapper), the downstream math instantly utilizes AVX/NEON.
    FORCE_INLINE Vector3D toFloatVector() const {
        return Vector3D(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }
};

// ==================================================================================
// AXIS-ALIGNED BOUNDING BOX (AABB)
// ==================================================================================
// Every mesh needs an invisible box around it. When the engine calculates collisions or checks if an object is visible, it does not check all 10,000 vertices of a 3D model. It checks the 8 corners of an AABB.
struct alignas(32) AABBMath {
    SIMDVector3D minBounds;     // Bottom-Left-Back corner
    SIMDVector3D maxBounds;     // Top-Right-Front corner

    // --- HARDWARE INTERSECTION TEST (CROSS-PLATFORM 100% SIMD) ---
    // Returns true if this box is overlapping with another box
    FORCE_INLINE bool Intersects(const AABBMath& other) const {
        using namespace Engine::Math::SIMD;

        // --- ARM NEON (Apple Silicon / Mobile) INTEL / AMD (SSE/AVX) ---
        Float4 maxGTmin = CmpGt(maxBounds.reg, other.minBounds.reg);      // a.max > b.min
        Float4 minLTmax = CmpLt(minBounds.reg, other.maxBounds.reg);      // a.min < b.max

        // Combine the results. If all lanes (X,Y,Z) are true, they intersect.
        Float4 result = BitwiseAnd(maxGTmin, minLTmax);
        
        // 0x07 (binary 0111) checks only the X, Y, and Z lanes (ignoring W)
        // We explicitly check if lanes 0, 1, and 2 (X, Y, Z) evaluated to true, ignoring W.
        return (MoveMask(result) & 0x07) == 0x07;
    }

    // --- WORLD SPACE TRANSFORMATION ---
    // Transforms a local AABB (3D Mesh Local Space centered around (0, 0, 0)) into World Space (Frustum) using the entity's Model Matrix
    FORCE_INLINE AABBMath Transform(const Matrix4x4_SIMD& m) const {
        // The most efficient way to transform an AABB is to multiply its center and extents
        SIMDVector3D center = (maxBounds + minBounds) * 0.5f;
        SIMDVector3D extents = (maxBounds - minBounds) * 0.5f;

        // Transform the center normally (Matrix * Vector applies rotation and translation)
        SIMDVector3D newCenter = m * center;

        // To transform extents, we must multiply by the ABSOLUTE values of the rotation matrix
        // (Strip the sign bit from columns 0, 1, and 2)
        using namespace Engine::Math::SIMD;

        // Cross-platform ABS mask: Clear the sign bit (the 32nd bit) of each float.
        #ifdef MATH_ISA_ARM
            Float4 absMask = vreinterpretq_f32_u32(vdupq_n_u32(0x7FFFFFFF)); // Bitwise mask to clear the sign bit (vreinterpretq_f32_u32(Set1_u32(0x7FFFFFFF)))
        #else
            Float4 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));  // Bitwise mask to clear the sign bit
        #endif
        

        Float4 absCol0 = BitwiseAnd(m.col[0], absMask);
        Float4 absCol1 = BitwiseAnd(m.col[1], absMask);
        Float4 absCol2 = BitwiseAnd(m.col[2], absMask);

        Float4 ex = Set1(extents.x());
        Float4 ey = Set1(extents.y());
        Float4 ez = Set1(extents.z());

        Float4 newExtents = Mul(ex, absCol0);
        newExtents = FMAdd(ey, absCol1, newExtents);
        newExtents = FMAdd(ez, absCol2, newExtents);

        SIMDVector3D finalExtents(newExtents);

        return { newCenter - finalExtents, newCenter + finalExtents };
    }
};

// ==================================================================================
// 4. MATRICES & INTERPOLATION
// ==================================================================================

// --- 3D CAMERA & MATRIX MATH ---
// --- 4x4 MATRIX MATH (Stack Allocated, Column-Major for OpenGL) ---
struct Matrix4 {
    // std::array for better safety and constexpr support than raw arrays.
    alignas(64) std::array<float, 16> m{}; // Initializes to all zeros

    // Creates an Identity Matrix
    static constexpr Matrix4 Identity() {
        Matrix4 mat;
        mat.m[0] = 1.0f; mat.m[5] = 1.0f; mat.m[10] = 1.0f; mat.m[15] = 1.0f;
        return mat;
    }

    // C++23 mdspan gives you 2D syntax (mat[row, col]) over a 1D memory block with zero overhead
    constexpr auto getView() {
        // std::extents<Type, Rows, Cols> guarantees zero runtime overhead
        return std::mdspan<float, std::extents<std::size_t, 4, 4>>(m.data());
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

    // Creates an Orthographic Projection Matrix
    static Matrix4 Orthographic(float left, float right, float bottom, float top, float nearZ, float farZ) {
        Matrix4 mat;
        float invRL = 1.0f / (right - left);
        float invTB = 1.0f / (top - bottom);
        float invFN = 1.0f / (farZ - nearZ);

        mat.m[0] = 2.0f * invRL;
        mat.m[5] = 2.0f * invTB;
        mat.m[10] = -2.0f * invFN;
        mat.m[12] = -(right + left) * invRL;
        mat.m[13] = -(top + bottom) * invTB;
        mat.m[14] = -(farZ + nearZ) * invFN;
        mat.m[15] = 1.0f;
        return mat;
    }

    // --- STANDARD CAMERA LOOK-AT ---
    // Creates a View Matrix to transform from World Space to View (Camera) Space.
    static Matrix4 LookAt(const Vector3D& eye, const Vector3D& target, const Vector3D& upVec) {

        // 1. Calculate the Forward Vector (Z) from [Eye to Target], (Right-handed systems: camera looks down -Z axis) .
        Vector3D f = Vector3D(target.x - eye.x, target.y - eye.y, target.z - eye.z); // (target - eye) points INTO the screen, but we will negate it when building the matrix to point it back at us.
        float fLen = std::sqrt(f.dot(f));

        if (fLen > 1e-8f) {
            f.x *= (1.0f / fLen); f.y *= (1.0f / fLen); f.z *= (1.0f / fLen);
        }

        // 2. Calculate the Right Vector (X) using the cross product of Forward and Up.
        Vector3D r = f.cross(upVec);
        float rLen = std::sqrt(r.dot(r));
        if (rLen > 1e-8f) {
            r.x *= (1.0f / rLen); r.y *= (1.0f / rLen); r.z *= (1.0f / rLen);
        }

        // 3. Recalculate the orthogonal Up Vector (Y).
        Vector3D u = r.cross(f);

        // 4. Build Column-Major Matrix
        Matrix4 mat = Identity();
        mat.m[0] = r.x;  mat.m[4] = r.y;  mat.m[8] = r.z;      // Col 0: Right
        mat.m[1] = u.x;  mat.m[5] = u.y;  mat.m[9] = u.z;      // Col 1: Up
        mat.m[2] = -f.x; mat.m[6] = -f.y; mat.m[10] = -f.z;    // Col 2: Forward (Negated for right-handed coordinates)
        
        // 5. Calculate and inject Translation Offsets (moving the world inverse to the camera).
        mat.m[12] = -r.dot(eye);
        mat.m[13] = -u.dot(eye);
        mat.m[14] = f.dot(eye);   // Note: Positive dot product here because 'f' was negated above.
        return mat;
    }

    // --- LWC CAMERA-RELATIVE RENDERING LOOK-AT ---
    // View Matrix: No longer needs translation with camera-relative rendering, only handles rotation.
    // Notice the inputs are Vector3DWorld (double), but the matrix is float.
    static Matrix4 LookAtLWC(const Vector3DWorld& eye, const Vector3DWorld& target, const Vector3D& upVec) {
        
        // 1. Calculate the forward vector in 64-bit space to prevent jitter at massive distances
        Vector3DWorld worldForward = target - eye;
        
        // 2. Cast down to 32-bit float for the math. 
        // Because it's a directional vector (difference), the cast is perfectly safe!
        Vector3D f = worldForward.toFloatVector();
        
        // Use your fast SIMD dot product to normalize
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) {
            f *= (1.0f / std::sqrt(fLenSq));
        }

        // 3. Right Vector (X)
        Vector3D r = f.cross(upVec);
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) {
            r *= (1.0f / std::sqrt(rLenSq));
        }

        // 4. Up Vector (Y)
        Vector3D u = r.cross(f);

        // 5. Build Column-Major Matrix
        Matrix4 mat = Identity();
        mat.m[0] = r.x;  mat.m[4] = r.y;  mat.m[8] = r.z;
        mat.m[1] = u.x;  mat.m[5] = u.y;  mat.m[9] = u.z;
        mat.m[2] = -f.x; mat.m[6] = -f.y; mat.m[10] = -f.z;
        
        // 6. ZERO TRANSLATION!
        // Because every object will be rendered relative to the camera, the camera is always at (0,0,0).
        mat.m[12] = 0.0f; 
        mat.m[13] = 0.0f; 
        mat.m[14] = 0.0f; 

        return mat;
    }
    /*
        - Build entities using its camera relative position.
        - Don't build entities using its absolute position.
        - By isolating double purely to a single property (world coordinate) and a single operation (subtraction), the engine gets a large scale and preserves all AVX2/SSE optimizations for physics, culling , and rendering.

        // 1. The player's camera sits 50,000 kilometers away from the origin.
        Vector3DWorld cameraWorldPos(50000000.0, 10.0, 200000.0);

        // 2. A tree is sitting right next to the player.
        Vector3DWorld treeWorldPos(50000015.0, 10.0, 200005.0);

        // 3. Right before rendering, we calculate the tree's position relative to the camera.
        // (treeWorldPos - cameraWorldPos) yields: (15.0, 0.0, 5.0) in double precision.
        Vector3DWorld relativeWorldPos = treeWorldPos - cameraWorldPos;

        // 4. We safely cast this small difference to 32-bit floats.
        Vector3D relativeLocalPos = relativeWorldPos.toFloatVector();

        // 5. Now, you build your Model Matrix using `relativeLocalPos` and send it to the GPU!
        Matrix4 treeModelMatrix = BuildTranslationMatrix(relativeLocalPos);
    */

    // --- SCALAR MATRIX INVERSE ---
    // Essential for generating Raycasts from the Camera, and generating Normal Matrices for shaders.
    Matrix4 Inverse() const {
        Matrix4 inv;
        float* out = inv.m.data();
        const float* in = m.data();

        out[0] = in[5]  * in[10] * in[15] - in[5]  * in[11] * in[14] - in[9]  * in[6]  * in[15] + in[9]  * in[7]  * in[14] + in[13] * in[6]  * in[11] - in[13] * in[7]  * in[10];
        out[4] = -in[4]  * in[10] * in[15] + in[4]  * in[11] * in[14] + in[8]  * in[6]  * in[15] - in[8]  * in[7]  * in[14] - in[12] * in[6]  * in[11] + in[12] * in[7]  * in[10];
        out[8] = in[4]  * in[9]  * in[15] - in[4]  * in[11] * in[13] - in[8]  * in[5]  * in[15] + in[8]  * in[7]  * in[13] + in[12] * in[5]  * in[11] - in[12] * in[7]  * in[9];
        out[12] = -in[4]  * in[9]  * in[14] + in[4]  * in[10] * in[13] + in[8]  * in[5]  * in[14] - in[8]  * in[6]  * in[13] - in[12] * in[5]  * in[10] + in[12] * in[6]  * in[9];
        out[1] = -in[1]  * in[10] * in[15] + in[1]  * in[11] * in[14] + in[9]  * in[2]  * in[15] - in[9]  * in[3]  * in[14] - in[13] * in[2]  * in[11] + in[13] * in[3]  * in[10];
        out[5] = in[0]  * in[10] * in[15] - in[0]  * in[11] * in[14] - in[8]  * in[2]  * in[15] + in[8]  * in[3]  * in[14] + in[12] * in[2]  * in[11] - in[12] * in[3]  * in[10];
        out[9] = -in[0]  * in[9]  * in[15] + in[0]  * in[11] * in[13] + in[8]  * in[1]  * in[15] - in[8]  * in[3]  * in[13] - in[12] * in[1]  * in[11] + in[12] * in[3]  * in[9];
        out[13] = in[0]  * in[9]  * in[14] - in[0]  * in[10] * in[13] - in[8]  * in[1]  * in[14] + in[8]  * in[2]  * in[13] + in[12] * in[1]  * in[10] - in[12] * in[2]  * in[9];
        out[2] = in[1]  * in[6]  * in[15] - in[1]  * in[7]  * in[14] - in[5]  * in[2]  * in[15] + in[5]  * in[3]  * in[14] + in[13] * in[2]  * in[7]  - in[13] * in[3]  * in[6];
        out[6] = -in[0]  * in[6]  * in[15] + in[0]  * in[7]  * in[14] + in[4]  * in[2]  * in[15] - in[4]  * in[3]  * in[14] - in[12] * in[2]  * in[7]  + in[12] * in[3]  * in[6];
        out[10] = in[0]  * in[5]  * in[15] - in[0]  * in[7]  * in[13] - in[4]  * in[1]  * in[15] + in[4]  * in[3]  * in[13] + in[12] * in[1]  * in[7]  - in[12] * in[3]  * in[5];
        out[14] = -in[0]  * in[5]  * in[14] + in[0]  * in[6]  * in[13] + in[4]  * in[1]  * in[14] - in[4]  * in[2]  * in[13] - in[12] * in[1]  * in[6]  + in[12] * in[2]  * in[5];
        out[3] = -in[1]  * in[6]  * in[11] + in[1]  * in[7]  * in[10] + in[5]  * in[2]  * in[11] - in[5]  * in[3]  * in[10] - in[9]  * in[2]  * in[7]  + in[9]  * in[3]  * in[6];
        out[7] = in[0]  * in[6]  * in[11] - in[0]  * in[7]  * in[10] - in[4]  * in[2]  * in[11] + in[4]  * in[3]  * in[10] + in[8]  * in[2]  * in[7]  - in[8]  * in[3]  * in[6];
        out[11] = -in[0]  * in[5]  * in[11] + in[0]  * in[7]  * in[9]  + in[4]  * in[1]  * in[11] - in[4]  * in[3]  * in[9]  - in[8]  * in[1]  * in[7]  + in[8]  * in[3]  * in[5];
        out[15] = in[0]  * in[5]  * in[10] - in[0]  * in[6]  * in[9]  - in[4]  * in[1]  * in[10] + in[4]  * in[2]  * in[9]  + in[8]  * in[1]  * in[6]  - in[8]  * in[2]  * in[5];

        float det = in[0] * out[0] + in[1] * out[4] + in[2] * out[8] + in[3] * out[12];
        if (det != 0.0f) {
            float invDet = 1.0f / det;
            for (int i = 0; i < 16; i++) {
                out[i] *= invDet;
            }
        }
        return inv;
    }
};

// --- 4x4 SCALAR MATRIX MULTIPLICATION (COLUMN-MAJOR) ---
// Multiplies two 4x4 matrices (C = A * B)
FORCE_INLINE constexpr Matrix4 operator*(const Matrix4& a, const Matrix4& b) {
    Matrix4 res{}; // Initializes to zero

    // Standard 4x4 matrix multiplication unrolled for column-major memory layout.
    // Standard row-by-column multiplication adapted for 1D column-major arrays
    // m[col * 4 + row]
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            res.m[col * 4 + row] = 
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }
    return res;
}

// --- SCALAR MATRIX x VECTOR MULTIPLICATION ---
// Multiplies a 4x4 Matrix by a 3D Vector (V' = M * V)
FORCE_INLINE constexpr Vector3D operator*(const Matrix4& m, const Vector3D& v) {
    float x = (m.m[0] * v.x) + (m.m[4] * v.y) + (m.m[8] * v.z)  + (m.m[12] * v.w);
    float y = (m.m[1] * v.x) + (m.m[5] * v.y) + (m.m[9] * v.z)  + (m.m[13] * v.w);
    float z = (m.m[2] * v.x) + (m.m[6] * v.y) + (m.m[10] * v.z) + (m.m[14] * v.w);
    float w = (m.m[3] * v.x) + (m.m[7] * v.y) + (m.m[11] * v.z) + (m.m[15] * v.w);

    return Vector3D(x, y, z);
}

// ==================================================================================
// VIEW FRUSTUM (CAMERA CULLING)
// ==================================================================================
/*
    - Extracts the 6 planes of the camera's view so you don't render objects behind the player.
    - A plane is defined as Ax + By + Cz + D = 0. 
    - We store (A,B,C) as the normal vector, and D as W.
*/
struct Frustum {
    // 6 Planes (Left, Right, Bottom, Top, Near, Far)
    SIMDVector3D planes[6];

    // Extracts the 6 frustum planes from a combined View-Projection matrix
    void ExtractFromVP(const Matrix4x4_SIMD& vp) {
        using namespace Engine::Math::SIMD;

        // 1. Fetch the 4 columns
        Float4 r0 = vp.col[0];
        Float4 r1 = vp.col[1];
        Float4 r2 = vp.col[2];
        Float4 r3 = vp.col[3];

        // 2. Hardware Transpose flips them so r0-r3 now represent the ROWS of the matrix
        Transpose4(r0, r1, r2, r3);
        
        // 3. NOW we can safely extract the planes!
        planes[0] = SIMDVector3D(Add(r3, r0)); // Left
        planes[1] = SIMDVector3D(Sub(r3, r0)); // Right
        planes[2] = SIMDVector3D(Add(r3, r1)); // Bottom
        planes[3] = SIMDVector3D(Sub(r3, r1)); // Top
        planes[4] = SIMDVector3D(Add(r3, r2)); // Near
        planes[5] = SIMDVector3D(Sub(r3, r2)); // Far

        // Normalize all 6 planes using SIMD
        for (int i = 0; i < 6; ++i) {
            SIMDVector3D normal = planes[i].asDirection(); // Mask out W
            float lengthSq = normal.dot(normal);
            if (lengthSq > 1e-8f) {
                planes[i] = planes[i] * (1.0f / std::sqrt(lengthSq));
            }
        }
    }

    // --- FRUSTUM CULLING TEST (CROSS-PLATFORM 100% SIMD) ---
    // Returns true if the AABB is inside or touching the frustum
    FORCE_INLINE bool IsBoxVisible(const AABBMath& box) const {
        using namespace Engine::Math::SIMD;
        for (int i = 0; i < 6; ++i) {
            // 1. Create a mask of where the plane's normal is greater than 0
            Float4 cmpMask = CmpGt(planes[i].reg, Zero());

            // 2. Blend maxBounds and minBounds based on that mask.
            // If normal > 0, pick maxBounds. If normal <= 0, pick minBounds.
            Float4 pVec = BlendVariable(box.minBounds.reg, box.maxBounds.reg, cmpMask);
            
            // 3. Hardware Dot Product: (P_xyz dot Normal_xyz) + Plane_W
            // We use the 0x7F mask to calculate the dot product of X, Y, Z and write it to all lanes.
            Float4 dotResult = Set1(Dot3(pVec, planes[i].reg)); // Only dot X,Y,Z

            // Extract the plane's W component (Distance from origin)
            Float4 planeW = Set1(ExtractW(planes[i].reg));
            
            // Final Distance = Dot(P, Normal) + W
            Float4 distance = Add(dotResult, planeW);

            // 4. If distance < 0.0f, the box is outside the frustum.
            if (ExtractX(distance) < 0.0f) {
                return false;
            }
        }
        return true;

        /*  
            - Do not extract scalars, it is slow. 
            - It forces _mm_shuffle_ps to extract those individual floats out of the __mm128 register.
            - 18 shuffle instructions just to check if a box is on the screen.
        */
        // for (int i = 0; i < 6; ++i) {
        //     // Find the corner of the AABB that is furthest along the plane's normal
        //     float px = (planes[i].x() > 0.0f) ? box.maxBounds.x() : box.minBounds.x();
        //     float py = (planes[i].y() > 0.0f) ? box.maxBounds.y() : box.minBounds.y();
        //     float pz = (planes[i].z() > 0.0f) ? box.maxBounds.z() : box.minBounds.z();

        //     // Calculate distance from the plane to the positive vertex
        //     float distance = (planes[i].x() * px) + (planes[i].y() * py) + (planes[i].z() * pz) + planes[i].w();

        //     // If the furthest corner is behind the plane, the entire box is invisible!
        //     if (distance < 0.0f) return false; 
        // }
        // return true;
    }
};
