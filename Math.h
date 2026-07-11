#pragma once

#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <numbers>     // C++20/26 Standardized Math Constants
#include <cstdint>
#include <array>
#include <mdspan>
#include <cstddef>
#include <span>
#include <algorithm>   // Required for std::min, std::copy, std::swap
#include <bit>         // Required for std::bit_cast
#include <utility>     // Required for std::pair

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
    #include <arm_neon.h>      // ARM NEON (64-bit), SVE (128-bit)
    #define MATH_ISA_ARM       // Apple Silicon (M-series), Mobile (iPhone, Android) (ARM64)
#else
    #include <immintrin.h>     // AVX2 (256-bit), SSE (128-bit), MMX (64-bit).
    #define MATH_ISA_X86       // Intel/AMD (PS5, Xbox, PC) (x86_x64)
#endif

// ====================================================
// HEXADECIMAL FLOATING-POINT LITERALS (0x)
// ====================================================
/*
    - Base-10 decimals need to be converted (or parsed) into a Base-2 binary fraction (IEEE-754 format), so the CPU can use it (i.e., text to binary conversion).
    - Different compilers round these conversions differently (i.e., compiled float on linux may be slightly different than a Window server at the very end of the mantissa).

      Non-Deterministic [Server Authoritative Networking, Interpolation]: Minor floating-point drift does not matter b/c the server dictates the absolute truth (i.e., clients interpolate their local objects to match the server) in online first-person shooters and online open-world games.

        1. Provide exact same inputs (e.g., 3.14159265359f) to MSVC compiler.
        2. Provide exact same inputs (e.g., 3.14159265359f) to Clang compiler.
        3. Both will not simulate the exact same physics and math results.

    - A hex-float bypases Base-10 to Base-2 string parsing phase completely (i.e., zero chance of compiler rounding differences).
    - Hands the compiler the exact raw silicon IEEE-754 mantissa and exponent bits.
    - Provides bit-perfect deterministic guarantees across every compiler b/c it strips the compilers ability of rounding the math during the text-parsing phase.
    - Ensures it runs identical on Xbox Series X (AMD), Playstation 5 (AMD), iPhone (ARM), and PC (AMD/Intel).

      Deterministic [Rollback Netcode, Lock-Step Networking]: prevents desynchronization over the network in online fighting games.

        1. Provide exact same inputs (e.g., 0x1.921fb6p+1f) to MSVC compiler.
        2. Provide exact same inputs (e.g., 0x1.921fb6p+1f) to Clang compiler.
        3. Both will simulate the exact same physics and math resultsdown to the final decimal place forever.
*/

namespace Engine::Math::Constants {

    // Standard floating-point Pi
    inline constexpr float PI = std::numbers::pi_v<float>;         // = 0x1.921fb6p+1f; | 3.14159265359f
    inline constexpr float PI_HALF = PI * 0.5f;                    // = 0x1.921fb6p+0f; | 1.57079632679f
    inline constexpr float PI_DOUBLE = PI * 2.0f;                  // = 0x1.921fb6p+2f; | 6.28318530718f

    // Fast conversion multipliers (Evaluated at compile-time)
    inline constexpr float DEG_TO_RAD = PI / 180.0f;               // = 0x1.1df46ap-6f; | 0.01745329251f (PI / 180.0f)
    inline constexpr float RAD_TO_DEG = 180.0f / PI;               // = 0x1.ca5dc2p+5f; | 57.2957795131f (180.0f / PI)

    // ======================================================================
    // HARDWARE CONSTANTS
    // ======================================================================
    /*
        - Pre-broadcasted 128-bit hardware constants.
        - No Set1 broadcasts required; constants are pulled instantly from L1 Cache.
        - This allows the CPU to load the entire vector instantly using a single 'movaps' instruction, completely eliminating the runtime cost of scalar broadcasting (_mm_set1_ps).
    */
    
    #ifdef MATH_ISA_ARM
        // --- ARM APPLE SILICON / MOBILE (NEON) ---
        using Float4 = float32x4_t;

        #define SIMD_CONST(val) {val, val, val, val}
        #define SIMD_VEC(x, y, z, w) {x, y, z, w}
    #else
        // --- INTEL / AMD PC (SSE / AVX2) ---
        using Float4 = __m128;

        // MSVC/GCC will evaluate this at compile time for static consts
        #define SIMD_CONST(val) _mm_set1_ps(val) 
        // Intel intrinsic _mm_set_ps requires arguments in reverse (W, Z, Y, X) order!
        #define SIMD_VEC(x, y, z, w) _mm_set_ps(w, z, y, x) 
    #endif

    // --- CORE ENGINE CONSTANTS (L1 CACHED) ---
    static const Float4 SIMD_NEG_ONE  = SIMD_CONST(-0x1.0p+0f); // -1.0f
    static const Float4 SIMD_NEG_ZERO = SIMD_CONST(-0x0.0p+0f); // -0.0f
    static const Float4 SIMD_ZERO     = SIMD_CONST(0x0.0p+0f);  //  0.0f
    static const Float4 SIMD_HALF     = SIMD_CONST(0x1.0p-1f);  //  0.5f 
    static const Float4 SIMD_ONE      = SIMD_CONST(0x1.0p+0f);  //  1.0f
    static const Float4 SIMD_TWO      = SIMD_CONST(0x1.0p+1f);  //  2.0f
    static const Float4 SIMD_THREE    = SIMD_CONST(0x1.8p+1f);  //  3.0f
    

    // Common Vector Masks
    static const Float4 SIMD_MASK_W_ONE = SIMD_VEC(0x0.0p+0f, 0x0.0p+0f, 0x0.0p+0f, 0x1.0p+0f); // 0.0f, 0.0f, 0.0f, 1.0f

    // FastSin Constants (Horner's Method)
    static const Float4 SIN_C9 = SIMD_CONST(0x1.71de3ap-19f);  //  0.00000275573f  ( 1/9!)
    static const Float4 SIN_C7 = SIMD_CONST(-0x1.a01a02p-13f); // -0.00019841269f  (-1/7!)
    static const Float4 SIN_C5 = SIMD_CONST(0x1.111112p-7f);   //  0.00833333333f  ( 1/5!)
    static const Float4 SIN_C3 = SIMD_CONST(-0x1.555556p-3f);  // -0.16666666667f  (-1/3!)
    static const Float4 SIN_1  = SIMD_CONST(0x1.0p+0f);        //  1.0f

    // FastCos Constants (Horner's Method)
    static const Float4 COS_C8 = SIMD_CONST(0x1.a01a02p-16f);  //  0.00002480158f  ( 1/8!)
    static const Float4 COS_C6 = SIMD_CONST(-0x1.6c16c2p-10f); // -0.00138888888f  (-1/6!)
    static const Float4 COS_C4 = SIMD_CONST(0x1.555556p-5f);   //  0.04166666666f  ( 1/4!)
    static const Float4 COS_C2 = SIMD_CONST(-0x1.000000p-1f);  // -0.5f            (-1/2!)
    static const Float4 COS_1  = SIMD_CONST(0x1.0p+0f);        //  1.0f

    // FastACos Constants
    static const Float4 ACOS_C3 = SIMD_CONST(-0x1.32d164p-6f); // -0.0187293f
    static const Float4 ACOS_C2 = SIMD_CONST(0x1.302b1ap-4f);  //  0.0742610f
    static const Float4 ACOS_C1 = SIMD_CONST(-0x1.b2694ep-3f); // -0.2121144f
    static const Float4 ACOS_1  = SIMD_CONST(0x1.0p+0f);       //  1.0f

    // FastTan Constants (Rational Approximation)
    static const Float4 TAN_C0 = SIMD_CONST(0x1.0p+0f);        //  1.0f
    static const Float4 TAN_C1 = SIMD_CONST(-0x1.111112p-3f);  // -0.13333333f
    static const Float4 TAN_C2 = SIMD_CONST(0x1.ba2e8cp-9f);   //  0.00338663f
    static const Float4 TAN_C3 = SIMD_CONST(0x1.0p+0f);        //  1.0f
    static const Float4 TAN_C4 = SIMD_CONST(-0x1.d55556p-2f);  // -0.45833333f
    static const Float4 TAN_C5 = SIMD_CONST(0x1.218526p-5f);   //  0.03534943f

    // PI Constants
    static const Float4 ACOS_C0 = SIMD_CONST(PI_HALF);         //  1.57079632679f
    static const Float4 ACOS_PI = SIMD_CONST(PI);              //  3.14159265359f

    // Cody-Waite Range Reduction Constants (Pre-Broadcasted for SIMD FastSinCos)
    static const Float4 INV_TWO_PI = SIMD_CONST(0x1.45f306p-3f);   // 1.0f / (2.0f * PI)
    static const Float4 TWO_PI_HI   = SIMD_CONST(0x1.921fb0p+2f);  // High-precision chunk of (2 * PI)
    static const Float4 TWO_PI_LO   = SIMD_CONST(0x1.4442d1p-20f); // Low-precision tail of (2 * PI)
}

// Scalar Domain (1D Scalar Math): strictly for operations that only operate on scalar floats.
namespace Engine::Math::Functions {   

    // ======================================================================
    // SCALAR HARDWARE SQUARE ROOT (FPU)
    // ======================================================================
    /*
        - Bypasses the `<cmath>` standard library overhead and 'errno' domain checks.
        - Maps directly to the CPU's dedicated scalar FPU square root instruction.
        - Exactly as accurate as std::sqrt, but significantly faster due to zero branching (i.e., std::sqrt replacement).
    */
    FORCE_INLINE float FastSqrt(float x) {
        #ifdef MATH_ISA_ARM
            // --- ARM APPLE SILICON / MOBILE (ARM64) ---
            #if defined(__clang__) || defined(__GNUC__)
                // Compiles directly down to a single hardware 'fsqrt' instruction.
                // Eradicates the Vector-Scalar-Transition penalty.
                return __builtin_sqrtf(x);
            #else
                // Fallback for non-Clang/GCC ARM compilers
                return vgetq_lane_f32(vsqrtq_f32(vsetq_lane_f32(x, vdupq_n_f32(0.0f), 0)), 0);
            #endif
        #else
            // Intel/AMD SSE scalar square root
            // _mm_set_ss loads a single float into the lowest 32-bits, leaving the rest 0.
            // _mm_sqrt_ss calculates the sqrt of only that lowest 32-bits.
            // _mm_cvtss_f32 extracts it back to a standard float.
            return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
        #endif
    }

    // ======================================================================
    // SCALAR HARDWARE FLOOR (FPU)
    // ======================================================================
    /*
        - Bypasses the `<cmath>` standard library overhead and domain checks.
        - Maps directly to the CPU's dedicated scalar FPU rounding instruction.
        - Exactly as accurate as std::floor, but compiles down to a single instruction.
    */
    FORCE_INLINE float FastFloor(float x) {
        #ifdef MATH_ISA_ARM
            // --- ARM APPLE SILICON / MOBILE (ARM64) ---
            #if defined(__clang__) || defined(__GNUC__)
                // Compiles directly down to a single hardware 'frintm' instruction.
                return __builtin_floorf(x);
            #else
                // Fallback for non-Clang/GCC ARM compilers (Round to Minus Infinity)
                return vgetq_lane_f32(vrndmq_f32(vsetq_lane_f32(x, vdupq_n_f32(0.0f), 0)), 0);
            #endif
        #else
            // --- INTEL / AMD PC (SSE4.1) ---
            // _mm_floor_ss calculates the floor of the lowest 32-bits.
            return _mm_cvtss_f32(_mm_floor_ss(_mm_set_ss(x), _mm_set_ss(x)));
        #endif
    }
    
    // --- SCALAR TRANSCENDENTAL APPROXIMATIONS ) ---

    // 1. Standalone Scalar Sine (9th-Order Horner's Method) that is valid strictly for angles between [-PI, PI] (e.g., calculating a bullet trajectory arc).
    FORCE_INLINE float FastSin(float x) {
        float x2 = x * x;
        float s = 0x1.71de3ap-19f;             // SIN_C9
        s = (s * x2) - 0x1.a01a02p-13f;        // SIN_C7
        s = (s * x2) + 0x1.111112p-7f;         // SIN_C5
        s = (s * x2) - 0x1.555556p-3f;         // SIN_C3
        s = (s * x2) + 1.0f;                   // SIN_1
        return s * x;
    }

    // 2. Standalone Scalar Cosine (8th-Order Horner's Method) that is valid strictly for angles between [-PI, PI].
    FORCE_INLINE float FastCos(float x) {
        float x2 = x * x;
        float c = 0x1.a01a02p-16f;             // COS_C8
        c = (c * x2) - 0x1.6c16c2p-10f;        // COS_C6
        c = (c * x2) + 0x1.555556p-5f;         // COS_C4
        c = (c * x2) - 0.5f;                   // COS_C2
        c = (c * x2) + 1.0f;                   // COS_1
        return c;
    }

    // ==============================================
    // --- EULER CONVERSIONS (COMBINED (SIN-COS)) ---
    // ==============================================
    /*
        - Euler conversions usually need both Sine & Cosine of the same angle.
        - FastSinCos: use ONLY when BOTH values are needed for the same angle (e.g., Euler conversions, rotation matrix).
        - Calculating them in one function is significantly faster than calling two separate functions because x^2 only needs to be calculated once.
        - Valid strictly for angles between [-PI: -3.14 radians, PI: 3.14 radians].
    */

    // 3. 
    FORCE_INLINE std::pair<float, float> FastSinCos(float x) {
        // 1. Range Reduction to [-3.14 radians, 3.14 radians], wraps massive angles (e.g., x = 200.0f [200^8 = 2.56 x 10^18]) safely back to the -3.14 to 3.14 valid polynomial window to prevent NaN.
        float cycles = FastFloor((x * 0x1.45f306p-3f) + 0.5f); // 0x1.45f306p-3f = 1.0f / (2.0f * PI)

        // Cody-Waite 2-part FMA subtraction preserves mantissa precision for massive coordinates
        // 0x1.921fb0p+2f is the high-precision chunk of (2 * PI). 0x1.4442d1p-20f is the low-precision tail of (2 * PI).
        x = (x - (cycles * 0x1.921fb0p+2f)) - (cycles * 0x1.4442d1p-20f);

        // 2. Evaluates both Sine and Cosine simultaneously to share the x^2 multiplication to save CPU cycles.
        float x2 = x * x;

        float s = 0x1.71de3ap-19f;             // SIN_C9
        s = (s * x2) - 0x1.a01a02p-13f;        // SIN_C7
        s = (s * x2) + 0x1.111112p-7f;         // SIN_C5
        s = (s * x2) - 0x1.555556p-3f;         // SIN_C3
        s = (s * x2) + 1.0f;                   // SIN_1
        float outSin = s * x;

        float c = 0x1.a01a02p-16f;             // COS_C8
        c = (c * x2) - 0x1.6c16c2p-10f;        // COS_C6
        c = (c * x2) + 0x1.555556p-5f;         // COS_C4
        c = (c * x2) - 0.5f;                   // COS_C2
        c = (c * x2) + 1.0f;                   // COS_1
        float outCos = c;

        // C++17: structured binding (i.e., tuple) constructs and packs the variables exactly where they are declared.
        return {outSin, outCos};
    }

    // ======================================================================
    // SCALAR RATIONAL APPROXIMATION (TANGENT)
    // ======================================================================
    /*
        - Pure Scalar Rational Approximation of Tangent.
        - Highly accurate within [-PI/4, PI/4] (Perfect for FOV calculations).
        - Uses a rational polynomial to gracefully handle the curve near asymptotes.
    */
    FORCE_INLINE float FastTan(float x) {
        float x2 = x * x;

        // Numerator Polynomial: 1.0 + x^2 * (C1 + x^2 * C2)
        float num = 0x1.ba2e8cp-9f;           // TAN_C2 (0.00338663f)
        num = (num * x2) - 0x1.111112p-3f;    // TAN_C1 (-0.13333333f)
        num = (num * x2) + 1.0f;              // TAN_C0 (1.0f)

        // Denominator Polynomial: 1.0 + x^2 * (C4 + x^2 * C5)
        float den = 0x1.218526p-5f;           // TAN_C5 (0.03534943f)
        den = (den * x2) - 0x1.d55556p-2f;    // TAN_C4 (-0.45833333f)
        den = (den * x2) + 1.0f;              // TAN_C3 (1.0f)

        // Result = x * (Numerator / Denominator)
        // Standard FPU division is used here as it is highly optimized for scalar floats.
        return x * (num / den);
    }

    // Fast pure scalar absolute value. Compiles down to a single instruction (zero-cast bitwise clear).
    FORCE_INLINE constexpr float abs(float v) {
        // Treat the float as an integer, strip the 31st sign bit, and treat it as a float again.
        uint32_t i = std::bit_cast<uint32_t>(v);
        i &= 0x7FFFFFFF; // Clear the sign bit
        return std::bit_cast<float>(i);
    }
}

// Vector Domain (4D Vector Math): Strictly for operations that consume and return wide registers.
namespace Engine::Math::SIMD {    
    // DYNAMIC DISPATCH: Wire the engine to the silicon's maximum limit
    #ifdef MATH_ISA_ARM
        // --- ARM APPLE SILICON / MOBILE (NEON) ---
        using Float4 = float32x4_t;

        FORCE_INLINE Float4 Zero() { return vdupq_n_f32(0.0f); }
        FORCE_INLINE Float4 Set(float x, float y, float z, float w) { 
            #if defined(__GNUC__) || defined(__clang__)
                // Modern Clang/GCC maps this directly to 'fmov' and 'ins' register instructions
                return (float32x4_t){x, y, z, w}; 
            #else
                // Fallback: Pure register insertions (Zero Memory Access)
                float32x4_t res = vdupq_n_f32(x);
                res = vsetq_lane_f32(y, res, 1);
                res = vsetq_lane_f32(z, res, 2);
                res = vsetq_lane_f32(w, res, 3);
                return res;
            #endif
        }
        FORCE_INLINE Float4 Set1(float v) { return vdupq_n_f32(v); }
        
        FORCE_INLINE Float4 Add(Float4 a, Float4 b) { return vaddq_f32(a, b); }
        FORCE_INLINE Float4 Sub(Float4 a, Float4 b) { return vsubq_f32(a, b); }
        FORCE_INLINE Float4 Mul(Float4 a, Float4 b) { return vmulq_f32(a, b); }
        FORCE_INLINE Float4 FMAdd(Float4 a, Float4 b, Float4 c) { return vfmaq_f32(c, a, b); } // c + (a * b)

        // Blend: Replace W in 'a' with W from 'b'
        FORCE_INLINE Float4 BlendMaskW(Float4 a, Float4 b) { return vsetq_lane_f32(vgetq_lane_f32(b, 3), a, 3); }

        // vsetq_lane_f32 inserts a scalar directly into the specified lane (0, 1, 2, or 3)
        FORCE_INLINE Float4 InsertX(Float4 v, float val) { return vsetq_lane_f32(val, v, 0); }
        FORCE_INLINE Float4 InsertY(Float4 v, float val) { return vsetq_lane_f32(val, v, 1); }
        FORCE_INLINE Float4 InsertZ(Float4 v, float val) { return vsetq_lane_f32(val, v, 2); }
        FORCE_INLINE Float4 InsertW(Float4 v, float val) { return vsetq_lane_f32(val, v, 3); }
        
        FORCE_INLINE float ExtractX(Float4 v) { return vgetq_lane_f32(v, 0); }
        FORCE_INLINE float ExtractY(Float4 v) { return vgetq_lane_f32(v, 1); }
        FORCE_INLINE float ExtractZ(Float4 v) { return vgetq_lane_f32(v, 2); }
        FORCE_INLINE float ExtractW(Float4 v) { return vgetq_lane_f32(v, 3); }

        // NEON allows broadcasting a specific lane directly from one register to another
        FORCE_INLINE Float4 BroadcastX(Float4 v) { return vdupq_lane_f32(vget_low_f32(v), 0); }
        FORCE_INLINE Float4 BroadcastY(Float4 v) { return vdupq_lane_f32(vget_low_f32(v), 1); }
        FORCE_INLINE Float4 BroadcastZ(Float4 v) { return vdupq_lane_f32(vget_high_f32(v), 0); }
        FORCE_INLINE Float4 BroadcastW(Float4 v) { return vdupq_lane_f32(vget_high_f32(v), 1); }

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

        FORCE_INLINE Float4 Negate(Float4 v) { return vnegq_f32(v); }

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

        // --- ARM NEON MATRIX INVERSE (Geometric Cramer's Rule) ---
        FORCE_INLINE void Inverse4x4(Float4& c0, Float4& c1, Float4& c2, Float4& c3) {
            // 1. Transpose the matrix to work with rows
            Float4 r0 = c0, r1 = c1, r2 = c2, r3 = c3;
            Transpose4(r0, r1, r2, r3);

            // 2. 2x2 Sub-determinants
            Float4 v0 = Cross(r0, r1);
            Float4 v1 = Cross(r2, r3);

            // --- Vectorized Adjugate Matrix --
            // NEON doesn't support Intel's specific hex shuffles, so it uses its superior hardware Fused-Multiply-Add to evaluate the sub-determinants.
            Float4 adj0 = Sub(Mul(r1, Set1(ExtractX(v1))), Mul(r2, Set1(ExtractY(v1))));
            adj0 = Add(adj0, Mul(r3, Set1(ExtractZ(v1))));

            Float4 adj1 = Sub(Mul(r2, Set1(ExtractX(v0))), Mul(r3, Set1(ExtractY(v0))));
            adj1 = Add(adj1, Mul(r0, Set1(ExtractZ(v0))));

            Float4 adj2 = Sub(Mul(r3, Set1(ExtractX(v0))), Mul(r0, Set1(ExtractY(v0))));
            adj2 = Add(adj2, Mul(r1, Set1(ExtractZ(v0))));

            Float4 adj3 = Sub(Mul(r0, Set1(ExtractX(v1))), Mul(r1, Set1(ExtractY(v1))));
            adj3 = Add(adj3, Mul(r2, Set1(ExtractZ(v1))));

            // 3. Calculate Determinant (Dot product of Row 0 and Adjugate Column 0)
            float det = Dot4(r0, adj0);

             // If det is 0, the matrix is singular (cannot be inverted). We just return Identity to prevent NaN explosions.
            if (Engine::Math::Functions::abs(det) < 1e-8f) {
                c0 = Set(1.0f, 0.0f, 0.0f, 0.0f); c1 = Set(0.0f, 1.0f, 0.0f, 0.0f);
                c2 = Set(0.0f, 0.0f, 1.0f, 0.0f); c3 = Set(0.0f, 0.0f, 0.0f, 1.0f);
                return;
            }

            // 4. Divide by Determinant (Multiply by Inverse)
            Float4 invDet = Set1(1.0f / det);
            c0 = Mul(adj0, invDet); c1 = Mul(adj1, invDet);
            c2 = Mul(adj2, invDet); c3 = Mul(adj3, invDet);

            // 5. Final Transpose back to Column-Major
            Transpose4(c0, c1, c2, c3);
        }

        // ARM NEON Absolute Value and Square Root
        FORCE_INLINE Float4 Abs(Float4 v) { return vabsq_f32(v); } // NEON has a dedicated absolute value instruction!
        FORCE_INLINE Float4 Sqrt(Float4 v) { return vsqrtq_f32(v); }

        // --- SIMD HARDWARE FLOOR ---
        FORCE_INLINE Float4 FastFloor(Float4 x) { return vrndmq_f32(x); } // ARM NEON Round to Minus Infinity

        // --- ARM Hardware Reciprocal (Estimate + Newton-Raphson Step) ---
        FORCE_INLINE Float4 Reciprocal(Float4 v) {
            Float4 rec = vrecpeq_f32(v);                // Initial hardware precision estimate 
            return vmulq_f32(rec, vrecpsq_f32(v, rec)); // Refinement step for accuracy (i.e., doubles precision to match standard IEEE-754 accuracy).
        }

        // --- ARM Hardware Reciprocal Square Root (Estimate + Newton-Raphson Step) ---
        FORCE_INLINE Float4 ReciprocalSqrt(Float4 v) {
            Float4 est = vrsqrteq_f32(v);                // Initial hardware estimate
            return vmulq_f32(est, vrsqrtsq_f32(v, est)); // Newton-Raphson refinement
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

        // =========================
        // --- SIMD PACKER (SET) ---
        // =========================
        /*  
            - Takes 4 distinct, separate values (or scalars) and packs them together into a single 128-bit wide vector register. 
            - e.g., used when constructing positions, quaternions, or matrices where every lane stores a different piece of data.
            - Memory layout in the hardware register:

              [ 1.5f | 2.0f | 3.5f | 1.0f ]
                (X)    (Y)    (Z)    (W)
        */ 
        
        FORCE_INLINE Float4 Set(float x, float y, float z, float w) { return _mm_set_ps(w, z, y, x); } // Note: _mm_set_ps takes arguments in reverse order (w, z, y, x). 

        // ===============================
        // --- SIMD BROADCASTER (SET1) ---
        // ===============================
        /*  
            - Takes a single scalar value and copies (or broadcasts) it across all four lanes of the 128-bit wide vector register simultaneously.
            - Forces CPU to broadcast (or clone) the FPU scalar across the 128-bit vector register natively, avoiding memory entirely
            - e.g., used when you need to perform uniform math on an entire vector (i.e., scalar multiplication, uniform offsets, and FMA constants).
            - Memory layout in the hardware register:

              [ 5.0f | 5.0f | 5.0f | 5.0f ]
                (X)    (Y)    (Z)    (W)
        */ 
        FORCE_INLINE Float4 Set1(float v) { return _mm_set1_ps(v); } // Broadcasts scalar to all 4 slots.       
        
        // ==============================
        // --- MATHEMATICAL OPERATORS ---
        // ==============================
        // By returning by value, the compiler uses Return Value Optimization (RVO).
        // The data never touches the stack; it stays perfectly inside the CPU registers.
        FORCE_INLINE Float4 Add(Float4 a, Float4 b) { return _mm_add_ps(a, b); }
        FORCE_INLINE Float4 Sub(Float4 a, Float4 b) { return _mm_sub_ps(a, b); }
        FORCE_INLINE Float4 Mul(Float4 a, Float4 b) { return _mm_mul_ps(a, b); }

        // _mm_fmadd_ps: requirees FMA3 instructions, which is part of AVX2, supported by almost all CPUs made after 2013.
        FORCE_INLINE Float4 FMAdd(Float4 a, Float4 b, Float4 c) { return _mm_fmadd_ps(a, b, c); } // (a * b) + c

    
        // Mask 0x08 (binary 1000) tells the hardware: "Take X, Y, Z from 'reg', take W from the zero vector."
        FORCE_INLINE Float4 BlendMaskW(Float4 a, Float4 b) { return _mm_blend_ps(a, b, 0x08); }

        FORCE_INLINE int MoveMask(Float4 v) { return _mm_movemask_ps(v); }

        // ======================================================================
        // HARDWARE SETTERS (SSE4.1)
        // ======================================================================
        // Allows individual lane mutations without spilling the register to the stack.
        // _mm_insert_ps takes the source value and inserts it into a specific lane by using a bitmask (0x10, 0x20, 0x30) to target Y, Z, and W.
        FORCE_INLINE Float4 InsertX(Float4 v, float val) { return _mm_move_ss(v, _mm_set_ss(val)); }          // _mm_move_ss replaces the lowest 32-bits (X) safely and keeps the high 96-bits intact.
        FORCE_INLINE Float4 InsertY(Float4 v, float val) { return _mm_insert_ps(v, _mm_set_ss(val), 0x10); }  // 0x10 = Source Index 0, Destination Index 1 (Y)
        FORCE_INLINE Float4 InsertZ(Float4 v, float val) { return _mm_insert_ps(v, _mm_set_ss(val), 0x20); }  // 0x20 = Source Index 0, Destination Index 2 (Z)
        FORCE_INLINE Float4 InsertW(Float4 v, float val) { return _mm_insert_ps(v, _mm_set_ss(val), 0x30); }  // 0x30 = Source Index 0, Destination Index 3 (W)

        // ======================================================================
        // HARDWARE GETTERS (ZERO MEMORY ACCESS)
        // ======================================================================
        // Extracts the float directly from the XMM register. 
        // _mm_cvtss_f32: faster than _mm_store_ss to a stack variable
        FORCE_INLINE float ExtractX(Float4 v) { return _mm_cvtss_f32(v); } // X is in the lowest 32 bits, so we just convert scalar.                                               
        FORCE_INLINE float ExtractY(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1))); }  // Y requires a 1-cycle shuffle to move to lowest 32 bits before extraction.
        FORCE_INLINE float ExtractZ(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2))); }  // Z requires a 2-cycle shuffle to move to lowest 32 bits before extraction.
        FORCE_INLINE float ExtractW(Float4 v) { return _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3))); }  // W requires a 3-cycle shuffle to move to lowest 32 bits before extraction.

        // x86 uses shuffles to broadcast lanes without leaving the XMM registers
        FORCE_INLINE Float4 BroadcastX(Float4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)); }
        FORCE_INLINE Float4 BroadcastY(Float4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)); }
        FORCE_INLINE Float4 BroadcastZ(Float4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)); }
        FORCE_INLINE Float4 BroadcastW(Float4 v) { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)); }

        // ======================================
        // DOT PRODUCT INSTRUCTION (_mm_dp_ps)
        // ======================================
        /*
            - On modern Intel/AMD architectures, _mm_dp_ps is implemented in slow microcode (~9-14 clock cycles).
            - It tries to do too many things (multiply, horizontal add, and mask) simultaneously.
            - Hogs CPU execution ports because the silicilon has to internally decode it into a sequence of multiplies, adds, and bitwise masks.
            - 0x7F mask: 0111 calculates dots for first 3 elements | 1111 (write to all 4 for safety, or 0001 for just lowest).
        */

        FORCE_INLINE float Dot4_mm_dp_ps(Float4 a, Float4 b) { return _mm_cvtss_f32(_mm_dp_ps(a, b, 0xFF)); }
        FORCE_INLINE float Dot3_mm_dp_ps(Float4 a, Float4 b) { return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x7F)); } 

        // ===========================================
        // DOT PRODUCT (MANUAL HORIZONTAL REDUCTION)
        // ===========================================
        /*
            - _mm_dp_ps(reg, other.reg, 0xFF) is a trap for performance because it tries to do too many things in one shot.
            - It bundles 3 operations into one instruction.

              1. Four float multiplications.
              2. Multiple horizontal additions.
              3. Apply a bitwise mask (0xFF).

            - Bypass the _mm_dp_ps hardware trap by writing these instructions separately (i.e., fast manual reduction).
            - Operates on separate execution ports and can overlap (i.e., interleave) these manual instructions.
            - Separate the mathematical operations and run them back to back manually to have the CPU interleave the operation pipelines to result in a much higher throughput (+2x performance boost).
            - FMA/shuffle reduction will yield significant performance boost in heavy vector math operations (i.e., dot product).
            
              [_mm_mul_ps]      ~4 cycles
              [_mm_movehl_ps]   ~1 cycle
              [_mm_shuffle_ps]  ~1 cycle
              [_mm_add_ps]      ~3-4 cycles
              [_mm_add_ss]      ~3-4 cycles
        */

        FORCE_INLINE float Dot3(Float4 a, Float4 b) { 
            // 1. Multiply the vectors 
            __m128 mul = _mm_mul_ps(a, b);
            
            // 2. We only want X, Y, and Z. Force W to 0.0f before the horizontal add!
            mul = _mm_blend_ps(mul, _mm_setzero_ps(), 0x08);

            // 3. Fast Horizontal Sum (since w = 0.0f, we can safely horizontal sum all 4 lanes)
            __m128 shuf = _mm_movehl_ps(mul, mul); 
            mul = _mm_add_ps(mul, shuf);           
            shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)); 
            mul = _mm_add_ss(mul, shuf);           
            return _mm_cvtss_f32(mul);
        }

        FORCE_INLINE float Dot4(Float4 a, Float4 b) { 
            // 1. Multiply the vectors 
            __m128 mul = _mm_mul_ps(a, b);

            // 2. Fast Horizontal Sum all 4 lanes
            __m128 shuf = _mm_movehl_ps(mul, mul); 
            mul = _mm_add_ps(mul, shuf);           
            shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)); 
            mul = _mm_add_ss(mul, shuf);           
            return _mm_cvtss_f32(mul);
        }

        // SHUFFLE: Rearranges the (x, y, z) values inside the register, so we can multiply them all at once.
        FORCE_INLINE Float4 Cross(Float4 a, Float4 b) {
            __m128 tmp0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
            __m128 tmp1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 tmp2 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
            __m128 tmp3 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
            return _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
        }

        // =================================
        // --- HARDWARE MATRIX TRANSPOSE ---
        // =================================
        /*
            - Rendering APIs (OpenGL, Vulkan) require matrices to be formatted in Column-Major order.
            - Instead of extracting 16 floats sequentially to memory to flip rows into columns, we use pure register unpacking and moves.
            - This fuction mirrors Intel's _MM_TRANSPOSE4_PS(r0, r1, r2, r3) macros, but encapsulates it safely.
        */

         // Guarantees compiler properly tracks variable scopes and passed in references that were modified (i.e., ensures compiler tracks the mutations correctly).
        FORCE_INLINE void Transpose4(/*Right Vector*/Float4& r0, /*Up Vector*/Float4& r1, /*Forward Vector*/Float4& r2, /*Translation*/Float4& r3) {
            // _mm_unpacklo_ps takes lower two floats from both registers and zips them together (interleaved). 
            Float4 tmp0 = _mm_unpacklo_ps(r0, r1); // r0 = (x0, y0, z0, w0), r1 = (x1, y1, z1, w1) -> tmp0 = [x0, x1, y0, y1]
            Float4 tmp2 = _mm_unpacklo_ps(r2, r3); // r2 = (x2, y2, z2, w2), r3 = (x3, y3, z3, w3) -> tmp2 = [x2, x3, y2, y3]

            // _mm_unpackhi_ps takes upper two floats from both registers and zips them together.
            Float4 tmp1 = _mm_unpackhi_ps(r0, r1); // r0 = (x0, y0, z0, w0), r1 = (x1, y1, z1, w1) -> tmp1 = [z0, z1, w0, w1]
            Float4 tmp3 = _mm_unpackhi_ps(r2, r3); // r2 = (x2, y2, z2, w2), r3 = (x3, y3, z3, w3) -> tmp3 = [z2, z3, w2, w3]
 
            // 1. _mm_movelh_ps (Move Low to High) copies lower 64-bits (Low bits) of the second parameter (source register) into the upper 64-bits (High bits) of the first (destination register).
            r0 = _mm_movelh_ps(tmp0, tmp2);  // [x0, x1] from tmp0, [x2, x3] from tmp2 -> r0 = [x0, x1, x2, x3] (Column 0)
            r2 = _mm_movelh_ps(tmp1, tmp3);  // [z0, z1] from tmp1, [z2, z3] from tmp3 -> r2 = [z0, z1, z2, z3] (Column 2)

            // _mm_movehl_ps (Move High to Low) copies upper 64-bits (High bits) of the second parameter (source register) into lower 64-bits (Low bits) of the first (destination register).
            r1 = _mm_movehl_ps(tmp2, tmp0);  // High of tmp2 [y2, y3] into low, High of tmp0 [y0, y1] kept -> r1 = [y0, y1, y2, y3] (Column 1)
            r3 = _mm_movehl_ps(tmp3, tmp1);  // High of tmp3 [w2, w3] into low, High of tmp1 [w0, w1] kept -> r3 = [w0, w1, w2, w3] (Column 3)
        }

        // Conjugate: Flip the sign bit of 3 floats (x, y, z) instantly using XOR (-x, -y, -z, w), but leaves w alone (i.e., inverse rotation).
        FORCE_INLINE Float4 FlipSignXYZ(Float4 v) {
            __m128 signMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000));
            return _mm_xor_ps(v, signMask);
        }

        // Shortest Path: Flip the sign bit of all 4 floats instantly using XOR (-x, -y, -z, -w) to represent the same 3D orientation in space, but sit on opposite poles of the 4D hypersphere's surface.
        FORCE_INLINE Float4 Negate(Float4 v) { 
            return _mm_xor_ps(v, Engine::Math::Constants::SIMD_NEG_ZERO); // Instant XOR, zero transition
        }

        // Q1 = this (a, b, c, d) | Q2 = rhs (x, y, z, w)
        FORCE_INLINE Float4 QuaternionMul(Float4 q1, Float4 q2) {
            // Shuffle Q1
            __m128 w1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 3, 3, 3));
            __m128 x1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 y1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(2, 2, 2, 2));

            // Shuffle Q2 for the specific Hamilton cross-terms
            __m128 tmp0 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 0));  // w, z, y, x
            __m128 tmp1 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(2, 3, 0, 1));  // z, w, x, y
            __m128 tmp2 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(1, 0, 3, 2));  // y, x, w, z

            // FMA (Fused Multiply-Add/Sub) sequence to resolve the complex numbers
            __m128 res = _mm_mul_ps(w1, q2);
            
            // We use bitwise XOR to flip the signs for the subtraction terms in the Hamilton formula
            __m128 signX = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0, 0x80000000));
            __m128 signY = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0x80000000));
            __m128 signZ = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0));

            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(x1, tmp0), signX));
            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(y1, tmp1), signY));
            res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(z1, tmp2), signZ));

            return res;
        }

        // --- INTEL SSE MATRIX INVERSE (AP-928 Cramer's Rule Unroll) ---
        // Uses parallel hex shuffles to calculate 2x2 sub-determinants in registers.
        FORCE_INLINE void Inverse4x4(Float4& c0, Float4& c1, Float4& c2, Float4& c3) {
            __m128 tmp1, row0, row1, row2, row3;
            __m128 minor0, minor1, minor2, minor3;
            __m128 det;

            // 1. Transpose the matrix
            tmp1 = _mm_unpacklo_ps(c0, c1);
            __m128 tmp2 = _mm_unpacklo_ps(c2, c3);
            __m128 tmp3 = _mm_unpackhi_ps(c0, c1);
            __m128 tmp4 = _mm_unpackhi_ps(c2, c3);
            row0 = _mm_movelh_ps(tmp1, tmp2);
            row1 = _mm_movehl_ps(tmp2, tmp1);
            row2 = _mm_movelh_ps(tmp3, tmp4);
            row3 = _mm_movehl_ps(tmp4, tmp3);

            // 2. 2x2 Sub-determinants & Minors via parallel shuffles
            tmp1 = _mm_mul_ps(row2, row3);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            minor0 = _mm_mul_ps(row1, tmp1);
            minor1 = _mm_mul_ps(row0, tmp1);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
            minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
            minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

            tmp1 = _mm_mul_ps(row1, row2);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
            minor3 = _mm_mul_ps(row0, tmp1);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
            minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
            minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

            tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            row2 = _mm_shuffle_ps(row2, row2, 0x4E);
            minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
            minor2 = _mm_mul_ps(row0, tmp1);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
            minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
            minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

            tmp1 = _mm_mul_ps(row0, row1);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
            minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row3, tmp1));
            minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

            tmp1 = _mm_mul_ps(row0, row3);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
            minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
            minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

            tmp1 = _mm_mul_ps(row0, row2);
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
            minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
            minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
            tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
            minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
            minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

            // 3. Calculate Determinant
            det = _mm_mul_ps(row0, minor0);
            det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
            det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
            
            // Hardware Reciprocal to divide by determinant
            tmp1 = _mm_rcp_ss(det);
            det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
            det = _mm_shuffle_ps(det, det, 0x00);

            // 4. Multiply Minor by Inverse Determinant
            c0 = _mm_mul_ps(det, minor0);
            c1 = _mm_mul_ps(det, minor1);
            c2 = _mm_mul_ps(det, minor2);
            c3 = _mm_mul_ps(det, minor3);
        }

        FORCE_INLINE Float4 CmpGt(Float4 a, Float4 b) { return _mm_cmpgt_ps(a, b); }
        FORCE_INLINE Float4 CmpLt(Float4 a, Float4 b) { return _mm_cmplt_ps(a, b); }
        FORCE_INLINE Float4 BitwiseAnd(Float4 a, Float4 b) { return _mm_and_ps(a, b); }
        FORCE_INLINE Float4 BlendVariable(Float4 a, Float4 b, Float4 mask) { return _mm_blendv_ps(a, b, mask); }

        // Intel SSE Absolute Value (Bitwise clear of the sign bit) and Square Root
        FORCE_INLINE Float4 Abs(Float4 v) { 
            __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
            return _mm_and_ps(v, absMask); 
        }

        FORCE_INLINE Float4 Sqrt(Float4 v) { return _mm_sqrt_ps(v); }

        // --- SIMD HARDWARE FLOOR (FPU) --- 
        FORCE_INLINE Float4 FastFloor(Float4 x) { return _mm_floor_ps(x); }

        // ==================================================================
        // HARDWARE RECIPROCAL (NEWTON-RAPHSON)
        // ==================================================================
        /*
            - Need Newton-Raphson iteration to provide stable rendering (no jitter) at extreme distances.
            - Is vastly faster than standard division (~10-15 clock cycles) due to interleaving of instructions.
            - Note: this is for AOS layouts, SOA layouts use _mm256_rsqrt_ps.
        */

        // --- Intel Hardware Reciprocal (Estimate + Newton-Raphson Step) ---
        FORCE_INLINE Float4 Reciprocal(Float4 v) {
            // 1. Initial hardware estimate (~11 bits of precision)
            __m128 est = _mm_rcp_ps(v); 

            // 2. Newton-Raphson iteration: est * (2.0 - v * est)
            // Doubles precision to full 22-24 bit IEEE-754 standard accuracy.
            __m128 v_est = _mm_mul_ps(v, est);
            __m128 two_minus = _mm_sub_ps(Engine::Math::Constants::SIMD_TWO, v_est);
            
            return _mm_mul_ps(est, two_minus);
        }

        // --- Intel Hardware Reciprocal Square Root (Estimate + Newton-Raphson Step) ---
        FORCE_INLINE Float4 ReciprocalSqrt(Float4 v) {
            // Initial hardware estimate (~11 bits precision)
            __m128 est = _mm_rsqrt_ps(v); 

            // Newton-Raphson iteration: 0.5 * est * (3.0 - v * est * est)
            __m128 est_squared = _mm_mul_ps(est, est);
            __m128 half_est = _mm_mul_ps(est, Engine::Math::Constants::SIMD_HALF);
            __m128 three_minus_v_est2 = _mm_sub_ps(Engine::Math::Constants::SIMD_THREE, _mm_mul_ps(v, est_squared));
            
            return _mm_mul_ps(half_est, three_minus_v_est2);
        }
    #endif

    // ======================================================================
    // TRANSCENDENTAL RATIONAL APPROXIMATION (TANGENT)
    // ======================================================================
    /*
        - Pure SIMD Rational Approximation of Tangent using a rational function, which divides one polynomial by another.
        - Highly accurate within [-PI/4, PI/4]. 
        - Automatically uses fast hardware division (rcp).
    */

    FORCE_INLINE Float4 FastTan(Float4 x) {
        Float4 x2 = Mul(x, x); // x^2

        // Numerator Polynomial: 1.0 + x^2 * (C1 + x^2 * C2)
        Float4 num = FMAdd(x2, Engine::Math::Constants::TAN_C2, Engine::Math::Constants::TAN_C1); // C2 * x^2 + C1
        num = FMAdd(num, x2, Engine::Math::Constants::TAN_C0);           // num * x^2 + 1.0

        // Denominator Polynomial: 1.0 + x^2 * (C4 + x^2 * C5)
        Float4 den = FMAdd(x2, Engine::Math::Constants::TAN_C5, Engine::Math::Constants::TAN_C4); // C5 * x^2 + C4
        den = FMAdd(den, x2, Engine::Math::Constants::TAN_C3);           // den * x^2 + 1.0

        // 3. Perform cross-platform hardware reciprocal (1 / den) to avoid slow CPU division
        // ARM NEON Fast Reciprocal Approximation, Newton-Raphson refinement step
        Float4 rec = Reciprocal(den);

        // Result = x * (Numerator * (1 / Denominator))
        Float4 res = Mul(num, rec);
        return Mul(res, x);
    }

    // ======================================================================
    // TRANSCENDENTAL APPROXIMATION (TAYLOR/MINIMAX POLYNOMIAL EXPANSION)
    // ======================================================================
    /*
        [std::sine]: ~50 cycles

        - Use Horner's Method to factor the polynomial, which minimizes the number of multiplications and allows the CPU to pipeline the math perfectly.

          Horner's Method: x * (1 + x^2 * (C3 + x^2 * (C5 + x^2 * (C7 + x^2 * C9))))

        - Calculates the sine of 4 separate angles simultaneously in ~20 cycles.
        - Drastically improves performance compared to [std::sine].
    */
    // Pure SIMD 9th-Order Polynomial Approximation of Sine.
    // Valid for angles in radians between [-PI, PI]. Extremely fast.
    FORCE_INLINE Float4 FastSin(Float4 x) {
        Float4 x2 = Mul(x, x); // x^2

        // 2. Execute Horner's Method purely between existing SIMD registers
        Float4 res = FMAdd(x2, Engine::Math::Constants::SIN_C9, Engine::Math::Constants::SIN_C7); // C9  * x^2 + C7
        res = FMAdd(res, x2, Engine::Math::Constants::SIN_C5);           // res * x^2 + C5
        res = FMAdd(res, x2, Engine::Math::Constants::SIN_C3);           // res * x^2 + C3
        res = FMAdd(res, x2, Engine::Math::Constants::SIN_1);            // res * x^2 + 1.0                      

        return Mul(res, x); // Final multiplication by x
    }

    // ======================================================================
    // TRANSCENDENTAL APPROXIMATION (POLYNOMIAL APPROXIMATION (Arccosine))
    // ======================================================================
    /*
        - Derived from Nvidia CG Toolkit.
        - Max absolute error: ~0.0001 radians.
    */
    FORCE_INLINE Float4 FastACos(Float4 x) {
        // Absolute value of x
        Float4 absX = Abs(x);

        // Horner's Method: C0 + x * (C1 + x * (C2 + x * C3))
        Float4 res = FMAdd(absX, Engine::Math::Constants::ACOS_C3, Engine::Math::Constants::ACOS_C2);  // C3  * x + C2
        res = FMAdd(res, absX, Engine::Math::Constants::ACOS_C1);             // res * x + C1
        res = FMAdd(res, absX, Engine::Math::Constants::ACOS_C0);             // res * x + C0 (Pi/2)

        // 3. Sqrt and blend [res = (res * sqrt(1.0 - absX))]
        Float4 oneMinusX = Sub(Engine::Math::Constants::ACOS_1, absX);
        Float4 sqrtOneMinusX = Sqrt(oneMinusX);
        res = Mul(res, sqrtOneMinusX);

        // If x is negative, the result is Pi - res
        Float4 resNeg = Sub(Engine::Math::Constants::ACOS_PI, res);

        // Blend based on original sign of x, if x < 0.0, use (Pi - res), else use res
        Float4 cmpLtZero = CmpLt(x, Zero());
        return BlendVariable(res, resNeg, cmpLtZero);
    }

    // ======================================================================
    // Pure SIMD 8th-Order Polynomial Approximation of Cosine.
    // ======================================================================
    FORCE_INLINE Float4 FastCos(Float4 x) {
        Float4 x2 = Mul(x, x); // x^2

        // Horner's Method: 1 + x^2 * (C2 + x^2 * (C4 + x^2 * (C6 + x^2 * C8)))
        Float4 res = FMAdd(x2, Engine::Math::Constants::COS_C8, Engine::Math::Constants::COS_C6); // C8  * x^2 + C6
        res = FMAdd(res, x2, Engine::Math::Constants::COS_C4);           // res * x^2 + C4
        res = FMAdd(res, x2, Engine::Math::Constants::COS_C2);           // res * x^2 + C2
        res = FMAdd(res, x2, Engine::Math::Constants::COS_1);            // res * x^2 + 1.0                      

        return res;
    }

    // ======================================================================
    // COMBINED SIMD TRANSCENDENTAL APPROXIMATION (SINE & COSINE)
    // ======================================================================
    /*
        - Evaluates both Sine and Cosine simultaneously across all lanes.
        - Uses Cody-Waite range reduction to safely process massive coordinates.
    */
    FORCE_INLINE std::pair<Float4, Float4> FastSinCos(Float4 x) {
        // 1. Range Reduction to [-PI, PI] across all 4 lanes
        Float4 cycles = FastFloor(Add(Mul(x, Engine::Math::Constants::INV_TWO_PI), Engine::Math::Constants::SIMD_HALF));

        // Cody-Waite FMA subtraction: x = (x - (cycles * TWO_PI_A)) - (cycles * TWO_PI_B)
        x = Sub(Sub(x, Mul(cycles, Engine::Math::Constants::TWO_PI_A)), Mul(cycles, Engine::Math::Constants::TWO_PI_B));

        // 2. Evaluate both Sine and Cosine simultaneously (Sharing x^2)
        Float4 x2 = Mul(x, x);

        // Sine Evaluation (Horner's Method)
        Float4 s = FMAdd(x2, Engine::Math::Constants::SIN_C9, Engine::Math::Constants::SIN_C7);
        s = FMAdd(s, x2, Engine::Math::Constants::SIN_C5);
        s = FMAdd(s, x2, Engine::Math::Constants::SIN_C3);
        s = FMAdd(s, x2, Engine::Math::Constants::SIN_1);
        Float4 outSin = Mul(s, x);

        // Cosine Evaluation (Horner's Method)
        Float4 c = FMAdd(x2, Engine::Math::Constants::COS_C8, Engine::Math::Constants::COS_C6);
        c = FMAdd(c, x2, Engine::Math::Constants::COS_C4);
        c = FMAdd(c, x2, Engine::Math::Constants::COS_C2);
        c = FMAdd(c, x2, Engine::Math::Constants::COS_1);
        Float4 outCos = c;

        return {outSin, outCos};
    }
} // namespace Engine::Math::SIMD

// ==================================================================================
// PURE SCALAR VECTORS (NON-SIMD)
// ==================================================================================
/*
    - Used for one-off calculations like the camera and compile time calculations.
    - Pure Scalar Vectors execute math operations in one lane.
    - "float x, y, z, w" and "float data[4]" are treated the same by the compiler in memory except for syntax (data[0] vs x).
    - The w only matters when multiplying a vector with a matrix.
    - w = 0.0f (Direction): Represents a vector (like gravity or camera's forward axis). When multiplied by a matrix it ignores translation (i.e., you cannot move gravity).
    - w = 1.0f (Point): Represents a position in space (like a player or vertex). When multiplied by a matrix, the translation is applied.
*/
class Vector3D {
public:
    // Point     [w = 1]: (apply  translation)
    // Direction [w = 0]: (ignore translation)
    float x, y, z, w; // 16-bytes in memory

    // Constructor: Default initializes to {0.0f, 0.0f, 0.0f, 0.0f}
    constexpr Vector3D(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f) 
        : x(x), y(y), z(z), w(w) {}

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
        return Engine::Math::Functions::FastSqrt(lengthSquared());
    }

    FORCE_INLINE void Normalize() {
        float lenSq = lengthSquared();
        if (lenSq > 1e-8f) {
            *this = *this * (1.0f / Engine::Math::Functions::FastSqrt(lenSq));
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
// 128-BIT SIMD ACCELERATED VECTORS (CROSS-PLATFORM: SSE, ARM)
// ==================================================================================
/* 
    - SIMD Accelerated Vector that needs to be used for bulk data processing.
    - SIMD Vectors must perform math operations on all its available lanes (4, 8, 16) simultaneously in hardware registers.
    - alignas(16) forces every instance of this class to align to 16 bytes (This tells the compiler: "Every instance of this class must start at a 16-byte boundary in memory").
    - Guarantees that whenever this struct is created, it starts on a 16-byte boundary.

      // Usage is clean and readable
      SIMDVector3D a(1.0f, 0.0f, 0.0f);
      SIMDVector3D b(0.0f, 1.0f, 0.0f);
      SIMDVector3D c = a + (b * 5.0f); // Completely optimized into registers by the compiler
*/
class alignas(16) SIMDVector3D {
public:
    Engine::Math::SIMD::Float4 reg; // 4 (32-bit) floats = 128-bits

    // Default constructor (Zero initialization)
    FORCE_INLINE SIMDVector3D() : reg(Engine::Math::SIMD::Zero()) {}

    // Constructor from floats
    FORCE_INLINE SIMDVector3D(float _x, float _y, float _z, float _w = 0.0f) 
        : reg(Engine::Math::SIMD::Set(_x, _y, _z, _w)) {}

    // Constructor directly from native register (Crucial for fast operators)
    FORCE_INLINE SIMDVector3D(Engine::Math::SIMD::Float4 m) : reg(m) {}

    // --- HARDWARE SETTERS ---
    // Allows individual lane mutations without spilling the register to the stack (Avoids Load-Hit-Store penalty).
    FORCE_INLINE void setX(float val) { reg = Engine::Math::SIMD::InsertX(reg, val); }
    FORCE_INLINE void setY(float val) { reg = Engine::Math::SIMD::InsertY(reg, val); }
    FORCE_INLINE void setZ(float val) { reg = Engine::Math::SIMD::InsertZ(reg, val); }
    FORCE_INLINE void setW(float val) { reg = Engine::Math::SIMD::InsertW(reg, val); }

    // --- HARDWARE GETTERS ---
    // Disabled implicit conversion to prevent automatic casting SIMDVector3D to Vector3D, so we must explicitly extract scalars to ensure the programmer is 100% aware of when data is leaving the fast lanes (VU, SIMD register) and entering the slow lanes (FPU, Scalar register).
    FORCE_INLINE float x() const { return Engine::Math::SIMD::ExtractX(reg); } 
    FORCE_INLINE float y() const { return Engine::Math::SIMD::ExtractY(reg); } // Y requires a 1-cycle shuffle to move to lowest 32 bits before extraction.
    FORCE_INLINE float z() const { return Engine::Math::SIMD::ExtractZ(reg); } // Z requires a 2-cycle shuffle to move to lowest 32 bits before extraction.
    FORCE_INLINE float w() const { return Engine::Math::SIMD::ExtractW(reg); } // W requires a 3-cycle shuffle to move to lowest 32 bits before extraction.

    // ======================================================================
    // C++20 BRIDGE (ZERO-COST MEMORY)
    // ======================================================================
    // It guarantees zero-overhead assembly.
    FORCE_INLINE std::array<float, 4> asArray() const {
        // If you need to interface with OpenGL/Vulkan APIs or loop through the vector like an array, std::bit_cast is perfectly standard compliant.
        return std::bit_cast<std::array<float, 4>>(reg);
    }

    // --- AXIS INDEXING ---
    // Safely extracts X (0), Y (1), Z (2), or W (3) dynamically without breaking strict aliasing.
    FORCE_INLINE float operator[](int axis) const {
        // We use std::bit_cast (C++20) to treat the register as a safe array entirely on the stack, allowing dynamic indexing without UB.
        auto arr = std::bit_cast<std::array<float, 4>>(reg);
        return arr[axis];
    }

    // --- SIMD LINEAR INTERPOLATION (V = a + t * (b - a)) ---
    static FORCE_INLINE SIMDVector3D Lerp(const SIMDVector3D& a, const SIMDVector3D& b, float t) {
        // 1. Broadcast the scalar 't' across all 4 lanes of a register
        Engine::Math::SIMD::Float4 tReg = Engine::Math::SIMD::Set1(t);
        // 2. Calculate the difference: (b - a)
        Engine::Math::SIMD::Float4 diff = Engine::Math::SIMD::Sub(b.reg, a.reg);
        // 3. Fused Multiply-Add: diff * t + a
        return SIMDVector3D(Engine::Math::SIMD::FMAdd(diff, tReg, a.reg));
    }

    // ==========================================
    // --- STANDARD MATHEMATICAL OPERATORS ---
    // ==========================================

    // --- MATHEMATICAL OPERATORS ---
    FORCE_INLINE SIMDVector3D operator+(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Add(reg, other.reg)); } // Addition: result = this + other
    FORCE_INLINE SIMDVector3D operator-(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Sub(reg, other.reg)); } // Subtraction: result = this - other
    FORCE_INLINE SIMDVector3D operator*(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Mul(reg, other.reg)); } // Vector Multiplication (Component-wise / Hadamard Product): result = this * other, useful for scaling a vector by a non-uniform scale (e.g., scaling an object's X axis by 2, but its Y axis by 5).                          
    FORCE_INLINE SIMDVector3D operator*(float scalar) const { return SIMDVector3D(Engine::Math::SIMD::Mul(reg, Engine::Math::SIMD::Set1(scalar))); } // Scalar Multiplication: returns a brand new scaled copy of the vector.

    // ==========================================
    // --- IN-PLACE ASSIGNMENT OPERATORS ---
    // ==========================================
    
    // Adds another vector to this vector in place
    FORCE_INLINE SIMDVector3D& operator+=(const SIMDVector3D& other) {
        reg = Engine::Math::SIMD::Add(reg, other.reg);
        return *this;
    }

    // Subtracts another vector from this vector in place
    FORCE_INLINE SIMDVector3D& operator-=(const SIMDVector3D& other) {
        reg = Engine::Math::SIMD::Sub(reg, other.reg);
        return *this;
    }

    // Multiplies another vector component-wise in place
    FORCE_INLINE SIMDVector3D& operator*=(const SIMDVector3D& other) { 
        reg = Engine::Math::SIMD::Mul(reg, other.reg); 
        return *this;
    }

    // Adds a uniform scalar to all components in place
    FORCE_INLINE SIMDVector3D& operator+=(float scalar) {
        reg = Engine::Math::SIMD::Add(reg, Engine::Math::SIMD::Set1(scalar));
        return *this;
    }

    // Subtracts a uniform scalar from all components in place
    FORCE_INLINE SIMDVector3D& operator-=(float scalar) {
        reg = Engine::Math::SIMD::Sub(reg, Engine::Math::SIMD::Set1(scalar));
        return *this;
    }

    // Scales all components by a uniform scalar in place
    FORCE_INLINE SIMDVector3D& operator*=(float scalar) { 
        reg = Engine::Math::SIMD::Mul(reg, Engine::Math::SIMD::Set1(scalar)); 
        return *this;
    }
    
    // --- DOT & CROSS PRODUCT ---

    // Dot Product: returns (x1*x2 + y1*y2 + z1*z2)
    FORCE_INLINE float dot(const SIMDVector3D& other) const { return Engine::Math::SIMD::Dot3(reg, other.reg); }
    FORCE_INLINE SIMDVector3D cross(const SIMDVector3D& other) const { return SIMDVector3D(Engine::Math::SIMD::Cross(reg, other.reg)); }

    // --- HOMOGENEOUS COORDINATE ENFORCEMENT ---
    /*
        - Allow addition, cross product, and dot products to generate garbage in the W lane (let the math be dirty). 
        - The w component does not really matter for most vector operations.
        - Clean it at the boundary where w actually matters (i.e., when multiplying a vector [SIMDVector3D] by a matrix (Matrix4x4_SIMD)).
        - Force it to either a Point [w = 1, which applies translation], or a Direction [w = 0, which ignores translation] right before the multiplication.
        - We blend our register with a vector containing 1.0f in the W lane [_mm_set_ps takes (W, Z, Y, X)].
    */

    // Blend in a 0.0f to the W lane (mask 0x08 = 1000 binary)! Forces W = 0.0f (Treats the vector as a Direction/Normal)
    FORCE_INLINE SIMDVector3D asDirection() const { return SIMDVector3D(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::SIMD::Zero())); }
    
    // Blend in a 1.0f to the W lane (mask 0x08 = 1000 binary)! Forces W = 1.0f (Treats the vector as a Position/Point in space)
    FORCE_INLINE SIMDVector3D asPoint() const { 
        return SIMDVector3D(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::Constants::SIMD_MASK_W_ONE)); // [0.0f, 0.0f, 0.0f, 0.0f] pre-baked in ultra-fast L1 cache for fast operations.
    }

    // --- MAGNITUDE & NORMALIZATION ---
    FORCE_INLINE float lengthSquared() const {
        return dot(*this);
    }

    // ======================================
    // VECTOR MAGNITUDE (LENGTH)
    // ======================================
    /*
        - std::sqrt is too slow and causes branch stalls (~50 clock cycle cost).

          return std::sqrt(lengthSquared());        
    */

    FORCE_INLINE float length() const {
        float lenSq = lengthSquared();
        
        // Prevent division-by-zero or NaN explosions on zero-length vectors
        if (lenSq < 1e-8f) {
            return 0.0f;
        }

        // Calculate length using the fast Reciprocal Square Root identity: sqrt(x) = x * (1.0 / sqrt(x))
        Engine::Math::SIMD::Float4 lenSqReg = Engine::Math::SIMD::Set1(lenSq);
        Engine::Math::SIMD::Float4 invLenReg = Engine::Math::SIMD::ReciprocalSqrt(lenSqReg);
        
        return lenSq * Engine::Math::SIMD::ExtractX(invLenReg);
    }

    FORCE_INLINE void Normalize() {
        // 1. Calculate scalar dot product for the branch check
        float dotScalar = Engine::Math::SIMD::Dot3(reg, reg);
        
        if (dotScalar > 1e-8f) {
            // 2. Broadcast the scalar into SIMD lanes for the hardware reciprocal
            Engine::Math::SIMD::Float4 dotReg = Engine::Math::SIMD::Set1(dotScalar);
            // 3. HARDWARE RECIPROCAL SQRT (1/sqrt)
            Engine::Math::SIMD::Float4 invLenReg = Engine::Math::SIMD::ReciprocalSqrt(dotReg);
            // 4. DIRECT SIMD MULTIPLICATION
            reg = Engine::Math::SIMD::Mul(reg, invLenReg);
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
// LARGE WORLD COORDINATES (LWC)
// ==================================================================================
/*
    - The toFloatVector() is the bridge between Large World Coordinates (64-bit) and Camera/Rendering Matrices (32-bit).
*/
struct Vector3DWorld {
    // 3 (64-bit) dedicated scalar values.
    double x, y, z;

    constexpr Vector3DWorld(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}

    // Standard addition for moving objects in the world.
    FORCE_INLINE constexpr Vector3DWorld operator+(const Vector3DWorld& other) const { return Vector3DWorld(x + other.x, y + other.y, z + other.z); }

    // It returns the difference between two massive world coordinates (subtraction is the most important operator in LWC).
    FORCE_INLINE constexpr Vector3DWorld operator-(const Vector3DWorld& other) const { return Vector3DWorld(x - other.x, y - other.y, z - other.z); }

    // --- THE LWC BRIDGE ---
    // Safely casts 64-bit world differences down to your ultra-fast 32-bit hardware registers
    FORCE_INLINE SIMDVector3D toFloatVector() const {
        // Appending 0.0f to the W lane explicitly marks this difference as a directional vector! By returning a SIMDVector3D (SIMD wrapper), the downstream math instantly utilizes AVX/NEON.
        return SIMDVector3D(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), 0.0f);
    }
};

// ==================================================================================
// EULER ANGLES (GIMBAL LOCK) & QUATERNIONS
// ==================================================================================
/*
    - Euler Angles: Yaw, Pitch, Roll (requires pitch to be clamped to prevent screen flipping).
    - Prevents smooth interpolation and complicates multi-axis rotations (gimbal lock).

      // Constrain pitch to prevent screen-flipping (Gimbal Lock Prevention)
      if (Pitch > 89.0f) Pitch = 89.0f;
      if (Pitch < -89.0f) Pitch = -89.0f;

    - Quaternions: Represents a rotation in 3D space using a 4D complex number (q = w + xi + yi + zi).
    - Eradicates gimbal lock entirely because it represents spherical rotation s directly (no snapping or axis flips). 
    - Allows us to combine rotations using pure SIMD Fused Multiply-Add (FMA) arithmetic.
    - Maps perfectly to a 128-bit register (Hamiltonian Product) using a handful of insruction cycles. 
*/

// ==================================================================================
// SIMD QUATERNION (128-bit) (CROSS-PLATFORM)
// ==================================================================================
struct alignas(16) SIMDQuaternion {
    Engine::Math::SIMD::Float4 reg;

    FORCE_INLINE SIMDQuaternion() : reg(Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f)) {} // SSE: _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f)
    FORCE_INLINE SIMDQuaternion(Engine::Math::SIMD::Float4 m) : reg(m) {}
    FORCE_INLINE SIMDQuaternion(float _x, float _y, float _z, float _w) : reg(Engine::Math::SIMD::Set(_x, _y, _z, _w)) {}

    // --- DOT PRODUCT ---
    FORCE_INLINE float dot(const SIMDQuaternion& other) const { return Engine::Math::SIMD::Dot4(reg, other.reg);}

    // --- HARDWARE SETTERS ---
    // Allows individual lane mutations without spilling the register to the stack (Avoids Load-Hit-Store penalty).
    FORCE_INLINE void setX(float val) { reg = Engine::Math::SIMD::InsertX(reg, val); }
    FORCE_INLINE void setY(float val) { reg = Engine::Math::SIMD::InsertY(reg, val); }
    FORCE_INLINE void setZ(float val) { reg = Engine::Math::SIMD::InsertZ(reg, val); }
    FORCE_INLINE void setW(float val) { reg = Engine::Math::SIMD::InsertW(reg, val); }

    // --- HARDWARE GETTERS ---
    FORCE_INLINE float x() const { return Engine::Math::SIMD::ExtractX(reg); }  // SSE: _mm_cvtss_f32(reg)
    FORCE_INLINE float y() const { return Engine::Math::SIMD::ExtractY(reg); }  // SSE: _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(1, 1, 1, 1)))
    FORCE_INLINE float z() const { return Engine::Math::SIMD::ExtractZ(reg); }  // SSE: _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(2, 2, 2, 2)))
    FORCE_INLINE float w() const { return Engine::Math::SIMD::ExtractW(reg); }  // SSE: _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3)))

    // --- DIRECTIONAL VECTOR ACCESSORS ---
    FORCE_INLINE SIMDVector3D GetForwardVector() const { return RotateVector(SIMDVector3D(0.0f, 0.0f, -1.0f, 0.0f)); } // Returns the normalized forward vector (assuming -Z is forward)
    FORCE_INLINE SIMDVector3D GetRightVector() const { return RotateVector(SIMDVector3D(1.0f, 0.0f, 0.0f, 0.0f)); }    // Returns the normalized right vector (+X)
    FORCE_INLINE SIMDVector3D GetUpVector() const { return RotateVector(SIMDVector3D(0.0f, 1.0f, 0.0f, 0.0f)); }       // Returns the normalized up vector (+Y)

    // --- NORMALIZED LERP (N-Lerp) ---
    /*
        - Draws a straight mathematical line (a chord) through the center of a 4D sphere to get from point A to point B.
        - It then normalizes the result to snap it back to the surface of the sphere.
        - Uses pure SIMD addition and multiplcation (i.e., ultra-fast).
        - Causes the camera's rotational speed to accelerate and decelerate slightly between waypoints (i.e., velocity is not constant since the rotation speeds up slightly in the middle of interpolation and slows down at the ends).
        - e.g., 99% of gameplay (bullet physics, character spines, hitboxes), general gameplay rotation, microscopic adjustments, making an AI smoothly turn to face the player, snapping the camera behind the player's back.
        - e.g. useful as a Slerp fallback when an angle between two quaternions is incredibly small or nearly identical to prevent Slerp formula from dividing by sin(0) which leads to NaN.
    */
    static FORCE_INLINE SIMDQuaternion NLerp(const SIMDQuaternion& q1, const SIMDQuaternion& q2, float t) {
        
        // 1. Shortest Path Enforcement
        // If the dot product is negative, the quaternions point to opposite hemispheres which means the camera is going to take longest path around sphere to reach target.
        // We must negate q2 to ensure the interpolation takes the shortest visual path to reach the target.
        Engine::Math::SIMD::Float4 q2Reg = q2.reg;
        if (q1.dot(q2) < 0.0f) {
            // Enforce shortest path by flipping ALL signs (X, Y, Z, W)
            q2Reg = Engine::Math::SIMD::Negate(q2Reg);
        }

        // 2. Linear Interpolation (res = q1*(1-t) + q2*t)
        Engine::Math::SIMD::Float4 tReg = Engine::Math::SIMD::Set1(t);
        Engine::Math::SIMD::Float4 oneMinusT = Engine::Math::SIMD::Sub(Engine::Math::SIMD::Set1(1.0f), tReg);
        
        Engine::Math::SIMD::Float4 res = Engine::Math::SIMD::Add(
            Engine::Math::SIMD::Mul(q1.reg, oneMinusT), 
            Engine::Math::SIMD::Mul(q2Reg, tReg)
        );
        
        // 3. Normalize to snap it back to the rotation sphere
        SIMDQuaternion result(res);
        result.Normalize();
        return result;
    }

    // --- SPHERICAL LINEAR INTERPOLATION (SLERP) --- 
    /*
        - Traces a path along the surface of a 4D sphere (i.e., interpolate rotations).
        - Constant velocity rotation along the shortest path of the sphere (i.e. generates constant velocity for CinematicTrackController).
        - e.g., rotational speed will be smooth from start to finish for a camera panning between two targets.
        - e.g., used for implementing smooth camera controls, cinematic splines, or AI rotation targeting (making AI entities slowly turn towards the player).
        - e.g., cinematic camera sweeps, interpolating keyframes in skeletal animation like smoothly rotating a joint between two distant animation frames.
        - e.g., slow-motion sniper bullet camera.
    */
   static FORCE_INLINE SIMDQuaternion Slerp(const SIMDQuaternion& q1, const SIMDQuaternion& q2, float t) {
        float cosOmega = q1.dot(q2);
        Engine::Math::SIMD::Float4 q2Reg = q2.reg;

        // 1. SHORTEST PATH ENFORCEMENT: If the dot product is negative, the quaternions point to opposite hemispheres.
        if (cosOmega < 0.0f) {
            // We flip Q2 to force the camera to take the shortest physical rotation path.
            cosOmega = -cosOmega;

            // Enforce shortest path by flipping ALL signs (X, Y, Z, W) to represent the same 3D orientation in space, but sit on opposite poles of the 4D hypersphere's surface.
            q2Reg = Engine::Math::SIMD::Negate(q2Reg);                  // SSE: q2Reg = _mm_xor_ps(q2Reg, _mm_set1_ps(-0.0f));
        }

        // 2. GIMBAL / PRECISION FALLBACK: If the quaternions are nearly identical (angle is basically 0), division by sin(Omega) will cause a NaN explosion. Fallback to N-Lerp.
        if (cosOmega > 0.9999f) {
            // Linear Interpolation (res = q1*(1-t) + q2*t)
            Engine::Math::SIMD::Float4 tReg = Engine::Math::SIMD::Set1(t);
            Engine::Math::SIMD::Float4 oneMinusT = Engine::Math::SIMD::Sub(Engine::Math::SIMD::Set1(1.0f), tReg);
            SIMDQuaternion result(Engine::Math::SIMD::Add(Engine::Math::SIMD::Mul(q1.reg, oneMinusT), Engine::Math::SIMD::Mul(q2Reg, tReg)));
            
            // 3. Normalize to snap it back to the rotation sphere
            result.Normalize();
            return result;
        }

        // --- SPHERICAL MATH (TRANSCENDENTALS: SINE/COS) --- 

        // 3. Extract the angle (Omega) using Fast SIMD Arccosine. We broadcast cosOmega into a register, and evaluate it instantly.
        Engine::Math::SIMD::Float4 cosOmegaReg = Engine::Math::SIMD::Set1(cosOmega);
        Engine::Math::SIMD::Float4 omegaReg = Engine::Math::SIMD::FastACos(cosOmegaReg);

        // 4. BATCHED SIMD SINE EVALUATION (Zero Load-Hit-Store)

        // To build this without LHS (Load-Hit-Store) packing, we use the Set command ONCE with scalars, then multiply the entire register by omegaReg.
        Engine::Math::SIMD::Float4 multiplierReg = Engine::Math::SIMD::Set(1.0f, (1.0f - t), t, 0.0f);  // We need a register formatted as: [ omega, (1-t), t, 0.0 ]
        Engine::Math::SIMD::Float4 angles = Engine::Math::SIMD::Mul(omegaReg, multiplierReg); // angles = [ omega*1, omega*(1-t), omega*t, 0.0 ]

        // Evaluate all 3 sines simultaneously in ~20 clock cycles
        Engine::Math::SIMD::Float4 sines = Engine::Math::SIMD::FastSin(angles);


        // 5. FULLY SIMD WEIGHT CALCULATION (Zero Extraction, Zero Scalar Division)

        // Broadcast sin(omega) from the X lane to all 4 lanes: [sinOmega, sinOmega, sinOmega, sinOmega]
        Engine::Math::SIMD::Float4 sinOmegaReg = Engine::Math::SIMD::BroadcastX(sines);

        // Fast hardware reciprocal: 1.0f / sin(omega) (~4 cycles instead of ~15)
        Engine::Math::SIMD::Float4 invSinOmegaReg = Engine::Math::SIMD::Reciprocal(sinOmegaReg);

        // Calculate the transcendental weights, multiply sines by the reciprocal to get the final weights!
        // weights = [ 1.0f, weight0, weight1, garbage ]
        Engine::Math::SIMD::Float4 weights = Engine::Math::SIMD::Mul(sines, invSinOmegaReg);

        // 6. SIMD RE-ASSEMBLY [res = (q1 * w0) + (q2 * w1)]
        Engine::Math::SIMD::Float4 res = Engine::Math::SIMD::Add(
            // Broadcast weight0 (Y lane) and weight1 (Z lane) directly from the register into the quaternion multipliers
            Engine::Math::SIMD::Mul(q1.reg, Engine::Math::SIMD::BroadcastY(weights)), 
            Engine::Math::SIMD::Mul(q2Reg, Engine::Math::SIMD::BroadcastZ(weights))
        );

        return SIMDQuaternion(res);
    }

    // --- ANGLE AXIS CONVERSION ---
    // Converts a normalized 3D axis and an angle into a Quaternion. Used when converting mouse/keyboard input to a rotation.
    static FORCE_INLINE SIMDQuaternion AngleAxis(float angleDegrees, const SIMDVector3D& axis) {
        using namespace Engine::Math::SIMD;

        float halfAngleRad = angleDegrees * Engine::Math::Constants::DEG_TO_RAD * 0.5f;
        
        // Replaces std::sin and std::cos (~100 cycles) with a single batched polynomial approximation (~15 cycles).
        auto [sin_d, cos_d] = Engine::Math::Functions::FastSinCos(halfAngleRad);

        // 2. Multiply the normalized axis by sin(half_angle)
        Float4 sinVec = Set1(sin_d);
        Float4 axisScaled = Mul(axis.reg, sinVec);

        // 3. Insert the Cosine value directly into the W lane! Skips the Blend instruction.
        Float4 result = InsertW(axisScaled, cos_d);
        
        return SIMDQuaternion(result);
    }

    // --- THE HAMILTON PRODUCT (SIMD QUATERNION MULTIPLICATION) ---
    // Combines two rotations into one.
    FORCE_INLINE SIMDQuaternion operator*(const SIMDQuaternion& rhs) const {
        return SIMDQuaternion(Engine::Math::SIMD::QuaternionMul(reg, rhs.reg));
    }

    // --- HARDWARE NORMALIZATION ---
    FORCE_INLINE void Normalize() {
        float dot = Engine::Math::SIMD::Dot4(reg, reg);
        if (dot > 1e-8f) {
            Engine::Math::SIMD::Float4 dotReg = Engine::Math::SIMD::Set1(dot);
            Engine::Math::SIMD::Float4 invLenReg = Engine::Math::SIMD::ReciprocalSqrt(dotReg);
            reg = Engine::Math::SIMD::Mul(reg, invLenReg); // Multiply directly, no extraction needed!
        }
    }

    // --- CONJUGATE (INVERSE ROTATION) ---
    FORCE_INLINE SIMDQuaternion Conjugate() const {
        // Negates X, Y, and Z. Required to generate View Matrices!
        return SIMDQuaternion(Engine::Math::SIMD::FlipSignXYZ(reg));
    }
    
    // --- PURE SIMD ROTATE VECTOR ---
    // Rotates a 3D vector by this quaternion: V' = Q * V * Q^-1
    // Drastically faster than extracting x, y, and z to memory! Fast path for rotating a vector by a quaternion.
    FORCE_INLINE SIMDVector3D RotateVector(const SIMDVector3D& v) const {
        // 1. Mask out W (Force it to 0.0) to get purely the imaginary (x,y,z) axis
        SIMDVector3D qVec(Engine::Math::SIMD::BlendMaskW(reg, Engine::Math::SIMD::Zero()));

        // 2. Broadcast the Real (w) component natively within the registers!
        Engine::Math::SIMD::Float4 wReg = Engine::Math::SIMD::BroadcastW(reg);
        
        // 3. V' = V + 2w(Q_xyz x V) + 2(Q_xyz x (Q_xyz x V))
        SIMDVector3D t = qVec.cross(v) * 2.0f;

        // Multiply t by w directly in the registers, avoiding scalar extraction
        SIMDVector3D tw(Engine::Math::SIMD::Mul(t.reg, wReg)); 
        return v + tw + qVec.cross(t);
    }

    // --- DIRECTION TO QUATERNION ---
    // Converts a normalized forward vector into a rotation without using Trigonometry.
    static FORCE_INLINE SIMDQuaternion FromDirection(const SIMDVector3D& dir) {
        SIMDVector3D baseForward(0.0f, 0.0f, -1.0f, 0.0f); 
        float dot = baseForward.dot(dir);
        
        // Edge Case: The entity (or camera) needs to perfectly turn around 180 degrees
        if (dot < -0.9999f) {
            return SIMDQuaternion(0.0f, 1.0f, 0.0f, 0.0f); // 180-degree Yaw
        }
        
        // Build the Quaternion using the cross product axis and the half-way dot product
        SIMDVector3D axis = baseForward.cross(dir);
        SIMDQuaternion q(axis.x(), axis.y(), axis.z(), 1.0f + dot);
        q.Normalize();
        return q;
    }

    // --- EULER TO QUATERNION (SIMD) ---
    // Converts human-readable Euler Angles [Pitch (X), Yaw (Y), Roll (Z) in degrees] into quaternions.
    static FORCE_INLINE SIMDQuaternion FromEuler(float pitch, float yaw, float roll) {
        using namespace Engine::Math::SIMD;

        float p = pitch * Engine::Math::Constants::DEG_TO_RAD * 0.5f;
        float y = yaw   * Engine::Math::Constants::DEG_TO_RAD * 0.5f;
        float r = roll  * Engine::Math::Constants::DEG_TO_RAD * 0.5f;

        // 1. BATCHED SIMD TRANSCENDENTAL EVALUATION
        // Pack all three scalar angles into a single 128-bit register. (X=p, Y=y, Z=r, W=0)
        Float4 angles = Set(p, y, r, 0.0f);

        // This replaces 6 scalar (std::sin, std::cos) standard library calls (~300 cycles) with 2 SIMD polynomial expansions (~40 cycles total).
        // Float4 sines   = FastSin(angles);
        // Float4 cosines = FastCos(angles);

        // 2. Evaluate all 3 Sines and 3 Cosines simultaneously. Shares the hardware squaring (x^2) and range reduction across all axes (<40 cycles total)!
        auto [sines, cosines] = FastSinCos(angles);

        // 3. Extract the evaluated results
        float sp = ExtractX(sines);
        float sy = ExtractY(sines);
        float sr = ExtractZ(sines);

        float cp = ExtractX(cosines);
        float cy = ExtractY(cosines);
        float cr = ExtractZ(cosines);

        // 4. Scalar Assembly
        // The compiler will pipeline these standard FPU multiplications perfectly.
        return SIMDQuaternion(
            sr * cp * cy - cr * sp * sy, // X
            cr * sp * cy + sr * cp * sy, // Y
            cr * cp * sy - sr * sp * cy, // Z
            cr * cp * cy + sr * sp * sy  // W
        );
    }

    // --- EULER TO QUATERNION (OPTIMIZED SCALAR) ---
    // Converts human-readable Euler Angles [Pitch (X), Yaw (Y), Roll (Z) in degrees] into quaternions.
    static FORCE_INLINE SIMDQuaternion FromEulerScalar(float pitch, float yaw, float roll) {
        // 1. Convert to Half-Radians
        float pitch_rad = pitch * Engine::Math::Constants::DEG_TO_RAD * 0.5f;
        float yaw_rad = yaw   * Engine::Math::Constants::DEG_TO_RAD * 0.5f;
        float roll_rad = roll  * Engine::Math::Constants::DEG_TO_RAD * 0.5f;

        // 2. Custom Scalar FastSinCos (3 calls * (~12 clock cycles)) = ~36 clock cycles total!
        auto [sine_pitch, cos_pitch] = Engine::Math::Functions::FastSinCos(pitch_rad);
        auto [sine_yaw, cos_yaw] = Engine::Math::Functions::FastSinCos(yaw_rad);
        auto [sine_roll, cos_roll] = Engine::Math::Functions::FastSinCos(roll_rad);

        // 2. std::sin, std::cos (6 calls * (~50 clock cycles)) = ~300 clock cycles total
        // float sin_pitch = std::sin(pitch_rad); float cos_pitch = std::cos(pitch_rad);
        // float sin_yaw = std::sin(yaw_rad); float cos_yaw = std::cos(yaw_rad);
        // float sin_roll = std::sin(roll_rad); float cos_roll = std::cos(roll_rad);

        // 3. Final Assembly
        return SIMDQuaternion(
            sine_roll * cos_pitch  * cos_yaw  - cos_roll  * sine_pitch * sine_yaw, // X
            cos_roll  * sine_pitch * cos_yaw  + sine_roll * cos_pitch  * sine_yaw, // Y
            cos_roll  * cos_pitch  * sine_yaw - sine_roll * sine_pitch * cos_yaw,  // Z
            cos_roll  * cos_pitch  * cos_yaw  + sine_roll * sine_pitch * sine_yaw  // W
        );
    }
};

// ==================================================================================
// LOAD-HIT-STORE PENALTY (SIMD -> SCALAR -> SIMD)  
// ==================================================================================
/*
    - Modern CPUs have distinct execution units and register files.

        1. Floating Point Unit (FPU) is used for scalar math that handles one floating-point operation at a time (e.g., A = B + C) with its own set of scalar registers.
        2. Vector Units (VU) are used for SIMD math that is designed to handle multiple data points simultaneously (e.g., 4, 8, or 16 floats) with its own separate set of wider registers (e.g., __m128: XMM, __m256: YMM).
        
    - When dealing with a single entity (or struct), packing scalar data (e.g., "float data[4]") into 128 bit SSE registers actually hurts performance (i.e., Load-Hit-Store / Shuffle Penalty).
    - CPU wastes clock cycles shuffling the data from the FPU, packing it into the Vector Units (or SSE registers) for the cross product and then unpacking it again for the entity (e.g., Vector3DStack).

        1. Loads 4 separate floats (x, y, z, w) from memory (or scalar registers).
        2. Pack them into into 128-bit SIMD registers (i.e., slow).
        3. Perform the SIMD math.
        4. Unpack (or extract) the resultant floats back out of the SIMD registers and store it into scalar variables (i.e., slow).

    - Takes more clock cycles to perform the packing and unpacking than just doing three multiplications and subtractions on the standard FPU.
    - Never mix pure scalars (e.g., float) with SIMD registers (e.g., __m128).

      // Rips the data out of the fast SIMD lanes and places it into the slow lanes and places it back into the fast lane (SIMD -> Scalar -> SIMD).
      static FORCE_INLINE Matrix4x4_SIMD TRS(const SIMDQuaternion& rotation) {
            // Extracts the 4 (32-bit) floats out of the 128-bit SIMDQuaternion register and puts it into 4 scalar registers (SIMD -> Scalar, slow).
            float x2 = rotation.x();
            float xx = rotation.x();
            float yy = rotation.y();
            float wx = rotation.w();

            // Extracts the 3 (32-bit) floats out of the 128-bit SIMDQuaternion register and puts it into 3 scalar registers (SIMD -> Scalar, slow).
            float sx = scale.x();
            float sy = scale.y();
            float sz = scale.z();

            // 3. The Re-packing Bottleneck (Scalar -> SIMD, slow)
            mat.col[0] = Engine::Math::SIMD::Set((1.0f - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0.0f);
      }

    - In modern CPU architectures, the processor is heavily pipelined (i.e., means it has different dedicated lanes or pathways for different types of data).
    - SIMD pipeline is designed to handle 128-bit registers (e.g., SIMDQuaternion).
    - Scalar pipeline is a narrow lane designed to handle single, 32-bit floats.
    - Memory controller is a pathway that moves data to and from L1/L2 caches and RAM.
    - Load-Hit-Store penalties occur when code forces the CPU to move data between the SIMD pipeline and Scalar pipeline (SIMD -> Scalar -> SIMD).

      class Vector3DStack {
        public:
            // Storing data in an array of floats inside a struct that is meant to be used for SIMD is an anti-pattern (i.e., Load-Hit-Store penalty).
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

            // In-place Scalar Multiplication: A *= scalar
            FORCE_INLINE void operator*=(float scalar) {
                __m128 v1 = _mm_load_ps(this->data);
                __m128 s = _mm_set1_ps(scalar);
                _mm_store_ps(this->data, _mm_mul_ps(v1, s)); // Store directly back into itself
            }

            // Cross Product: v = <(y1 * z2) - (z1 * y2), (z1 * x2) - (x1 * z2), (x1 * y2) - (y1 * x2)> (Load-Hit-Store Penalty)
            FORCE_INLINE Vector3DStack cross(const Vector3DStack& other) const {
                Vector3DStack result;
                
                // 1. Memory Load: Reads 16-bytes (4 floats) from stack memory into a 128-bit SIMD register (i.e., pack them into into 4 (128-bit) SIMD registers (Slow, ~4-5 cycles)).
                __m128 a = _mm_load_ps(this->data); // 4 floats (128-bit SIMD register), memory read from L1 cache directly into the XMM vector registers.
                __m128 b = _mm_load_ps(other.data); // 4 floats (128-bit SIMD register), memory read from L1 cache directly into the XMM vector registers.

                // 2. Rearrange: Swizzles the lanes (x, y, z, w) inside the registers to prepare for the cross product (i.e., register-to-register hardware manipulation (Very fast, ~1 cycle)).
                __m128 tmp0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
                __m128 tmp1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
                __m128 tmp2 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
                __m128 tmp3 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));

                // 3. Perform the SIMD math (i.e., parallel multiplication (~3-4 cycles), parallel subtraction).
                __m128 res = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));

                // 4. Memory Store: Dumps the entire 128-bit register back to stack memory (i.e., triggers load-hit-store stall because caller of this function will try to read this memory as a scalar).
                _mm_store_ps(result.data, res);
                
                // 5. When caller tries to access result.data[0] it causes a CPU pipeline stall (Very slow, ~10-12 cycles per infraction) b/c we have to wait for that 16-byte block to finish writing to memory before we can read the scalar float back out forcing it to flush the pipeline.
                return result;
            }

            // Utility to easily print the vector
            void print() const {
                std::println("[{}, {}, {}, {}]", data[0], data[1], data[2], data[3]);
            }
    };
*/

// ==================================================================================
// SIMD 4x4 MATRIX & INTERPOLATION  (CROSS-PLATFORM) 
// ==================================================================================
/* 
    - A SIMD 4x4 matrix struct should not contain a raw array of 16 floats because every operation would require _mm_load_ps to fetch from memory and _mm_store_ps to save the result (i.e., Load-Hit-Store penalty).
    - To solve this, never leave SIMD registers (Matrix4x4_SIMD and SIMDVector3D are always 128-bit SIMD registers). 
    - Instead it should be an array of four 128-bit SIMD registers to keep our camera math on the silicon.

      1. Load the camera data into an SSE register.
      2. Once the camera's position is loaded (or enters) a SIMD register it should stay there for its entire mathematical lifecycle.
      3. Perform all LookAt, Projection and View Matrix math inside SIMD (SSE).
      4. Only extract the data when pushing it to the GPU via uniform buffers.

    - Matrix4x4_SIMD uses full 128-bit width of the SSE registers.
    - Allocates 4 (128-bit) hardware execution registers inside the CPU.
    - 4 x 128 = 512-bits = 64-bytes (perfectly fits in cache).
*/
struct alignas(64) Matrix4x4_SIMD {
    // 4 columns, each taking up exactly one 128-bit register, guarantees that once data enters the Vector Units, it stays in it.
    Engine::Math::SIMD::Float4 col[4];

    // Creates a blank slate matrix (No rotation, no scale, at origin 0,0,0).
    static FORCE_INLINE Matrix4x4_SIMD Identity() {
        Matrix4x4_SIMD mat;

        // Creates an Identity Matrix entirely inside the registers.
        mat.col[0] = Engine::Math::SIMD::Set(1.0f, 0.0f, 0.0f, 0.0f);  // { 1, 0, 0, 0 }  SSE: mat.col[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f); 
        mat.col[1] = Engine::Math::SIMD::Set(0.0f, 1.0f, 0.0f, 0.0f);  // { 0, 1, 0, 0 }  SSE: mat.col[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f); 
        mat.col[2] = Engine::Math::SIMD::Set(0.0f, 0.0f, 1.0f, 0.0f);  // { 0, 0, 1, 0 }  SSE: mat.col[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f); 
        mat.col[3] = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f);  // { 0, 0, 0, 1 }  SSE: mat.col[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); 
        return mat;
    }

    // ============================================================
    // --- HIGH-PERFORMANCE SIMD MATRIX INVERSE (Cramer's Rule) ---
    // ============================================================
    /*
        - Guassian Elimination requires lots of branching (if-statements) to handle pivoting which breaks SIMD execution.
        - Cramer's Rule is the most mathematically optimal way to invert 4x4 matrices (i.e., no branching).
        - Can calculate the determinants of multiple sub-matrices in parallel across 4 lanes of a register.
        - Usually pulled directly from the DirectXMath or GLM SIMD headers to guarantee floating-point stability), but this is a third-party dependency that we don't want. 
        - If a matrix moves an object 5 units down, its inverse moves it 5 units up (i.e., mathematical opposite of a transformation).
        - e.g., required for mouse picking, screen-to-world raycasting, and normal matrix generation.
        - e.g., translates (or converts) 2D mouse clicks [coordinate (x, y)] into 3D world rays to let us know what the user clicked in the world.
        - e.g., converts a 2D screen click back into a 3D world direction (Screen Space -> World space).
    */
   FORCE_INLINE Matrix4x4_SIMD Inverse() const {
        Matrix4x4_SIMD inv;
        
        // 1. Extract columns
        inv.col[0] = col[0];
        inv.col[1] = col[1];
        inv.col[2] = col[2];
        inv.col[3] = col[3];

        // 2x2 Sub-determinants (ARM/Intel's Cramer's Rule implementation), let the Hardware Abstraction Layer route to the fastest microcode.
        Engine::Math::SIMD::Inverse4x4(inv.col[0], inv.col[1], inv.col[2], inv.col[3]);
        
        // Returns the fully inverted matrix
        return inv;
    }

    // =====================================================
    // --- 2D ORTHOGRAPHIC RENDERING (PROJECTION MATRIX) ---
    // =====================================================
    /*
        - Objects remain the exact same size regardless of how far they are from the camera. 
        - Parallel lines remain perfectly parallel forever; they never converge (i.e., no concept of vanishing points).
        - The Z coordinate is used strictly to determine rendering order (which object is drawn on top of the other), but is never used to scale or warp the X and Y coordinates.
        - The viewable area is shaped like a perfect rectangular box (i.e., a rectangular prism).
        - Critical for rendering 2D UI (e.g., text, minimaps, crosshairs, etc..)
        - Used for rendering the scene from the sun's perspective when calculating Shadow Maps (shadows casted by directional light).
    */

    // Orthographic Projection Matrix
    static FORCE_INLINE Matrix4x4_SIMD Orthographic_SIMD(float left, float right, float bottom, float top, float nearZ, float farZ) {
        Matrix4x4_SIMD mat;
        
        float invRL = 1.0f / (right - left);
        float invTB = 1.0f / (top - bottom);
        float invFN = 1.0f / (farZ - nearZ);

        using namespace Engine::Math::SIMD;
        
        mat.col[0] = Set(2.0f * invRL, 0.0f, 0.0f, 0.0f);    // Col 0: Right (X scale)
        mat.col[1] = Set(0.0f, 2.0f * invTB, 0.0f, 0.0f);    // Col 1: Up (Y scale)
        mat.col[2] = Set(0.0f, 0.0f, -2.0f * invFN, 0.0f);   // Col 2: Forward (Z scale)
        mat.col[3] = Set(-(right + left) * invRL, -(top + bottom) * invTB, -(farZ + nearZ) * invFN, 1.0f);  // Col 3: Translation offsets

        return mat;
    }

    // ====================================================
    // --- 3D PERSPECTIVE RENDERING (PROJECTION MATRIX) ---
    // ====================================================
    /*
        - Mimics how the way human eyes and camera lenses perceive the real world (i.e., objects appear smaller the further they are away from the camera).
        - Parallel lines appear to converge at a vanishing point on the horizon.
        - X and Y coordinates of a vertex are divided by its Z coordinate (depth) to shrink the object (i.e., perspective divide).
        - A frustum is the viewable area that is shaped like a truncated pyramid, the near plane is small (right in front of camera), the far plane is massive (stretches out to the horizon).
        - Used for 99% of games (first-person shooters, third-person action games, racing games, VR, etc..) to create a realistic and immersive 3D world feel for the player. 
    */

    // Perspective Projection Matrix
    static FORCE_INLINE Matrix4x4_SIMD Perspective_SIMD(float fovY_degrees, float aspect, float nearZ, float farZ) {
        Matrix4x4_SIMD mat;
        float fovY_rad = fovY_degrees * Engine::Math::Constants::DEG_TO_RAD;
        
        // 1. Pack the scalar angle into a 128-bit register
        Engine::Math::SIMD::Float4 halfFovReg = Engine::Math::SIMD::Set1(fovY_rad * 0.5f);
        
        // 2. Evaluate using the SIMD Fast Tangent
        Engine::Math::SIMD::Float4 tanReg = Engine::Math::SIMD::FastTan(halfFovReg);
        
        // 3. Extract the scalar result back out
        float tanHalfFovY = Engine::Math::SIMD::ExtractX(tanReg);

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

    // ============================================
    // --- STANDARD VIEW MATRIX (32-BIT FLOATS) ---
    // ============================================
    /*
        - Rotation + Translation
        - Builds a 4x4 matrix containing both the rotation and translation.
        - Used for 95% of standard games where the world is reasonably sized (under a few kilometers across) and constraied to 32-bit floats.
        - e.g., arena shooters, platformers, fighting games.
    */

    static FORCE_INLINE Matrix4x4_SIMD LookAt_SIMD(const SIMDVector3D& eye, const SIMDVector3D& target, const SIMDVector3D& upVec) {
        // 1. Forward Vector (Z)
        SIMDVector3D f = (target - eye).asDirection(); // Force W=0.0f
        f.Normalize();

        // 2. Right Vector (X)
        SIMDVector3D r = f.cross(upVec).asDirection();
        r.Normalize();

        // 3. Up Vector (Y)
        SIMDVector3D u = r.cross(f).asDirection();

        // 4. Negate Forward vector for Right-Handed coordinates (i.e., required for Right-Handed Coordinate Systems like OpenGL).
        Engine::Math::SIMD::Float4 negF = Engine::Math::SIMD::Mul(f.reg, Engine::Math::SIMD::Set1(-1.0f));

        // 5. Calculate Translation Vector
        // Standard View Matrix translation is: [-dot(R, eye), -dot(U, eye), dot(F, eye)]
        float tx = -r.dot(eye);
        float ty = -u.dot(eye);
        float tz = f.dot(eye); 

        // 6. Pack into a single register for the 4th column.
        Engine::Math::SIMD::Float4 translation = Engine::Math::SIMD::Set(tx, ty, tz, 1.0f);

        // 7. Setup rows for Hardware Transposition
        Engine::Math::SIMD::Float4 row0 = r.reg;   // { Rx, Ry, Rz, 0 }
        Engine::Math::SIMD::Float4 row1 = u.reg;   // { Ux, Uy, Uz, 0 }
        Engine::Math::SIMD::Float4 row2 = negF;    // {-Fx,-Fy,-Fz, 0 }
        Engine::Math::SIMD::Float4 row3 = Engine::Math::Constants::SIMD_MASK_W_ONE; // { 0,  0,  0,  1 } Instant Load

        // Hardware Transpose flips the 3x3 rotation axes into Column-Major format instantly.
        Engine::Math::SIMD::Transpose4(row0, row1, row2, row3);

        // 8. Store the columns, overwriting the transposed 4th column with our calculated translation.
        Matrix4x4_SIMD mat;
        mat.col[0] = row0;
        mat.col[1] = row1;
        mat.col[2] = row2;
        mat.col[3] = translation; 
        return mat;
    }

    // ============================================================
    // --- LARGE WORLD COORDINATES VIEW MATRIX (64-BIT DOUBLES) ---
    // ============================================================
    /*
        - Rotation
        - View matrix longer needs translation with camera-relative rendering, only handles rotation.
        - No translation b/c the camera is never moved and stays bolted to the center of the universe at (0.0, 0.0, 0.0) making it camera-relative.

          1. Before rendering, the engine takes every object in the game.
          2. Subtracts the camera's 64-bit world position from the object's 64-bit world position.
          3. Builds the object's ModelMatrix using that new relative position (i.e., camera-relative position). 
          4. Because every object is dynamically shifted to be be drawn relative to the camera, the camera's ViewMatrix does not need to contain any translation data. 

        - It only needs to handle rotations (i.e., where the player is looking).
        - Inputs are Vector3DWorld (double), but the matrix is float.
        - Is the Approaching Zero Overhead (AZDO) / Open-World camera view matrix.
        - Used for massive open-world games that span 50,000 kilometers! Standard 32-bit floats will fail and cause models to warp and jitter.
    */

    // --- LWC CAMERA-RELATIVE RENDERING LOOK-AT ---
    static FORCE_INLINE Matrix4x4_SIMD LookAtLWC_SIMD(const Vector3DWorld& eye, const Vector3DWorld& target, const SIMDVector3D& upVec) {
        
        // 1. Calculate the forward vector in 64-bit space to prevent floating-point jitter at massive distances
        Vector3DWorld worldForward = target - eye;
        
        // 2. Cast down to 32-bit float for the math. 
        // Because it's a directional vector (difference), the cast is perfectly safe!
        SIMDVector3D f = worldForward.toFloatVector();
        
        // Use your fast SIMD dot product to normalize forward vector
        f.Normalize();

        // 3. Right Vector (X) - SIMD Cross Product
        SIMDVector3D r = f.cross(upVec).asDirection();
        r.Normalize();

        // 4. Up Vector (Y) - SIMD Cross Product
        SIMDVector3D u = r.cross(f).asDirection();

        // 5. Negate Forward vector for Right-Handed coordinates systems
        Engine::Math::SIMD::Float4 negF = Engine::Math::SIMD::Mul(f.reg, Engine::Math::SIMD::Set1(-1.0f));

        // 6. Build Row vectors (Force W=0 for X,Y,Z rows, W=1 for Translation row)
        Engine::Math::SIMD::Float4 row0 = r.reg;        // { Rx, Ry, Rz, 0 }
        Engine::Math::SIMD::Float4 row1 = u.reg;        // { Ux, Uy, Uz, 0 }
        Engine::Math::SIMD::Float4 row2 = negF;         // {-Fx,-Fy,-Fz, 0 }
        Engine::Math::SIMD::Float4 row3 = Engine::Math::Constants::SIMD_MASK_W_ONE; // { 0,  0,  0,  1 } Instant Load

        // 7. Hardware Transpose flips our rows into Column-Major format in-place!
        Engine::Math::SIMD::Transpose4(row0, row1, row2, row3);

        // 8. Store the transposed registers directly into the matrix columns.
        Matrix4x4_SIMD mat;
        mat.col[0] = row0;
        mat.col[1] = row1;
        mat.col[2] = row2;
        
        // 9. ZERO TRANSLATION!
        // Because every object will be rendered relative to the camera, the camera is always at (0,0,0).
        // Row 3 after transpose contains the translation data, we overwrite it with [0,0,0,1].
        mat.col[3] = Engine::Math::SIMD::Set(0.0f, 0.0f, 0.0f, 1.0f);

        return mat;
    }

    // ==================================
    // --- MODEL MATRIX BUILDER (TRS) ---
    // ==================================
    /*
        - M = Translation * Rotation * Scale 

          1. Scale the object first (so you don't accidentally scale the translation distance).
          2. Rotate the object second (so it rotates around its own local center).
          3. Translate (move) the object last, putting in its final position in the world.

        - Formula for placing an object in a 3D world.
        - Converts ECS TransformComponent (SIMDVector3D and SIMDQuaternion) into the Matrix4x4_SIMD format that the GPU demands (i.e., Vulkan, OpenGL, DirectX 12).
        - When a 3D artist models a character or a tree in Blender or Maya, they usually model it centered eactly at the grid origin (0, 0, 0).
        - The raw vertex inside the (.obj) and (.gltf) file reflects those exact coordinates.
        - Without this function, every 3D mesh you load will spawn directly at the origin (0, 0, 0) at a default scale of 1.0.
        - If you send those raw vertices directly to the GPU without multiplying them by a ModelMatrix (TRS), the GPU will draw the mesh exactly where it was modeled (0, 0, 0) in the game world, facing its default direction, at exactly 1.0 scale.
    */

    // TRS Matrix physically moves the mesh from its local space into the vast game world.
    static FORCE_INLINE Matrix4x4_SIMD TRS(const SIMDVector3D& translation, const SIMDQuaternion& rotation, const SIMDVector3D& scale) {
        using namespace Engine::Math::SIMD;
        Matrix4x4_SIMD mat;

        Float4 q = rotation.reg;

        // 1. Broadcast the quaternion components: [X,X,X,X], [Y,Y,Y,Y], [Z,Z,Z,Z], [W,W,W,W]
        Float4 qx = BroadcastX(q);
        Float4 qy = BroadcastY(q);
        Float4 qz = BroadcastZ(q);
        Float4 qw = BroadcastW(q);

        // 2. Pre-calculate the doubled components: 2x, 2y, 2z
        Float4 two = Set1(2.0f);
        Float4 qx2 = Mul(qx, Engine::Math::Constants::SIMD_TWO);
        Float4 qy2 = Mul(qy, Engine::Math::Constants::SIMD_TWO);
        Float4 qz2 = Mul(qz, Engine::Math::Constants::SIMD_TWO);

        // 3. Calculate squared terms: 2x^2, 2y^2, 2z^2
        Float4 xx2 = Mul(qx, qx2);
        Float4 yy2 = Mul(qy, qy2);
        Float4 zz2 = Mul(qz, qz2);

        // 4. Calculate mixed terms
        Float4 xy2 = Mul(qx, qy2);
        Float4 xz2 = Mul(qx, qz2);
        Float4 yz2 = Mul(qy, qz2);
        Float4 wx2 = Mul(qw, qx2);
        Float4 wy2 = Mul(qw, qy2);
        Float4 wz2 = Mul(qw, qz2);

        // 5. Build the rotation columns (Notice the specific mapping for OpenGL column-major)
        // Col 0: [1 - 2y^2 - 2z^2, 2xy + 2wz, 2xz - 2wy, 0.0]
        Float4 col0 = Engine::Math::Constants::SIMD_ONE; // Instant Load
        col0 = Sub(col0, yy2);
        col0 = Sub(col0, zz2);
        col0 = InsertY(col0, ExtractX(Add(xy2, wz2))); // We use Insert to build the column
        col0 = InsertZ(col0, ExtractX(Sub(xz2, wy2)));
        col0 = InsertW(col0, 0.0f);

        // Col 1: [2xy - 2wz, 1 - 2x^2 - 2z^2, 2yz + 2wx, 0.0]
        Float4 col1 = Sub(xy2, wz2);                   // X lane holds the target value
        col1 = InsertY(col1, ExtractX(Sub(Sub(Set1(1.0f), xx2), zz2))); 
        col1 = InsertZ(col1, ExtractX(Add(yz2, wx2)));
        col1 = InsertW(col1, 0.0f);

        // Col 2: [2xz + 2wy, 2yz - 2wx, 1 - 2x^2 - 2y^2, 0.0]
        Float4 col2 = Add(xz2, wy2);
        col2 = InsertY(col2, ExtractX(Sub(yz2, wx2)));
        col2 = InsertZ(col2, ExtractX(Sub(Sub(Set1(1.0f), xx2), yy2)));
        col2 = InsertW(col2, 0.0f);

        // 6. Apply Scale directly to the rotation columns
        Float4 scaleReg = scale.reg;
        mat.col[0] = Mul(col0, BroadcastX(scaleReg));
        mat.col[1] = Mul(col1, BroadcastY(scaleReg));
        mat.col[2] = Mul(col2, BroadcastZ(scaleReg));

        // 7. Inject Translation into the 4th Column (Ensure W = 1.0f)
        mat.col[3] = BlendMaskW(translation.reg, Engine::Math::Constants::SIMD_MASK_W_ONE);

        return mat;
    }
};

// --- SIMD MATRIX VECTOR MULTIPLICATION  (Vector = Matrix1 * Vector1) --- Multiplies a Vector against a Matrix
FORCE_INLINE SIMDVector3D operator*(const Matrix4x4_SIMD& mat, const SIMDVector3D& v) {
    // 1. Broadcast components (X, Y, Z, W) directly from register to register.
    Engine::Math::SIMD::Float4 vx = Engine::Math::SIMD::BroadcastX(v.reg);   // {x, x, x, x}  SSE: __m128 vx = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(0, 0, 0, 0)); 
    Engine::Math::SIMD::Float4 vy = Engine::Math::SIMD::BroadcastY(v.reg);   // {y, y, y, y}  SSE: __m128 vy = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(1, 1, 1, 1)); 
    Engine::Math::SIMD::Float4 vz = Engine::Math::SIMD::BroadcastZ(v.reg);   // {z, z, z, z}  SSE: __m128 vz = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(2, 2, 2, 2)); 
    Engine::Math::SIMD::Float4 vw = Engine::Math::SIMD::BroadcastW(v.reg);   // {w, w, w, w}  SSE: __m128 vw = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(3, 3, 3, 3)); 

    // 2. Multiply each broadcasted component by its corresponding matrix column
    Engine::Math::SIMD::Float4 res = Engine::Math::SIMD::Mul(vx, mat.col[0]);              // SSE: __m128 res = _mm_mul_ps(vx, mat.col[0]);

    // 3. Fused Multiply-Add the rest of the columns [res = (vy * col1) + res]
    res = Engine::Math::SIMD::FMAdd(vy, mat.col[1], res);  // SSE: res = _mm_fmadd_ps(vy, mat.col[1], res);  
    res = Engine::Math::SIMD::FMAdd(vz, mat.col[2], res);  // SSE: res = _mm_fmadd_ps(vz, mat.col[2], res);
    res = Engine::Math::SIMD::FMAdd(vw, mat.col[3], res);  // SSE: res = _mm_fmadd_ps(vw, mat.col[3], res);

    return SIMDVector3D(res);
}

// --- SIMD MATRIX MULTIPLICATION (Matrix = Matrix1 * Matrix2) --- Multiplies a Matrix against a Matrix
FORCE_INLINE Matrix4x4_SIMD operator*(const Matrix4x4_SIMD& a, const Matrix4x4_SIMD& b) {
    Matrix4x4_SIMD res;

    // For each column in "b", broadcast its X, Y, Z, W components and multiply them against the corresponding columns of "a".
    for (int i = 0; i < 4; ++i) {
        // Data never leaves the SIMD registers
        Engine::Math::SIMD::Float4 vx = Engine::Math::SIMD::BroadcastX(b.col[i]);   // SSE: __m128 vx = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(0, 0, 0, 0));
        Engine::Math::SIMD::Float4 vy = Engine::Math::SIMD::BroadcastY(b.col[i]);   // SSE: __m128 vy = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(1, 1, 1, 1));
        Engine::Math::SIMD::Float4 vz = Engine::Math::SIMD::BroadcastZ(b.col[i]);   // SSE: __m128 vz = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(2, 2, 2, 2));
        Engine::Math::SIMD::Float4 vw = Engine::Math::SIMD::BroadcastW(b.col[i]);   // SSE: __m128 vw = _mm_shuffle_ps(b.col[i], b.col[i], _MM_SHUFFLE(3, 3, 3, 3));

        Engine::Math::SIMD::Float4 col = Engine::Math::SIMD::Mul(vx, a.col[0]);     // SSE: __m128 col = _mm_mul_ps(vx, a.col[0]);
        col = Engine::Math::SIMD::FMAdd(vy, a.col[1], col);   // SSE: col = _mm_fmadd_ps(vy, a.col[1], col);
        col = Engine::Math::SIMD::FMAdd(vz, a.col[2], col);   // SSE: col = _mm_fmadd_ps(vz, a.col[2], col);
        col = Engine::Math::SIMD::FMAdd(vw, a.col[3], col);   // SSE: col = _mm_fmadd_ps(vw, a.col[3], col);
        
        res.col[i] = col;
    }
    return res;
}

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
        using namespace Engine::Math::SIMD;

        // 1. Calculate Center and Extents (most efficient way to transform an AABB)
        SIMDVector3D center = (maxBounds + minBounds) * 0.5f;
        SIMDVector3D extents = (maxBounds - minBounds) * 0.5f;

        // 2. Transform the center normally (Matrix * Vector applies rotation and translation)
        SIMDVector3D newCenter = m * center;

        // Absolute Matrix columns (Strip the sign bit from columns 0, 1, and 2)
        // 3. To transform extents, we must multiply by the ABSOLUTE values of the rotation matrix to get the absolute value of the columns. 
        Float4 absCol0 = Abs(m.col[0]);
        Float4 absCol1 = Abs(m.col[1]);
        Float4 absCol2 = Abs(m.col[2]);

        // 4. Transform Extents (using the absolute columns and broadcasted extents)
        Float4 ex = BroadcastX(extents.reg);
        Float4 ey = BroadcastY(extents.reg);
        Float4 ez = BroadcastZ(extents.reg);

        // 5. Transform Extents (using the absolute columns)
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
        Matrix4 mat = Identity();
        float fovY_rad = fovY_degrees * Engine::Math::Constants::DEG_TO_RAD;
        float tanHalfFovY = Engine::Math::Functions::FastTan(fovY_rad * 0.5f);

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
        
        // 2. Fetch the hardware SIMD vector from our bridge
        SIMDVector3D simdF = worldForward.toFloatVector();
        
        // 3. Extract it safely back into a scalar Vector3D for this matrix math! 
        Vector3D f(simdF.x(), simdF.y(), simdF.z());
        
        // Use your fast scalar dot product to normalize
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) {
            // Replaced std::sqrt with FastSqrt!
            f *= (1.0f / Engine::Math::Functions::FastSqrt(fLenSq)); 
        }

        // 4. Right Vector (X)
        Vector3D r = f.cross(upVec);
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) {
            // Replaced std::sqrt with FastSqrt!
            r *= (1.0f / Engine::Math::Functions::FastSqrt(rLenSq)); 
        }

        // 5. Up Vector (Y)
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

    return Vector3D(x, y, z, w);
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
                Engine::Math::SIMD::Float4 lenSqReg = Engine::Math::SIMD::Set1(lengthSq);
                Engine::Math::SIMD::Float4 invLenReg = Engine::Math::SIMD::ReciprocalSqrt(lenSqReg);
                planes[i] = planes[i] * Engine::Math::SIMD::ExtractX(invLenReg);
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
