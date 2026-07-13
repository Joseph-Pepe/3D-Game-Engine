#pragma once
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <format>
#include <utility>
#include <print> // std::println

// ==================================================
// INSTRUCTION SET ARCHITECTURES (ISA)
// ==================================================
/*
    - x86_64 (AMD Zen 2): Xbox Series X/S & PS5 | AVX2 (256-bit registers)
    - ARM64 (ARM): Nintendo Switch 2 (Nvidia Tegra) & Apple Silicon (M1/M2/M3) | ARM NEON (128-bit registers)
    - Legacy: Baseline Legacy PC | SSE4.1 (128 bit registers)
*/

// --- HARDWARE DETECTION ---
#if defined(__F16C__) && !defined(_MSC_VER)
    #include <f16cintrin.h> // Required for Clang/GCC on Linux
#endif

#if defined(__AVX512F__)
    #include <immintrin.h> // SIMD intrinsics (AVX, SSE (128-bit), MMX (64-bit))

    // AVX512: Next-Gen (AVX512 silicon intrinsically includes AVX2 and SSE4.1)
    #define ENGINE_ARCH_AVX512 1

    // AVX512 implicitly guarantees AVX2 and SSE4.1 support 
    #define ENGINE_ARCH_AVX2 1
    #define ENGINE_ARCH_SSE41 1
#elif defined(__AVX2__)
    #include <immintrin.h>

    // AVX2: Xbox Series X/S, PS5, Modern PC
    #define ENGINE_ARCH_AVX2 1

    // AVX2 implicitly guarantees SSE4.1 support
    #define ENGINE_ARCH_SSE41 1
#elif defined(__SSE4_1__)
    // SSE4.1: Legacy PC Fallback
    #include <immintrin.h> // immintrin handles all x86 SIMD headers

    // SSE 4.1 hardware (pre-2012 CPU) does not support Float16 compression, it belongs to the FC16 hardware extension.
    #define ENGINE_ARCH_SSE41 1

    // Enforces [SSE4.1 + FC16 extension]
    #if !defined(__F16C__) && (defined(__clang__) || defined(__GNUC__))
        #error "Engine Compiler Error: SSE4.1 fallback strictly requires the F16C extension for Float16 memory compression. Please compile with -mf16c."
    #endif
    
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM NEON: Apple Silicon, Switch 2, Android, Windows on ARM
    #include <arm_neon.h>
    #define ENGINE_ARCH_NEON 1
#else
    #error "Engine Compiler Error: Unsupported CPU architecture. AVX512, AVX2, NEON, or SSE4.1 instruction sets are strictly required."
#endif

// --- HARDWARE EXTENSIONS ---
#if defined(__ARM_FEATURE_SVE)   
    // Apple Silicon (M4+)
    #include <arm_sve.h>
    #define ENGINE_ARCH_SVE 1
#endif

// ===================================================
// UNIFORM MEMORY ALIGNMENT MACROS (MSVC, Clang, GCC)
// ===================================================
/* 
    - Cache line sizes are typically 64 bytes on modern CPUs.
    - Vulkan/DirectX require 16-byte alignment for vec4.
    - Prevents the compiler from padding our structs differently on a Nintendo Switch vs PC. 
*/
#define CACHE_CHUNK_ALIGN_16 alignas(16)
#define CACHE_CHUNK_ALIGN_32 alignas(32)
#define CACHE_CHUNK_ALIGN_64 alignas(64)

// ==================================================
// LOOP UNROLLING (COMPILER OPTIMIZATION)
// ==================================================
/*
    - MSVC lacks an unroll pragma. 
    - We use 'ivdep' to guarantee no memory aliasing, which gives the MSVC optimizer the green light to aggressively unroll it for us.
*/
#if defined(_MSC_VER) && !defined(__clang__)
    #define ENGINE_UNROLL_4 __pragma(loop(ivdep))
#elif defined(__clang__)
    #define ENGINE_UNROLL_4 _Pragma("unroll 4")
#elif defined(__GNUC__)
    #define ENGINE_UNROLL_4 _Pragma("GCC unroll 4")
#else
    #define ENGINE_UNROLL_4 // Fallback to nothing if unsupported
#endif

// ==================================================
// FORCE INLINE (COMPILER OPTIMIZATION)
// ==================================================
/*
    - inline is a suggestion to the compiler.
    - __forceinline will force the compiler to flatten the math directly into the execution path.
*/
#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define FORCE_INLINE inline
#endif

// ==================================================
// RESTRICT (COMPILER OPTIMIZATION)
// ==================================================
/*
    - Makes a promise to the compiler that data arrays never overlap.
    - Cross-platform restrict macro for pointer aliasing guarantees.
*/
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT // Fallback to nothing if unsupported
#endif

void LogHardwareArchitecture() {
    #if defined(ENGINE_ARCH_AVX512)
        std::println("[AVX-512]: Supercomputer / Next-Gen (x86_64) architecture detected.");
    #elif defined(ENGINE_ARCH_AVX2)
        std::println("[AVX2]: Intel/AMD (x86_64) based architecture detected.");
    #elif defined(ENGINE_ARCH_NEON)
        std::println("[ARM64]: ARM based architecture detected.");
    #elif defined(ENGINE_ARCH_SSE41)
        std::println("[SSE4.1]: Legacy based architecture detected.");
    #endif
}

// Scalar Domain (1D Scalar Math): strictly for operations that only operate on scalar floats.
namespace Engine::Math::ScalarFunctions {
    // ======================================================================
    // SCALAR HARDWARE SQUARE ROOT (FPU)
    // ======================================================================
    /*
        - Bypasses the `<cmath>` standard library overhead and 'errno' domain checks.
        - Maps directly to the CPU's dedicated scalar FPU square root instruction.
        - Exactly as accurate as std::sqrt, but significantly faster due to zero branching (i.e., std::sqrt replacement).
    */
    FORCE_INLINE float sqrt(float x) {
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
    FORCE_INLINE float floor(float x) {
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

    // ======================================================================
    // SCALAR HARDWARE CEILING (FPU)
    // ======================================================================
    /*
        - Bypasses `<cmath>` overhead.
        - Exactly as accurate as std::ceil, but compiles to a single instruction.
    */
    FORCE_INLINE float ceil(float x) {
        #ifdef MATH_ISA_ARM
            // --- ARM APPLE SILICON / MOBILE (ARM64) ---
            #if defined(__clang__) || defined(__GNUC__)
                // Compiles directly down to a single hardware 'frintp' instruction.
                return __builtin_ceilf(x);
            #else
                // Fallback: Round to Plus Infinity
                return vgetq_lane_f32(vrndpq_f32(vsetq_lane_f32(x, vdupq_n_f32(0.0f), 0)), 0);
            #endif
        #else
            // --- INTEL / AMD PC (SSE4.1) ---
            // _mm_ceil_ss calculates the ceiling of the lowest 32-bits.
            return _mm_cvtss_f32(_mm_ceil_ss(_mm_set_ss(x), _mm_set_ss(x)));
        #endif
    }

    // ======================================================================
    // SCALAR HARDWARE INVERSE SQUARE ROOT (FPU) (~7 Clock Cycles)
    // ======================================================================
    /*
        - Instantly estimates 1/sqrt(x) using dedicated hardware.
        - Refines the estimate to 23-bit float precision using Newton-Raphson.
        - Significantly faster than 1.0f / std::sqrt(x).
    */
    FORCE_INLINE float rsqrt(float x) {
        #ifdef MATH_ISA_ARM
            // --- ARM APPLE SILICON / MOBILE (ARM64) ---
            // ARM has dedicated scalar instructions for this!
            // 1. Get the hardware approximation
            float approx = vrsqrte_f32(x);
            // 2. Newton-Raphson refinement step
            float step = vrsqrts_f32(x, approx * approx);
            // 3. Final 23-bit accurate result
            return approx * step;
        #else
            // --- INTEL / AMD PC (SSE) ---
            // 1. Load x into the lowest 32-bits
            __m128 scalar_x = _mm_set_ss(x);
            
            // 2. Hardware approximation (12-bit accuracy)
            __m128 approx = _mm_rsqrt_ss(scalar_x);
            
            // 3. Newton-Raphson: y = y * (1.5 - 0.5 * x * y * y)
            __m128 half_x = _mm_mul_ss(_mm_set_ss(0.5f), scalar_x);
            __m128 y_sq = _mm_mul_ss(approx, approx);
            __m128 term = _mm_sub_ss(_mm_set_ss(1.5f), _mm_mul_ss(half_x, y_sq));
            __m128 result = _mm_mul_ss(approx, term);
            
            return _mm_cvtss_f32(result);
        #endif
    }

    // ======================================================================
    // SCALAR HARDWARE RECIPROCAL (FPU)
    // ======================================================================
    /*
        - Replaces (1.0f / x) division with hardware approximation.
        - Refines to 23-bit precision.
        - Prevents division pipeline stalls.
    */
    FORCE_INLINE float rcp(float x) {
        #ifdef MATH_ISA_ARM
            // --- ARM APPLE SILICON / MOBILE (ARM64) ---
            // 1. Hardware approximation
            float approx = vrecpe_f32(x);
            // 2. Hardware refinement step
            float step = vrecps_f32(x, approx);
            // 3. Final 23-bit result
            return approx * step;
        #else
            // --- INTEL / AMD PC (SSE) ---
            __m128 scalar_x = _mm_set_ss(x);
            
            // 1. Hardware approximation
            __m128 approx = _mm_rcp_ss(scalar_x);
            
            // 2. Newton-Raphson: y = y * (2.0 - x * y)
            __m128 term = _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(scalar_x, approx));
            __m128 result = _mm_mul_ss(approx, term);
            
            return _mm_cvtss_f32(result);
        #endif
    }

    // Fast pure scalar absolute value. Compiles down to a single instruction (zero-cast bitwise clear).
    FORCE_INLINE constexpr float abs(float v) {
        // Treat the float as an integer, strip the 31st sign bit, and treat it as a float again.
        uint32_t i = std::bit_cast<uint32_t>(v);
        i &= 0x7FFFFFFF; // Clear the sign bit
        return std::bit_cast<float>(i);
    }
}

// ===================================
// SIMD Intrinsics & Memory Alignment
// ===================================
/*
    - SIMD: Single Instruction (SI), Multiple Data (MD)
    - SIMD and aligned memory is a great way to squeeze performance out of modern CPUs.
    - Allows us to perform operations on all components (x, y, z) simultaneoulsy in a single CPU cycle.
    - Compiler intrinsics are special functions that map directly to specific assembly instructions on your CPU.
*/

namespace Engine::ISAArch {
    // ==========================================
    // ABI NAMESPACE (Hardware Tags)
    // ==========================================
    namespace simd_abi {
        struct scalar {};
        struct sse41 {};
        struct avx2 {};
        struct neon {};
        struct avx512 {};
        struct sve {}; // Scalable Vector Extension (vector sizes change at runtime based on the silicon from 128-bits to 2048-bits)

        // C++26 'native' alias: Automatically deduces the best hardware vector length at compile time based on your compiler flags (e.g., /arch:AVX2).
        template <typename T>
        #if ENGINE_ARCH_AVX512
            using native = avx512;   // Supercomputers (512-bit (16 floats))
        #elif ENGINE_ARCH_AVX2
            using native = avx2;     // Xbox/PS5/PC (256-bit (8 floats))
        #elif ENGINE_ARCH_NEON
            using native = neon;     // Nintendo Switch 2 / Apple Silicon (128-bit (4 floats))
        #elif ENGINE_ARCH_SSE41
            using native = sse41;    // Legacy PC Fallback (128-bit (4 floats))
        #else
            using native = scalar;   // Fallback
        #endif
    }

    // ==========================================
    // BACKEND INTRINSICS (Traits/Storage)
    // ==========================================
    namespace detail {
        // Primary template (Undefined)
        template <typename T, typename Abi> struct simd_traits;

        #if ENGINE_ARCH_AVX512
            // ========================================================
            // --- AVX-512 BACKEND (16-Wide Processing) ---
            // ========================================================
            template <> struct simd_traits<float, simd_abi::avx512> {
                using register_type = __m512;
                using mask_type     = __mmask16; // Dedicated 16-bit hardware mask!
                static constexpr int size = 16;
                
                static inline register_type broadcast(float v) { return _mm512_set1_ps(v); }
                static inline register_type load(const float* mem) { return _mm512_loadu_ps(mem); }
                static inline void store(float* mem, register_type v) { _mm512_storeu_ps(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_ps(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm512_mul_ps(a, b); }
                
                // Mask Generation directly returns a __mmask16 integer
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ); }
                
                // Mask Logic operates on standard CPU integer registers, completely bypassing the vector ALU
                static inline mask_type mask_not(mask_type a) { return _knot_mask16(a); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask16(a, b); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask16(a, b); }

                static inline bool mask_any(mask_type a) { return a != 0; }
                static inline bool mask_all(mask_type a) { return a == 0xFFFF; }
                
                // Native hardware blending using the dedicated mask register
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm512_mask_blend_ps(mask, false_v, true_v);
                }

                static inline register_type concat(__m256 a, __m256 b) {
                    return _mm512_insertf32x8(_mm512_castps256_ps512(a), b, 1);
                }
                static inline void split(register_type a, __m256& out_low, __m256& out_high) {
                    out_low = _mm512_castps512_ps256(a);
                    out_high = _mm512_extractf32x8_ps(a, 1);
                }
                
                // AVX-512 FMA
                static inline register_type fmadd(register_type a, register_type b, register_type c) {
                    return _mm512_fmadd_ps(a, b, c);
                }
                
                static inline float reduce_add(register_type a) {
                    return _mm512_reduce_add_ps(a); // AVX-512 finally has native horizontal reduction!
                }

                static inline float reduce_min(register_type a) { return _mm512_reduce_min_ps(a); }
                static inline float reduce_max(register_type a) { return _mm512_reduce_max_ps(a); }

                static inline int mask_popcount(mask_type a) {
                    #ifdef _MSC_VER
                        return __popcnt16(a);
                    #else
                        return __builtin_popcount(a);
                    #endif
                }
                
                static inline int mask_find_first_set(mask_type a) {
                    if (a == 0) return -1;
                    #ifdef _MSC_VER
                        unsigned long index;
                        _BitScanForward(&index, a);
                        return static_cast<int>(index);
                    #else
                        return __builtin_ctz(a);
                    #endif
                }

                // --- AVX-512 FLOAT ARITHMETIC ---
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_ps(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm512_div_ps(a, b); }
                static inline register_type min(register_type a, register_type b) { return _mm512_min_ps(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_ps(a, b); }
                static inline register_type sqrt(register_type a) { return _mm512_sqrt_ps(a); }

                // --- AVX-512 NEWTON-RAPHSON ---
                static inline register_type rsqrt(register_type a) { 
                    __m512 approx = _mm512_rsqrt14_ps(a); // AVX-512 uses a 14-bit estimate native instruction
                    __m512 half_a = _mm512_mul_ps(_mm512_set1_ps(0.5f), a);
                    __m512 x0_sq  = _mm512_mul_ps(approx, approx);
                    return _mm512_mul_ps(approx, _mm512_fnmadd_ps(half_a, x0_sq, _mm512_set1_ps(1.5f)));
                }
                static inline register_type rcp(register_type a) { 
                    __m512 approx = _mm512_rcp14_ps(a);
                    return _mm512_mul_ps(approx, _mm512_fnmadd_ps(a, approx, _mm512_set1_ps(2.0f)));
                }

                // --- AVX-512 BITWISE & LOGIC ---
                static inline register_type abs(register_type a) { return _mm512_and_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF))); } // Uses the floating-point AND instruction
                static inline register_type floor(register_type a) { return _mm512_floor_ps(a); }
                static inline register_type ceil(register_type a) { return _mm512_ceil_ps(a); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); }

                // --- AVX-512 BITWISE & LOGIC (Pure AVX-512F Compliant) ---
                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_ps(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_ps(a, b); }
                static inline register_type bit_or(register_type a, register_type b)  { return _mm512_or_ps(a, b); }

                static inline register_type negate(register_type a) { 
                    // Uses the floating-point XOR instruction with -0.0f
                    return _mm512_xor_ps(a, _mm512_set1_ps(-0.0f)); 
                }

                // --- AVX-512 CASTING, SHUFFLE, & GATHER ---
                template <typename Target>
                static inline __m512i cast_to(register_type a) { 
                    static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t supported.");
                    return _mm512_cvttps_epi32(a); 
                }

                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) { return _mm512_shuffle_ps(a, a, _MM_SHUFFLE(i3, i2, i1, i0)); }

                static inline register_type gather(const float* base_addr, __m512i indices) { return _mm512_i32gather_ps(indices, base_addr, 4); }

                // Loads 256-bits of RAM and expands to a 512-bit register
                static inline register_type load_half(const uint16_t* mem) {
                    return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)));
                }
                // Compresses a 512-bit register down to 256-bits of RAM
                static inline void store_half(uint16_t* mem, register_type v) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
                }
            };

            // --- AVX-512 DOUBLE TRAITS (8 Elements per Register) ---
            template <> struct simd_traits<double, simd_abi::avx512> {
                using register_type = __m512d;
                using mask_type     = __mmask8; 
                static constexpr int size = 8;
                
                static inline register_type broadcast(double v) { return _mm512_set1_pd(v); }
                static inline register_type load(const double* mem) { return _mm512_loadu_pd(mem); }
                static inline void store(double* mem, register_type v) { _mm512_storeu_pd(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_pd(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_pd(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm512_mul_pd(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm512_div_pd(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm512_min_pd(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_pd(a, b); }
                static inline register_type sqrt(register_type a) { return _mm512_sqrt_pd(a); }
                
                // Precision is paramount for doubles, so we use actual division instead of Newton-Raphson approximation
                static inline register_type rsqrt(register_type a) { return _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a)); }
                static inline register_type rcp(register_type a) { return _mm512_div_pd(_mm512_set1_pd(1.0), a); }

                static inline register_type abs(register_type a) { return _mm512_and_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF))); }
                static inline register_type floor(register_type a) { return _mm512_floor_pd(a); }
                static inline register_type ceil(register_type a) { return _mm512_ceil_pd(a); }

                static inline register_type fmadd(register_type a, register_type b, register_type c) { return _mm512_fmadd_pd(a, b, c); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ); }

                static inline mask_type mask_not(mask_type a) { return _knot_mask8(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask8(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask8(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_pd(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm512_or_pd(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_pd(a, b); }
                static inline register_type negate(register_type a) { return _mm512_xor_pd(a, _mm512_set1_pd(-0.0)); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm512_mask_blend_pd(mask, false_v, true_v); }
                static inline double reduce_add(register_type a) { return _mm512_reduce_add_pd(a); }

                // Gather uses a 256-bit index register to fetch 8 doubles
                static inline register_type gather(const double* base_addr, __m512i indices) { 
                    return _mm512_i32gather_pd(_mm512_castsi512_si256(indices), base_addr, 8); 
                }

                static inline bool mask_any(mask_type a) { return !_ktestz_mask16_blank(a, a); } // Generates a hardware 'KTEST' instruction. Checks if the mask is empty entirely inside the vector mask registers (without touching standard scalar comparison registers).
                static inline bool mask_all(mask_type a) { return a == 0xFF; }

                template <int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7>
                static inline register_type shuffle(register_type a) { return a; } // (AVX-512 cross-lane swizzling is highly complex, omit for brevity unless needed)
            };

            // --- AVX-512 UINT8 TRAITS (64 Elements per Register) ---
            template <> struct simd_traits<uint8_t, simd_abi::avx512> {
                using register_type = __m512i;
                using mask_type     = __mmask64; // Massive 64-bit hardware mask!
                static constexpr int size = 64;

                static inline register_type broadcast(uint8_t v) { return _mm512_set1_epi8(v); }
                static inline register_type load(const uint8_t* mem) { return _mm512_loadu_si512(reinterpret_cast<const void*>(mem)); }
                static inline void store(uint8_t* mem, register_type v) { _mm512_storeu_si512(reinterpret_cast<void*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_epi8(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_epi8(a, b); }
                
                // Saturating Arithmetic (Clamps to 255 or 0)
                static inline register_type add_sat(register_type a, register_type b) { return _mm512_adds_epu8(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm512_subs_epu8(a, b); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmpgt_epu8_mask(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmplt_epu8_mask(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmpeq_epi8_mask(a, b); }

                static inline mask_type mask_not(mask_type a) { return _knot_mask64(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask64(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask64(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm512_min_epu8(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_epu8(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_epi32(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm512_or_epi32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_epi32(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm512_mask_blend_epi8(mask, false_v, true_v);
                }

                static inline bool mask_any(mask_type a) { return a != 0; }
                static inline bool mask_all(mask_type a) { return a == 0xFFFFFFFFFFFFFFFFULL; }
            };

            // --- AVX-512 INT16 TRAITS (32 Elements per Register) ---
            template <> struct simd_traits<int16_t, simd_abi::avx512> {
                using register_type = __m512i;
                using mask_type     = __mmask32; // 32-bit hardware mask!
                static constexpr int size = 32;

                static inline register_type broadcast(int16_t v) { return _mm512_set1_epi16(v); }
                static inline register_type load(const int16_t* mem) { return _mm512_loadu_si512(reinterpret_cast<const void*>(mem)); }
                static inline void store(int16_t* mem, register_type v) { _mm512_storeu_si512(reinterpret_cast<void*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_epi16(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_epi16(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm512_mullo_epi16(a, b); }
                
                // SIGNED Saturating Arithmetic (Audio Clipping Prevention)
                static inline register_type add_sat(register_type a, register_type b) { return _mm512_adds_epi16(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm512_subs_epi16(a, b); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmpgt_epi16_mask(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmplt_epi16_mask(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmpeq_epi16_mask(a, b); }

                static inline mask_type mask_not(mask_type a) { return _knot_mask32(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask32(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm512_min_epi16(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_epi16(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_epi32(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm512_or_epi32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_epi32(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm512_slli_epi16(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm512_srai_epi16(a, imm); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm512_mask_blend_epi16(mask, false_v, true_v);
                }

                static inline bool mask_any(mask_type a) { return a != 0; }
                static inline bool mask_all(mask_type a) { return a == 0xFFFFFFFF; }

                // Mismatch guard: Cannot easily cast 32 ints into 16 floats in one register
                template <typename Target>
                static inline __m512 cast_to(register_type a) { 
                    static_assert(sizeof(Target) == 0, "SIMD Mismatch: Cannot cast 32x16-bit to 16x32-bit directly."); return _mm512_setzero_ps(); 
                }
            };

            // --- AVX-512 UINT32 TRAITS ---
            template <> struct simd_traits<uint32_t, simd_abi::avx512> {
                using register_type = __m512i;
                using mask_type     = __mmask16; 
                static constexpr int size = 16;

                template <typename Target>
                static inline __m512 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "uint32_t-to-float cast.");
                    return _mm512_cvtepi32_ps(a); 
                }
                
                static inline register_type broadcast(uint32_t v) { return _mm512_set1_epi32(v); }
                static inline register_type load(const uint32_t* mem) { return _mm512_loadu_si512(reinterpret_cast<const void*>(mem)); }
                static inline void store(uint32_t* mem, register_type v) { _mm512_storeu_si512(reinterpret_cast<void*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm512_mullo_epi32(a, b); }
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmpgt_epi32_mask(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmplt_epi32_mask(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmpeq_epi32_mask(a, b); }

                static inline mask_type mask_not(mask_type a) { return _knot_mask16(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask16(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask16(a, b); }

                // Bitwise Math
                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_epi32(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm512_or_epi32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_epi32(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm512_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm512_srli_epi32(a, imm); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm512_mask_blend_epi32(mask, false_v, true_v);
                }

                static inline uint32_t reduce_add(register_type a) {
                    return _mm512_reduce_add_epi32(a); 
                }

                // Hardware Gather for AVX-512 (Scale = 4 bytes)
                static inline register_type gather(const uint32_t* base_addr, __m512i indices) {
                    return _mm512_i32gather_epi32(indices, base_addr, 4); 
                }

                // Boundary Math
                static inline register_type min(register_type a, register_type b) { return _mm512_min_epu32(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_epu32(a, b); }

                // Mask Evaluation
                static inline bool mask_any(mask_type a) { return a != 0; }
                static inline bool mask_all(mask_type a) { return a == 0xFFFF; }

                // Hardware Vector Swizzling (Symmetric across 128-bit lanes)
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    return _mm512_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0));
                }

            };

            template <> struct simd_traits<int32_t, simd_abi::avx512> {
                using register_type = __m512i;
                using mask_type     = __mmask16; 
                static constexpr int size = 16;

                template <typename Target>
                static inline __m512 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "int32_t-to-float cast.");
                    return _mm512_cvtepi32_ps(a); 
                }
                
                static inline register_type broadcast(int32_t v) { return _mm512_set1_epi32(v); }
                static inline register_type load(const int32_t* mem) { return _mm512_loadu_si512(reinterpret_cast<const void*>(mem)); }
                static inline void store(int32_t* mem, register_type v) { _mm512_storeu_si512(reinterpret_cast<void*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm512_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm512_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm512_mullo_epi32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm512_min_epi32(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm512_max_epi32(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm512_xor_epi32(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm512_or_epi32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm512_and_epi32(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm512_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm512_srai_epi32(a, imm); } // Arithmetic Shift

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm512_cmpgt_epi32_mask(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm512_cmplt_epi32_mask(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm512_cmpeq_epi32_mask(a, b); }

                static inline mask_type mask_not(mask_type a) { return _knot_mask16(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _kand_mask16(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _kor_mask16(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm512_mask_blend_epi32(mask, false_v, true_v); }
                static inline int32_t reduce_add(register_type a) { return _mm512_reduce_add_epi32(a); }
                static inline register_type gather(const int32_t* base_addr, __m512i indices) { return _mm512_i32gather_epi32(indices, base_addr, 4); }
                static inline bool mask_any(mask_type a) { return a != 0; }
                static inline bool mask_all(mask_type a) { return a == 0xFFFF; }
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) { return _mm512_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0)); }
            };
        #endif

        #if ENGINE_ARCH_AVX2 || ENGINE_ARCH_SSE41
            // ========================================================
            // --- SSE4.1 BACKEND (128-bit x64 Base) ---
            // ========================================================
            // SSE 4.1 traits are mandatory, even if the user has an AVX2 capable PC. 
            // We must use SSE registers for 4x4 matrices and 3D vectors to guarantee 16-byte data structures.
            template <> struct simd_traits<float, simd_abi::sse41> {
                using register_type = __m128;
                using mask_type     = __m128; 
                static constexpr int size = 4;
                
                static inline register_type broadcast(float v) { return _mm_set1_ps(v); }
                static inline register_type load(const float* mem) { return _mm_loadu_ps(mem); }
                static inline void store(float* mem, register_type v) { _mm_storeu_ps(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm_add_ps(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm_mul_ps(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_ps(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm_div_ps(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_ps(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_ps(a, b); }

                static inline register_type rsqrt(register_type a) { 
                    __m128 approx = _mm_rsqrt_ps(a);

                    // Newton-Raphson: x1 = x0 * (1.5 - (0.5 * a * x0 * x0))
                    __m128 half_a = _mm_mul_ps(_mm_set1_ps(0.5f), a);
                    __m128 x0_sq = _mm_mul_ps(approx, approx);
                    return _mm_mul_ps(approx, _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(half_a, x0_sq)));
                }

                static inline register_type rcp(register_type a) { 
                    // _mm_rcp_ps: is fast, but is an approximation with 11 to 14 bits of precision. If you use this to normalize vectors, this precision loss will cause objects to gradually drift, jitter or fail collision detection at world-space extremes.
                    __m128 approx = _mm_rcp_ps(a);

                    // Newton-Raphson: x1 = x0 * (2.0 - a * x0) to refine 11-bits into an accurate 23-bit float, 
                    return _mm_mul_ps(approx, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(a, approx)));
                }

                static inline register_type abs(register_type a) { 
                    return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF))); 
                }
                
                // SSE4.1 specific rounding intrinsics
                static inline register_type floor(register_type a) { return _mm_round_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
                static inline register_type ceil(register_type a) { return _mm_round_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC); }
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm_cmpgt_ps(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm_cmplt_ps(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_ps(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    // return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1))); 

                    // Zero-cost generation of 0xFFFFFFFF across all 4 lanes
                    __m128 all_ones = _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps());
                    return _mm_xor_ps(a, all_ones);

                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_ps(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_ps(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm_xor_ps(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_ps(a, b); }
                static inline register_type bit_or(register_type a, register_type b)  { return _mm_or_ps(a, b); }

                static inline register_type negate(register_type a) {
                    return _mm_xor_ps(a, _mm_set1_ps(-0.0f)); 
                }
                
                // SSE4.1 introduces native variable blending
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm_blendv_ps(false_v, true_v, mask);
                }

                // Emulated FMA for SSE processors (SSE 4.1 does not have native hardware Fused-Multiply-Add).
                static inline register_type fmadd(register_type a, register_type b, register_type c) {
                    return _mm_add_ps(_mm_mul_ps(a, b), c); 
                }
                
                static inline register_type sqrt(register_type a) {
                    return _mm_sqrt_ps(a);
                }

                template <typename Target>
                static inline __m128i cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t supported.");
                    return _mm_cvttps_epi32(a); 
                }

                // SSE Horizontal add
                static inline float reduce_add(register_type a) {
                    __m128 shuf = _mm_movehdup_ps(a);
                    __m128 sums = _mm_add_ps(a, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    return _mm_cvtss_f32(sums);
                }

                static inline float reduce_min(register_type a) {
                    __m128 shuf = _mm_movehdup_ps(a);
                    __m128 mins = _mm_min_ps(a, shuf);
                    shuf = _mm_movehl_ps(shuf, mins);
                    mins = _mm_min_ss(mins, shuf);
                    return _mm_cvtss_f32(mins);
                }
                static inline float reduce_max(register_type a) {
                    __m128 shuf = _mm_movehdup_ps(a);
                    __m128 maxs = _mm_max_ps(a, shuf);
                    shuf = _mm_movehl_ps(shuf, maxs);
                    maxs = _mm_max_ss(maxs, shuf);
                    return _mm_cvtss_f32(maxs);
                }

                static inline bool mask_any(mask_type a) { return _mm_movemask_ps(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_ps(a) == 0x0F; }

                static inline int mask_popcount(mask_type a) {
                    int mask_val = _mm_movemask_ps(a); // Use _mm_movemask_ps for SSE4.1
                    #ifdef _MSC_VER
                        return __popcnt(mask_val);
                    #else
                        return __builtin_popcount(mask_val);
                    #endif
                }
                static inline int mask_find_first_set(mask_type a) {
                    int mask_val = _mm_movemask_ps(a); // Use _mm_movemask_ps for SSE4.1
                    if (mask_val == 0) return -1;
                    #ifdef _MSC_VER
                        unsigned long index;
                        _BitScanForward(&index, mask_val);
                        return static_cast<int>(index);
                    #else
                        return __builtin_ctz(mask_val);
                    #endif
                }

                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    return _mm_shuffle_ps(a, a, _MM_SHUFFLE(i3, i2, i1, i0));
                }

                // // Optimized SSE4.1 Emulated Gather (No Memory Store Required) (SSE lacks AVX2's gather instruction)
                static inline register_type gather(const float* base_addr, __m128i indices) {
                    return _mm_set_ps(
                        base_addr[_mm_extract_epi32(indices, 3)], 
                        base_addr[_mm_extract_epi32(indices, 2)], 
                        base_addr[_mm_extract_epi32(indices, 1)], 
                        base_addr[_mm_extract_epi32(indices, 0)]
                    );
                }

                // ==================================
                // MEMORY COMPRESSION (Float16)
                // ==================================
                // Loads 64-bits of RAM (4 x 16-bit floats) and expands to a 128-bit register
                static inline register_type load_half(const uint16_t* mem) {
                    // Cast the pointer to __m128i* and load the 64-bit chunk natively
                    __m128i half_data = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mem));
                    return _mm_cvtph_ps(half_data);
                }
                // Compresses a 128-bit register down to 64-bits of RAM
                static inline void store_half(uint16_t* mem, register_type v) {
                    __m128i half_data = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(mem), half_data);
                }
            };

            // --- SSE4.1 DOUBLE TRAITS (2 Elements per Register) ---
            template <> struct simd_traits<double, simd_abi::sse41> {
                using register_type = __m128d;
                using mask_type     = __m128d; 
                static constexpr int size = 2;
                
                static inline register_type broadcast(double v) { return _mm_set1_pd(v); }
                static inline register_type load(const double* mem) { return _mm_loadu_pd(mem); }
                static inline void store(double* mem, register_type v) { _mm_storeu_pd(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm_add_pd(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_pd(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm_mul_pd(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm_div_pd(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_pd(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_pd(a, b); }
                static inline register_type sqrt(register_type a) { return _mm_sqrt_pd(a); }

                static inline register_type rsqrt(register_type a) { return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a)); }
                static inline register_type rcp(register_type a) { return _mm_div_pd(_mm_set1_pd(1.0), a); }

                static inline register_type abs(register_type a) { return _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF))); }
                static inline register_type floor(register_type a) { return _mm_floor_pd(a); }
                static inline register_type ceil(register_type a) { return _mm_ceil_pd(a); }

                static inline register_type fmadd(register_type a, register_type b, register_type c) { return _mm_add_pd(_mm_mul_pd(a, b), c); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm_cmpgt_pd(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm_cmplt_pd(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_pd(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m128d all_ones = _mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd());
                    return _mm_xor_pd(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_pd(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_pd(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm_xor_pd(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm_or_pd(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_pd(a, b); }
                static inline register_type negate(register_type a) { return _mm_xor_pd(a, _mm_set1_pd(-0.0)); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm_blendv_pd(false_v, true_v, mask); }

                static inline double reduce_add(register_type a) {
                    __m128d sums = _mm_add_pd(a, _mm_unpackhi_pd(a, a));
                    return _mm_cvtsd_f64(sums);
                }

                static inline register_type gather(const double* base_addr, __m128i indices) {
                    // Extracts the lowest 2 indices from the 4 provided by the uint32_t batch
                    return _mm_set_pd(base_addr[_mm_extract_epi32(indices, 1)], base_addr[_mm_extract_epi32(indices, 0)]);
                }

                static inline bool mask_any(mask_type a) { return _mm_movemask_pd(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_pd(a) == 0x3; }

                template <int i0, int i1>
                static inline register_type shuffle(register_type a) {
                    return _mm_shuffle_pd(a, a, _MM_SHUFFLE2(i1, i0));
                }
            };

            // --- SSE4.1 UINT8 TRAITS (16 Elements per Register) ---
            template <> struct simd_traits<uint8_t, simd_abi::sse41> {
                using register_type = __m128i;
                using mask_type     = __m128i; 
                static constexpr int size = 16;
                
                static inline register_type broadcast(uint8_t v) { return _mm_set1_epi8(v); }
                static inline register_type load(const uint8_t* mem) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem)); }
                static inline void store(uint8_t* mem, register_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm_add_epi8(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_epi8(a, b); }
                
                static inline register_type add_sat(register_type a, register_type b) { return _mm_adds_epu8(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm_subs_epu8(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_epu8(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_epu8(a, b); }

                static inline register_type bit_or(register_type a, register_type b) { return _mm_or_si128(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_si128(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return _mm_xor_si128(a, b); }
                
                // Relational Unsigned Limits (XOR highest bit 0x80 to shift into signed range for the hardware comparator)
                static inline mask_type cmp_gt(register_type a, register_type b) { 
                    __m128i sign_flip = _mm_set1_epi8(char(0x80));
                    return _mm_cmpgt_epi8(_mm_xor_si128(a, sign_flip), _mm_xor_si128(b, sign_flip)); 
                }
                static inline mask_type cmp_lt(register_type a, register_type b) { 
                    __m128i sign_flip = _mm_set1_epi8(char(0x80));
                    return _mm_cmpgt_epi8(_mm_xor_si128(b, sign_flip), _mm_xor_si128(a, sign_flip)); 
                }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_epi8(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
                    return _mm_xor_si128(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_si128(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_si128(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm_blendv_epi8(false_v, true_v, mask);
                }

                static inline bool mask_any(mask_type a) { return _mm_movemask_epi8(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_epi8(a) == 0xFFFF; }
            };

            // --- SSE4.1 INT16 TRAITS (8 Elements per Register) ---
            template <> struct simd_traits<int16_t, simd_abi::sse41> {
                using register_type = __m128i;
                using mask_type     = __m128i; 
                static constexpr int size = 8;
                
                static inline register_type broadcast(int16_t v) { return _mm_set1_epi16(v); }
                static inline register_type load(const int16_t* mem) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem)); }
                static inline void store(int16_t* mem, register_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm_add_epi16(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_epi16(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm_mullo_epi16(a, b); }
                
                // SIGNED Saturating Arithmetic
                static inline register_type add_sat(register_type a, register_type b) { return _mm_adds_epi16(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm_subs_epi16(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_epi16(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_epi16(a, b); }

                static inline register_type bit_or(register_type a, register_type b) { return _mm_or_si128(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_si128(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return _mm_xor_si128(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm_slli_epi16(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm_srai_epi16(a, imm); } // Arithmetic (preserves negative sound wave phase)
                
                // Native Signed Comparison
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm_cmpgt_epi16(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm_cmplt_epi16(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_epi16(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
                    return _mm_xor_si128(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_si128(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_si128(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm_blendv_epi8(false_v, true_v, mask); // epi8 blending works identically for epi16 memory layouts
                }

                static inline bool mask_any(mask_type a) { return _mm_movemask_epi8(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_epi8(a) == 0xFFFF; }

                template <typename Target>
                static inline __m128 cast_to(register_type a) { static_assert(sizeof(Target) == 0, "SIMD Mismatch: Cannot cast 8x16-bit to 4x32-bit directly."); return _mm_setzero_ps(); }
            };

            // --- SSE4.1 UINT32 TRAITS ---
            template <> struct simd_traits<uint32_t, simd_abi::sse41> {
                using register_type = __m128i;
                using mask_type     = __m128i; 
                static constexpr int size = 4;

                template <typename Target>
                static inline __m128 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "uint32_t-to-float cast.");
                    return _mm_cvtepi32_ps(a); 
                }
                
                // Memory
                static inline register_type broadcast(uint32_t v) { return _mm_set1_epi32(v); }
                static inline register_type load(const uint32_t* mem) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem)); }
                static inline void store(uint32_t* mem, register_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), v); }
                
                // Arithmetic
                static inline register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm_mullo_epi32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_epu32(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_epu32(a, b); }

                // Bitwise Math
                static inline register_type bit_or(register_type a, register_type b) { return _mm_or_si128(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_si128(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm_srli_epi32(a, imm); }
                
                // Relational [XOR the highest bit (0x80000000) of both vectors]
                static inline mask_type cmp_gt(register_type a, register_type b) { 
                    __m128i sign_flip = _mm_set1_epi32(0x80000000);
                    return _mm_cmpgt_epi32(_mm_xor_si128(a, sign_flip), _mm_xor_si128(b, sign_flip)); 
                }
                static inline mask_type cmp_lt(register_type a, register_type b) { 
                    __m128i sign_flip = _mm_set1_epi32(0x80000000);
                    return _mm_cmpgt_epi32(_mm_xor_si128(b, sign_flip), _mm_xor_si128(a, sign_flip)); 
                }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }

                // Mask Logic
                static inline mask_type mask_not(mask_type a) { 
                    // return _mm_xor_si128(a, _mm_set1_epi32(-1)); 

                    // Zero-cost generation of all 1s
                    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
                    return _mm_xor_si128(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_si128(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_si128(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm_blendv_epi8(false_v, true_v, mask);
                }

                // Horizontal Integer Reduction
                static inline uint32_t reduce_add(register_type a) {
                    __m128i shuf = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i sums = _mm_add_epi32(a, shuf);
                    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                    sums = _mm_add_epi32(sums, shuf);
                    return _mm_cvtsi128_si32(sums);
                }

                static inline bool mask_any(mask_type a) { return _mm_movemask_ps(_mm_castsi128_ps(a)) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_ps(_mm_castsi128_ps(a)) == 0x0F; }

                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    return _mm_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0));
                }

                // Emulated Hardware Gather for SSE4.1 UINT32
                static inline register_type gather(const uint32_t* base_addr, __m128i indices) {
                    return _mm_set_epi32(
                        base_addr[_mm_extract_epi32(indices, 3)], 
                        base_addr[_mm_extract_epi32(indices, 2)], 
                        base_addr[_mm_extract_epi32(indices, 1)], 
                        base_addr[_mm_extract_epi32(indices, 0)]
                    );
                }
            };

            template <> struct simd_traits<int32_t, simd_abi::sse41> {
                using register_type = __m128i;
                using mask_type     = __m128i; 
                static constexpr int size = 4;

                template <typename Target>
                static inline __m128 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "int32_t-to-float cast.");
                    return _mm_cvtepi32_ps(a); 
                }
                
                static inline register_type broadcast(int32_t v) { return _mm_set1_epi32(v); }
                static inline register_type load(const int32_t* mem) { return _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem)); }
                static inline void store(int32_t* mem, register_type v) { _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm_mullo_epi32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm_min_epi32(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm_max_epi32(a, b); }

                static inline register_type bit_or(register_type a, register_type b) { return _mm_or_si128(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm_and_si128(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return _mm_xor_si128(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm_srai_epi32(a, imm); } // Arithmetic
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm_cmpgt_epi32(a, b); } // Native signed check
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm_cmpgt_epi32(b, a); } 
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm_cmpeq_epi32(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m128i all_ones = _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128());
                    return _mm_xor_si128(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm_and_si128(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm_or_si128(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm_blendv_epi8(false_v, true_v, mask); }

                static inline int32_t reduce_add(register_type a) {
                    __m128i shuf = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i sums = _mm_add_epi32(a, shuf);
                    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                    sums = _mm_add_epi32(sums, shuf);
                    return _mm_cvtsi128_si32(sums);
                }
                static inline bool mask_any(mask_type a) { return _mm_movemask_ps(_mm_castsi128_ps(a)) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_ps(_mm_castsi128_ps(a)) == 0x0F; }
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) { return _mm_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0)); }
                static inline register_type gather(const int32_t* base_addr, __m128i indices) {
                    return _mm_set_epi32(base_addr[_mm_extract_epi32(indices, 3)], base_addr[_mm_extract_epi32(indices, 2)], base_addr[_mm_extract_epi32(indices, 1)], base_addr[_mm_extract_epi32(indices, 0)]);
                }
            };
        #endif // ENGINE_ARCH_AVX2 || ENGINE_ARCH_SSE41

        #if ENGINE_ARCH_AVX2
            // ========================================================
            // --- AVX2 BACKEND (Xbox Series X, PS5, PC) ---
            // ========================================================
            template <> struct simd_traits<float, simd_abi::avx2> {
                using register_type = __m256;
                using mask_type     = __m256; // AVX2 masks are bit patterns in identical registers
                static constexpr int size = 8;
                
                static inline register_type broadcast(float v) { return _mm256_set1_ps(v); }
                static inline register_type load(const float* mem) { return _mm256_loadu_ps(mem); }
                static inline void store(float* mem, register_type v) { _mm256_storeu_ps(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_ps(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm256_mul_ps(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_ps(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm256_div_ps(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_ps(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm256_max_ps(a, b); }

                // ============================================
                // VECTOR RESIZING (SPLIT & CONCAT)
                // ============================================

                // Glues two SSE vectors into one AVX2 vector
                static inline register_type concat(__m128 a, __m128 b) {
                    return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
                }
                // Splits an AVX2 vector into two SSE vectors
                static inline void split(register_type a, __m128& out_low, __m128& out_high) {
                    out_low = _mm256_castps256_ps128(a);
                    out_high = _mm256_extractf128_ps(a, 1);
                }

                // 1 / sqrt(x)
                static inline register_type rsqrt(register_type a) { 
                    __m256 approx = _mm256_rsqrt_ps(a);
                    // Newton-Raphson: x1 = x0 * (1.5 - (0.5 * a * x0 * x0))
                    __m256 half_a = _mm256_mul_ps(_mm256_set1_ps(0.5f), a);
                    __m256 x0_sq = _mm256_mul_ps(approx, approx);
                    return _mm256_mul_ps(approx, _mm256_fnmadd_ps(half_a, x0_sq, _mm256_set1_ps(1.5f)));
                }

                // 1 / x
                static inline register_type rcp(register_type a) { 
                    __m256 approx = _mm256_rcp_ps(a);
                    // Newton-Raphson: x1 = x0 * (2.0 - a * x0)
                    // Uses hardware FMA to collapse the math!
                    return _mm256_mul_ps(approx, _mm256_fnmadd_ps(a, approx, _mm256_set1_ps(2.0f)));
                } 

                static inline register_type abs(register_type a) { 
                    // Clear the sign bit using bitwise AND
                    return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))); 
                }
                static inline register_type floor(register_type a) { return _mm256_floor_ps(a); }
                static inline register_type ceil(register_type a) { return _mm256_ceil_ps(a); }
                
                // Relational Intrinsic
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }

                // Mask Logic
                static inline mask_type mask_not(mask_type a) { 
                    // XOR with all 1s (0xFFFFFFFF) flips every bit, but forces the CPU to load a constant from the data section of the binary.
                    // return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));

                    // Generates a register of all 1s instantly without touching memory. Compares a zero register to itself. Result is strictly 0xFFFFFFFF across all lanes. 
                    __m256 all_ones = _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ);
                    return _mm256_xor_ps(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_ps(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_ps(a, b); }

                // ============================================
                // FLOATING-POINT BITWISE LOGIC
                // ============================================
                /*
                    - To negate a float, multiplying it by -1.0f routes the data through the CPUs floating-point multiplier (takes 4-5 clock cycles).
                    - Bitwise XOR the float with 0x80000000 (-0.0f in binary), instantly flips the sign bit in 1 clock cycle using the CPUs integer execution ports.
                */

                // Floating-Point Bitwise Logic (Engine Extension)
                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_ps(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_ps(a, b); }
                static inline register_type bit_or(register_type a, register_type b)  { return _mm256_or_ps(a, b); }

                // Fast Unary Negation
                static inline register_type negate(register_type a) {
                    // -0.0f evaluates to exactly 0x80000000. 
                    // XORing by this flips the sign bit on all 8 floats instantly.
                    return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f)); 
                }
                
                // Branchless Conditional Blending
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm256_blendv_ps(false_v, true_v, mask);
                }

                // Hardware Math
                static inline register_type fmadd(register_type a, register_type b, register_type c) {
                    return _mm256_fmadd_ps(a, b, c); // (a * b) + c
                }
                static inline register_type sqrt(register_type a) {
                    return _mm256_sqrt_ps(a);
                }

                // SIMD Casting AVX2 doesn't have a direct float-to-unsigned-int instruction, it only has float-to-signed-int. 
                template <typename Target>
                static inline __m256i cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t casting is currently implemented.");
                    return _mm256_cvttps_epi32(a); 
                }

                // ===================================================
                // AVX2 HORIZONTAL SUM  (REDUCTION TREE ALGORITHM)
                // ===================================================
                /*
                    - Takes a large set of data (e.g., 8 floats) and reduces it down to a single scalar value by repeatedly applying a math operator.
                    - SIMD registers are wide (i.e., horizontal), so we apply this process across lanes (i.e., horizontal reduction).
                    - Reduces the 256-bit register down to a single float by adding these lanes together (i.e., horizontal sum).
                    - AVX2 doesn't have a single-instruction horizontal add across 256 bits, so we fold it into two parts repeatedly.

                      Lanes:  [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]
                                \   /       \   /       \   /       \   /
                      Pairs:    [0+1]       [2+3]       [4+5]       [6+7]
                                    \         /             \         /
                      Quads:     [(0+1)+(2+3)]           [(4+5)+(6+7)]
                                        \                     /
                      Scalar:         [((0+1)+(2+3)) + ((4+5)+(6+7))]

                    - Collapses an 8-wide AVX2 register into a single scalar float.
                    - Significantly faster than extracting to a float[8] array and looping.
                    - Uses hardware folding: 8 -> 4 -> 2 -> 1
                */
               
                // Folds an 8-wide __m256 register down to a single scalar float inside the silicon registers without touching memory.
                static inline float reduce_add(register_type a) {
                    // Step 1: Split the 256-bit register into two 128-bit registers (i.e., fold 8-Wide(256-bit) into 2 separate 4-Wide(128-bit)).
                    __m128 vlow = _mm256_castps256_ps128(a);
                    __m128 vhigh = _mm256_extractf128_ps(a, 1); 

                    // Step 2: After extracting the high 128 bits, add them to the low 128 bits
                    vlow = _mm_add_ps(vlow, vhigh); 
                    
                    // Step 3: Collapse the remaining 4 floats down to 1 (i.e., fold 128-bit (4 floats) down to 64-bit (2 floats), then fold 64-bit (2 floats) into 32-bit (1 float) scalar).
                    __m128 shuf = _mm_movehdup_ps(vlow);
                    __m128 sums = _mm_add_ps(vlow, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    
                    // Step 4: Extract the final collapsed single scalar float safely
                    return _mm_cvtss_f32(sums);
                }

                static inline float reduce_min(register_type a) {
                    __m128 vlow = _mm256_castps256_ps128(a);
                    __m128 vhigh = _mm256_extractf128_ps(a, 1);
                    vlow = _mm_min_ps(vlow, vhigh);
                    __m128 shuf = _mm_movehdup_ps(vlow);
                    __m128 mins = _mm_min_ps(vlow, shuf);
                    shuf = _mm_movehl_ps(shuf, mins);
                    mins = _mm_min_ss(mins, shuf);
                    return _mm_cvtss_f32(mins);
                }
                static inline float reduce_max(register_type a) {
                    __m128 vlow = _mm256_castps256_ps128(a);
                    __m128 vhigh = _mm256_extractf128_ps(a, 1);
                    vlow = _mm_max_ps(vlow, vhigh);
                    __m128 shuf = _mm_movehdup_ps(vlow);
                    __m128 maxs = _mm_max_ps(vlow, shuf);
                    shuf = _mm_movehl_ps(shuf, maxs);
                    maxs = _mm_max_ss(maxs, shuf);
                    return _mm_cvtss_f32(maxs);
                }

                static inline bool mask_any(mask_type a) { 
                    // Extracts the most significant bit of each of the 8 float lanes and packs them into an 8-bit integer inside a general purpose scalar register in 1 clock cycle. 
                    return _mm256_movemask_ps(a) != 0; 
                }
                static inline bool mask_all(mask_type a) { 
                    return _mm256_movemask_ps(a) == 0xFF; 
                }

                static inline int mask_popcount(mask_type a) {
                    int mask_val = _mm256_movemask_ps(a); // Use _mm_movemask_ps for SSE4.1
                    #ifdef _MSC_VER
                        return __popcnt(mask_val);
                    #else
                        return __builtin_popcount(mask_val);
                    #endif
                }

                static inline int mask_find_first_set(mask_type a) {
                    int mask_val = _mm256_movemask_ps(a); // Use _mm_movemask_ps for SSE4.1
                    if (mask_val == 0) return -1;
                    #ifdef _MSC_VER
                        unsigned long index;
                        _BitScanForward(&index, mask_val);
                        return static_cast<int>(index);
                    #else
                        return __builtin_ctz(mask_val);
                    #endif
                }

                // Hardware Vector Swizzling (Symmetric across 128-bit lanes)
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    // _MM_SHUFFLE natively expects indices in reverse order (z, y, x, w)
                    // Template parameters are compile-time constants, making this perfectly safe.
                    return _mm256_permute_ps(a, _MM_SHUFFLE(i3, i2, i1, i0));
                }

                // Hardware Gather (Scale = 4 bytes per float)
                static inline register_type gather(const float* base_addr, __m256i indices) {
                    return _mm256_i32gather_ps(base_addr, indices, 4); 
                }

                // ==================================
                // MEMORY COMPRESSION (Float16)
                // ==================================
                // Store data in memory as 16-bit floats, cutting bandwidth by 50%, instantly decode them to 32-bit __mm256 registers.

                // Loads 128-bits of RAM (8 x 16-bit floats) and expands to a 256-bit register
                static inline register_type load_half(const uint16_t* mem) {
                    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mem)));
                }
                // Compresses a 256-bit register down to 128-bits of RAM (Rounding to nearest)
                static inline void store_half(uint16_t* mem, register_type v) {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
                }
            };

            // --- AVX2 DOUBLE TRAITS (4 Elements per Register) ---
            template <> struct simd_traits<double, simd_abi::avx2> {
                using register_type = __m256d;
                using mask_type     = __m256d; 
                static constexpr int size = 4;
                
                static inline register_type broadcast(double v) { return _mm256_set1_pd(v); }
                static inline register_type load(const double* mem) { return _mm256_loadu_pd(mem); }
                static inline void store(double* mem, register_type v) { _mm256_storeu_pd(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_pd(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_pd(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm256_mul_pd(a, b); }
                static inline register_type div(register_type a, register_type b) { return _mm256_div_pd(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_pd(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm256_max_pd(a, b); }
                static inline register_type sqrt(register_type a) { return _mm256_sqrt_pd(a); }

                static inline register_type rsqrt(register_type a) { return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(a)); }
                static inline register_type rcp(register_type a) { return _mm256_div_pd(_mm256_set1_pd(1.0), a); }

                static inline register_type abs(register_type a) { return _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF))); }
                static inline register_type floor(register_type a) { return _mm256_floor_pd(a); }
                static inline register_type ceil(register_type a) { return _mm256_ceil_pd(a); }

                static inline register_type fmadd(register_type a, register_type b, register_type c) { return _mm256_fmadd_pd(a, b, c); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }

                static inline mask_type mask_not(mask_type a) { 
                    __m256d all_ones = _mm256_cmp_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), _CMP_EQ_OQ);
                    return _mm256_xor_pd(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_pd(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_pd(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_pd(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_pd(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_pd(a, b); }
                static inline register_type negate(register_type a) { return _mm256_xor_pd(a, _mm256_set1_pd(-0.0)); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm256_blendv_pd(false_v, true_v, mask); }

                static inline double reduce_add(register_type a) {
                    __m128d lo = _mm256_extractf128_pd(a, 0);
                    __m128d hi = _mm256_extractf128_pd(a, 1);
                    __m128d sums = _mm_add_pd(lo, hi);
                    sums = _mm_add_pd(sums, _mm_unpackhi_pd(sums, sums));
                    return _mm_cvtsd_f64(sums);
                }

                static inline register_type gather(const double* base_addr, __m256i indices) {
                    // Elegant API mapping: Takes the lowest 128-bits (4 indices) from the uint32 batch and pulls 4 doubles!
                    return _mm256_i32gather_pd(base_addr, _mm256_castsi256_si128(indices), 8); 
                }

                static inline bool mask_any(mask_type a) { return _mm256_movemask_pd(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm256_movemask_pd(a) == 0xF; }

                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) { return _mm256_permute4x64_pd(a, _MM_SHUFFLE(i3, i2, i1, i0)); }
            };

            // --- AVX2 UINT8 TRAITS (32 Elements per Register) ---
            template <> struct simd_traits<uint8_t, simd_abi::avx2> {
                using register_type = __m256i;
                using mask_type     = __m256i; 
                static constexpr int size = 32;
                
                static inline register_type broadcast(uint8_t v) { return _mm256_set1_epi8(v); }
                static inline register_type load(const uint8_t* mem) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)); }
                static inline void store(uint8_t* mem, register_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_epi8(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_epi8(a, b); }
                
                // Saturating Hardware Arithmetic
                static inline register_type add_sat(register_type a, register_type b) { return _mm256_adds_epu8(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm256_subs_epu8(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_epu8(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm256_max_epu8(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_si256(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
                
                // Relational Unsigned Limits
                static inline mask_type cmp_gt(register_type a, register_type b) { 
                    __m256i sign_flip = _mm256_set1_epi8(static_cast<char>(0x80));
                    return _mm256_cmpgt_epi8(_mm256_xor_si256(a, sign_flip), _mm256_xor_si256(b, sign_flip)); 
                }
                static inline mask_type cmp_lt(register_type a, register_type b) { 
                    __m256i sign_flip = _mm256_set1_epi8(static_cast<char>(0x80));
                    return _mm256_cmpgt_epi8(_mm256_xor_si256(b, sign_flip), _mm256_xor_si256(a, sign_flip)); 
                }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmpeq_epi8(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
                    return _mm256_xor_si256(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_si256(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_si256(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return _mm256_blendv_epi8(false_v, true_v, mask);
                }

                static inline bool mask_any(mask_type a) { return _mm256_movemask_epi8(a) != 0; }
                // _mm256_movemask_epi8 extracts the top bit of all 32 bytes. If all are true, the result is 32 1s (0xFFFFFFFF)
                static inline bool mask_all(mask_type a) { return _mm256_movemask_epi8(a) == (int)0xFFFFFFFF; }
            };

            // --- AVX2 INT16 TRAITS (16 Elements per Register) ---
            template <> struct simd_traits<int16_t, simd_abi::avx2> {
                using register_type = __m256i;
                using mask_type     = __m256i; 
                static constexpr int size = 16;
                
                static inline register_type broadcast(int16_t v) { return _mm256_set1_epi16(v); }
                static inline register_type load(const int16_t* mem) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)); }
                static inline void store(int16_t* mem, register_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_epi16(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_epi16(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm256_mullo_epi16(a, b); }
                
                // SIGNED Saturating Hardware Arithmetic
                static inline register_type add_sat(register_type a, register_type b) { return _mm256_adds_epi16(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return _mm256_subs_epi16(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_epi16(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm256_max_epi16(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_si256(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm256_slli_epi16(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm256_srai_epi16(a, imm); }
                
                // Native Signed Limits
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmpgt_epi16(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmpgt_epi16(b, a); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmpeq_epi16(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
                    return _mm256_xor_si256(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_si256(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_si256(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm256_blendv_epi8(false_v, true_v, mask); }

                static inline bool mask_any(mask_type a) { return _mm256_movemask_epi8(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm256_movemask_epi8(a) == (int)0xFFFFFFFF; }

                template <typename Target>
                static inline __m256 cast_to(register_type a) { static_assert(sizeof(Target) == 0, "SIMD Mismatch: Cannot cast 16x16-bit to 8x32-bit directly."); return _mm256_setzero_ps(); }
            };

            // --- AVX2 UINT32 TRAITS ---
            template <> struct simd_traits<uint32_t, simd_abi::avx2> {
                using register_type = __m256i;
                using mask_type     = __m256i; 
                static constexpr int size = 8;

                template <typename Target>
                static inline __m256 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "uint32_t-to-float cast is implemented.");
                    return _mm256_cvtepi32_ps(a); 
                }
                
                // Memory
                static inline register_type broadcast(uint32_t v) { return _mm256_set1_epi32(v); }
                static inline register_type load(const uint32_t* mem) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)); }
                static inline void store(uint32_t* mem, register_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm256_mullo_epi32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_epu32(a, b); }
                static inline register_type max(register_type a, register_type b) { return _mm256_max_epu32(a, b); }

                // Bitwise Math (Required for your expandBits_SIMD function)
                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_si256(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm256_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm256_srli_epi32(a, imm); }
                
                // Relational (Unsigned Hardware Limits)
                static inline mask_type cmp_gt(register_type a, register_type b) { 
                    __m256i sign_flip = _mm256_set1_epi32(0x80000000);
                    return _mm256_cmpgt_epi32(_mm256_xor_si256(a, sign_flip), _mm256_xor_si256(b, sign_flip)); 
                }
                static inline mask_type cmp_lt(register_type a, register_type b) { 
                    __m256i sign_flip = _mm256_set1_epi32(0x80000000);
                    return _mm256_cmpgt_epi32(_mm256_xor_si256(b, sign_flip), _mm256_xor_si256(a, sign_flip)); 
                }
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    // return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); 

                    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
                    return _mm256_xor_si256(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_si256(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_si256(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    // Integer blending inherently looks at the highest bit of each byte
                    return _mm256_blendv_epi8(false_v, true_v, mask);
                }

                // Horizontal Integer Reduction
                static inline uint32_t reduce_add(register_type a) {
                    __m128i lo = _mm256_castsi256_si128(a);
                    __m128i hi = _mm256_extracti128_si256(a, 1);
                    lo = _mm_add_epi32(lo, hi);
                    
                    // Shuffle and fold 128-bit down to scalar
                    __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i sums = _mm_add_epi32(lo, shuf);
                    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                    sums = _mm_add_epi32(sums, shuf);
                    
                    return _mm_cvtsi128_si32(sums);
                }

                static inline bool mask_any(mask_type a) { 
                    return _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0; 
                }
                static inline bool mask_all(mask_type a) { 
                    return _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF; 
                }

                // Hardware Vector Swizzling (Symmetric across 128-bit lanes)
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    // AVX2 uses a dedicated integer shuffle instruction
                    return _mm256_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0));
                }

                // Hardware Gather (Scale = 4 bytes per uint32)
                static inline register_type gather(const uint32_t* base_addr, __m256i indices) {
                    // The intrinsic explicitly requires a const int* pointer
                    return _mm256_i32gather_epi32(reinterpret_cast<const int*>(base_addr), indices, 4); 
                }
            };

            template <> struct simd_traits<int32_t, simd_abi::avx2> {
                using register_type = __m256i;
                using mask_type     = __m256i; 
                static constexpr int size = 8;

                template <typename Target>
                static inline __m256 cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "int32_t-to-float cast.");
                    return _mm256_cvtepi32_ps(a); 
                }
                
                static inline register_type broadcast(int32_t v) { return _mm256_set1_epi32(v); }
                static inline register_type load(const int32_t* mem) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)); }
                static inline void store(int32_t* mem, register_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v); }
                
                static inline register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return _mm256_sub_epi32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return _mm256_mullo_epi32(a, b); }

                static inline register_type min(register_type a, register_type b) { return _mm256_min_epi32(a, b); } // Native signed
                static inline register_type max(register_type a, register_type b) { return _mm256_max_epi32(a, b); }

                static inline register_type bit_xor(register_type a, register_type b) { return _mm256_xor_si256(a, b); }
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm256_slli_epi32(a, imm); }
                static inline register_type shift_r(register_type a, int imm) { return _mm256_srai_epi32(a, imm); } // Arithmetic
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmpgt_epi32(a, b); } // Native signed
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmpgt_epi32(b, a); } 
                static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }

                static inline mask_type mask_not(mask_type a) { 
                    __m256i all_ones = _mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256());
                    return _mm256_xor_si256(a, all_ones);
                }
                static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_si256(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_si256(a, b); }
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return _mm256_blendv_epi8(false_v, true_v, mask); }

                static inline int32_t reduce_add(register_type a) {
                    __m128i lo = _mm256_castsi256_si128(a);
                    __m128i hi = _mm256_extracti128_si256(a, 1);
                    lo = _mm_add_epi32(lo, hi);
                    __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128i sums = _mm_add_epi32(lo, shuf);
                    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                    sums = _mm_add_epi32(sums, shuf);
                    return _mm_cvtsi128_si32(sums);
                }
                static inline bool mask_any(mask_type a) { return _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0; }
                static inline bool mask_all(mask_type a) { return _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF; }
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) { return _mm256_shuffle_epi32(a, _MM_SHUFFLE(i3, i2, i1, i0)); }
                static inline register_type gather(const int32_t* base_addr, __m256i indices) { return _mm256_i32gather_epi32(reinterpret_cast<const int*>(base_addr), indices, 4); }
            };
        #endif // ENGINE_ARCH_AVX2

        #if ENGINE_ARCH_NEON
            // ========================================================
            // --- NEON BACKEND (Nintendo Switch 2, Apple Silicon) ---
            // ========================================================
            template <> struct simd_traits<float, simd_abi::neon> {
                using register_type = float32x4_t;
                using mask_type     = uint32x4_t; // NEON strictly separates math and mask register types
                static constexpr int size = 4;
                
                static inline register_type broadcast(float v) { return vdupq_n_f32(v); }
                static inline register_type load(const float* mem) { return vld1q_f32(mem); }
                static inline void store(float* mem, register_type v) { vst1q_f32(mem, v); }
                
                static inline register_type add(register_type a, register_type b) { return vaddq_f32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return vmulq_f32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_f32(a, b); }
                static inline register_type div(register_type a, register_type b) { return vdivq_f32(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_f32(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_f32(a, b); }

                // 1 / sqrt(x)
                static inline register_type rsqrt(register_type a) { 
                    // 1. Get the rough 8-bit estimate
                    float32x4_t approx = vrsqrteq_f32(a);
                    // 2. Execute the dedicated hardware Newton-Raphson step
                    float32x4_t step = vrsqrtsq_f32(a, vmulq_f32(approx, approx));
                    // 3. Multiply the estimate by the step to get 23-bit precision
                    return vmulq_f32(approx, step);
                }

                // 1 / x
                static inline register_type rcp(register_type a) { 
                    // 1. Get the rough 8-bit estimate
                    float32x4_t approx = vrecpeq_f32(a);
                    // 2. Execute the dedicated hardware Newton-Raphson step
                    float32x4_t step = vrecpsq_f32(a, approx);
                    // 3. Multiply the estimate by the step to get 23-bit precision
                    return vmulq_f32(approx, step);
                }

                static inline register_type abs(register_type a) { return vabsq_f32(a); }
                static inline register_type floor(register_type a) { return vrndmq_f32(a); } // Round towards Minus infinity
                static inline register_type ceil(register_type a) { return vrndpq_f32(a); }  // Round towards Plus infinity
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_f32(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_f32(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_f32(a, b); }

                static inline mask_type mask_not(mask_type a) { return vmvnq_u32(a); } // NEON Bitwise NOT
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u32(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u32(a, b); }

                // Floating-Point Bitwise Logic (Zero-cost reinterpret casting)
                static inline register_type bit_xor(register_type a, register_type b) { 
                    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); 
                }
                static inline register_type bit_and(register_type a, register_type b) { 
                    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); 
                }
                static inline register_type bit_or(register_type a, register_type b) { 
                    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); 
                }

                // Fast Unary Negation
                static inline register_type negate(register_type a) {
                    // ARM NEON natively has a dedicated float negate instruction!
                    return vnegq_f32(a); 
                }
                
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return vbslq_f32(mask, true_v, false_v);
                }

                // Hardware Math
                static inline register_type fmadd(register_type a, register_type b, register_type c) {
                    return vfmaq_f32(c, a, b); // accumulates (a * b) into c
                }
                static inline register_type sqrt(register_type a) {
                    return vsqrtq_f32(a);
                }

                // SIMD Casting: NEON explicitly supports float-to-unsigned-int conversion natively.
                template <typename Target>
                static inline uint32x4_t cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t casting is currently implemented.");
                    return vcvtq_u32_f32(a);
                }

                // Horizontal Reduction: Because the Switch 2 and Apple Silicon use AArch64, we don't have to do the folding dance like on AVX2. 
                // We get to use a gorgeous, single-cycle hardware reduction instruction!
                static inline float reduce_add(register_type a) {
                    return vaddvq_f32(a); 
                }

                static inline float reduce_min(register_type a) { return vminvq_f32(a); }
                static inline float reduce_max(register_type a) { return vmaxvq_f32(a); }

                static inline bool mask_any(mask_type a) { 
                    // Modern ARM64 vector lane addition instruction: Checks if any lanes in a 128-bit vector are active.
                    return vaddvq_u32(a) != 0;
                }
                static inline bool mask_all(mask_type a) { 
                    // If the minimum value across the vector is > 0, all lanes are true
                    return vminvq_u32(a) > 0; 
                }

                static inline int mask_popcount(mask_type a) {
                    // Safely extract lanes to scalar. Each true lane is guaranteed to be 0xFFFFFFFF.
                    int count = 0;
                    if (vgetq_lane_u32(a, 0)) count++;
                    if (vgetq_lane_u32(a, 1)) count++;
                    if (vgetq_lane_u32(a, 2)) count++;
                    if (vgetq_lane_u32(a, 3)) count++;
                    return count;
                }

                static inline int mask_find_first_set(mask_type a) {
                    // Extract lanes to scalar and find the first non-zero lane
                    if (vgetq_lane_u32(a, 0)) return 0;
                    if (vgetq_lane_u32(a, 1)) return 1;
                    if (vgetq_lane_u32(a, 2)) return 2;
                    if (vgetq_lane_u32(a, 3)) return 3;
                    return -1;
                }

                // Hardware Vector Swizzling
                template <int i0, int i1, int i2, int i3>
                static FORCE_INLINE register_type shuffle(register_type a) {
                    // Nintendo Switch 2 and Apple Silicon are compiled almost exclusively via Clang/LLVM.
                    #if defined(__clang__)
                        return __builtin_shufflevector(a, a, i0, i1, i2, i3);
                    #else
                        // Fallback for GCC/MSVC on ARM
                        // MSVC will aggressively fold this into a single instruction.
                        register_type res = vdupq_n_f32(0.0f); 
                        res = vsetq_lane_f32(vgetq_lane_f32(a, i0), res, 0);
                        res = vsetq_lane_f32(vgetq_lane_f32(a, i1), res, 1);
                        res = vsetq_lane_f32(vgetq_lane_f32(a, i2), res, 2);
                        res = vsetq_lane_f32(vgetq_lane_f32(a, i3), res, 3);
                        return res;
                    #endif
                }

                // ========================================
                // ARM NEON (AArch64)
                // ========================================
                /*
                    - ARM NEON does not have native 128-bit vector gather instruction.
                    - ARM SVE (Scalable Vector Extension) adds 128-bit vector gather instruction.
                    - We can emulate it.
                */

                // Emulated Hardware Gather
                static inline register_type gather(const float* base_addr, uint32x4_t indices) {
                    register_type res = vdupq_n_f32(0.0f);
                    // Pointer arithmetic implicitly scales by 4 bytes (sizeof float)
                    res = vsetq_lane_f32(base_addr[vgetq_lane_u32(indices, 0)], res, 0);
                    res = vsetq_lane_f32(base_addr[vgetq_lane_u32(indices, 1)], res, 1);
                    res = vsetq_lane_f32(base_addr[vgetq_lane_u32(indices, 2)], res, 2);
                    res = vsetq_lane_f32(base_addr[vgetq_lane_u32(indices, 3)], res, 3);
                    return res;
                }

                static inline register_type load_half(const uint16_t* mem) {
                    return vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t*>(mem)));
                }
                static inline void store_half(uint16_t* mem, register_type v) {
                    vst1_f16(reinterpret_cast<float16_t*>(mem), vcvt_f16_f32(v));
                }
            };

            // --- NEON DOUBLE TRAITS (2 Elements per Register) ---
            template <> struct simd_traits<double, simd_abi::neon> {
                using register_type = float64x2_t;
                using mask_type     = uint64x2_t;
                static constexpr int size = 2;
                
                static inline register_type broadcast(double v) { return vdupq_n_f64(v); }
                static inline register_type load(const double* mem) { return vld1q_f64(mem); }
                static inline void store(double* mem, register_type v) { vst1q_f64(mem, v); }

                static inline register_type add(register_type a, register_type b) { return vaddq_f64(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_f64(a, b); }
                static inline register_type mul(register_type a, register_type b) { return vmulq_f64(a, b); }
                static inline register_type div(register_type a, register_type b) { return vdivq_f64(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_f64(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_f64(a, b); }
                static inline register_type sqrt(register_type a) { return vsqrtq_f64(a); }

                static inline register_type rsqrt(register_type a) { return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(a)); }
                static inline register_type rcp(register_type a) { return vdivq_f64(vdupq_n_f64(1.0), a); }

                static inline register_type abs(register_type a) { return vabsq_f64(a); }
                static inline register_type floor(register_type a) { return vrndmq_f64(a); }
                static inline register_type ceil(register_type a) { return vrndpq_f64(a); }
                
                static inline register_type fmadd(register_type a, register_type b, register_type c) { return vfmaq_f64(c, a, b); }

                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_f64(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_f64(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_f64(a, b); }

                static inline mask_type mask_not(mask_type a) { return veorq_u64(a, vdupq_n_u64(-1)); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u64(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u64(a, b); }

                static inline register_type bit_or(register_type a, register_type b) { return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))); }
                static inline register_type bit_and(register_type a, register_type b) { return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))); }
                static inline register_type bit_xor(register_type a, register_type b) { return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b))); }
                static inline register_type negate(register_type a) { return vnegq_f64(a); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return vbslq_f64(mask, true_v, false_v); }

                static inline double reduce_add(register_type a) { return vaddvq_f64(a); }

                static inline register_type gather(const double* base_addr, uint32x4_t indices) {
                    register_type res = vdupq_n_f64(0.0);
                    res = vsetq_lane_f64(base_addr[vgetq_lane_u32(indices, 0)], res, 0);
                    res = vsetq_lane_f64(base_addr[vgetq_lane_u32(indices, 1)], res, 1);
                    return res;
                }

                static inline bool mask_any(mask_type a) {
                    return vaddvq_u64(a) != 0;
                }
                static inline bool mask_all(mask_type a) { return vminvq_u32(vreinterpretq_u32_u64(a)) > 0; }

                template <int i0, int i1>
                static inline register_type shuffle(register_type a) {
                    register_type res = vdupq_n_f64(0.0);
                    // ARM NEON lacks a fast cross-lane double shuffle, fallback to inserts
                    res = vsetq_lane_f64(vgetq_lane_f64(a, i0), res, 0);
                    res = vsetq_lane_f64(vgetq_lane_f64(a, i1), res, 1);
                    return res;
                }
            };

            // --- NEON UINT8 TRAITS (16 Elements per Register) ---
            template <> struct simd_traits<uint8_t, simd_abi::neon> {
                using register_type = uint8x16_t;
                using mask_type     = uint8x16_t;
                static constexpr int size = 16;
                
                static inline register_type broadcast(uint8_t v) { return vdupq_n_u8(v); }
                static inline register_type load(const uint8_t* mem) { return vld1q_u8(mem); }
                static inline void store(uint8_t* mem, register_type v) { vst1q_u8(mem, v); }

                static inline register_type add(register_type a, register_type b) { return vaddq_u8(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_u8(a, b); }
                
                // Saturating Hardware Arithmetic
                static inline register_type add_sat(register_type a, register_type b) { return vqaddq_u8(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return vqsubq_u8(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_u8(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_u8(a, b); }
                
                static inline register_type bit_or(register_type a, register_type b) { return vorrq_u8(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return vandq_u8(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return veorq_u8(a, b); }
                
                // Relational
                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_u8(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_u8(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_u8(a, b); }

                static inline mask_type mask_not(mask_type a) { return vmvnq_u8(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u8(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u8(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return vbslq_u8(mask, true_v, false_v);
                }

                static inline bool mask_any(mask_type a) { return vmaxvq_u8(a) > 0; }
                static inline bool mask_all(mask_type a) { return vminvq_u8(a) > 0; }
            };

            // --- NEON UINT32 TRAITS ---
            template <> struct simd_traits<uint32_t, simd_abi::neon> {
                using register_type = uint32x4_t;
                using mask_type     = uint32x4_t;
                static constexpr int size = 4;
                
                // Memory
                static inline register_type broadcast(uint32_t v) { return vdupq_n_u32(v); }
                static inline register_type load(const uint32_t* mem) { return vld1q_u32(mem); }
                static inline void store(uint32_t* mem, register_type v) { vst1q_u32(mem, v); }

                static inline register_type add(register_type a, register_type b) { return vaddq_u32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_u32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return vmulq_u32(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_f32(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_f32(a, b); }
                
                // Bitwise Math
                static inline register_type bit_or(register_type a, register_type b) { return vorrq_u32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return vandq_u32(a, b); }
                static inline register_type shift_l(register_type a, int imm) { 
                    // Broadcast the scalar 'imm' into a signed 32x4 vector, then shift
                    return vshlq_u32(a, vdupq_n_s32(imm)); 
                }
                static inline register_type shift_r(register_type a, int imm) { 
                    // Broadcast the negative shift amount as a signed 32-bit integer vector
                    return vshlq_u32(a, vdupq_n_s32(-imm)); 
                }
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_u32(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_u32(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_u32(a, b); }

                static inline mask_type mask_not(mask_type a) { return vmvnq_u32(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u32(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u32(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return vbslq_u32(mask, true_v, false_v);
                }

                // Horizontal Integer Reduction
                static inline uint32_t reduce_add(register_type a) {
                    return vaddvq_u32(a); 
                }

                static inline bool mask_any(mask_type a) { 
                    // If the maximum value across the vector is > 0, at least one lane is true
                    return vmaxvq_u32(a) > 0; 
                }
                static inline bool mask_all(mask_type a) { 
                    // If the minimum value across the vector is > 0, all lanes are true
                    return vminvq_u32(a) > 0; 
                }

                // Hardware Vector Swizzling
                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    register_type res = vdupq_n_u32(0); 
                    res = vsetq_lane_u32(vgetq_lane_u32(a, i0), res, 0);
                    res = vsetq_lane_u32(vgetq_lane_u32(a, i1), res, 1);
                    res = vsetq_lane_u32(vgetq_lane_u32(a, i2), res, 2);
                    res = vsetq_lane_u32(vgetq_lane_u32(a, i3), res, 3);
                    return res;
                }

                // Emulated Hardware Gather
                static inline register_type gather(const uint32_t* base_addr, uint32x4_t indices) {
                    register_type res = vdupq_n_u32(0);
                    res = vsetq_lane_u32(base_addr[vgetq_lane_u32(indices, 0)], res, 0);
                    res = vsetq_lane_u32(base_addr[vgetq_lane_u32(indices, 1)], res, 1);
                    res = vsetq_lane_u32(base_addr[vgetq_lane_u32(indices, 2)], res, 2);
                    res = vsetq_lane_u32(base_addr[vgetq_lane_u32(indices, 3)], res, 3);
                    return res;
                }

                template <typename Target>
                static inline float32x4_t cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "uint32_t-to-float cast is implemented.");
                    return vcvtq_f32_u32(a);
                }
            };

            template <> struct simd_traits<int32_t, simd_abi::neon> {
                using register_type = int32x4_t;
                using mask_type     = uint32x4_t;
                static constexpr int size = 4;
                
                static inline register_type broadcast(int32_t v) { return vdupq_n_s32(v); }
                static inline register_type load(const int32_t* mem) { return vld1q_s32(mem); }
                static inline void store(int32_t* mem, register_type v) { vst1q_s32(mem, v); }

                static inline register_type add(register_type a, register_type b) { return vaddq_s32(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_s32(a, b); }
                static inline register_type mul(register_type a, register_type b) { return vmulq_s32(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_s32(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_s32(a, b); }
                
                static inline register_type bit_or(register_type a, register_type b) { return vorrq_s32(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return vandq_s32(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return veorq_s32(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return vshlq_s32(a, vdupq_n_s32(imm)); }
                static inline register_type shift_r(register_type a, int imm) { return vshlq_s32(a, vdupq_n_s32(-imm)); } // Preserves sign
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_s32(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_s32(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_s32(a, b); }

                static inline mask_type mask_not(mask_type a) { return vmvnq_u32(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u32(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u32(a, b); }
                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) { return vbslq_s32(mask, true_v, false_v); }

                static inline int32_t reduce_add(register_type a) { return vaddvq_s32(a); }
                static inline bool mask_any(mask_type a) { return vmaxvq_u32(a) > 0; }
                static inline bool mask_all(mask_type a) { return vminvq_u32(a) > 0; }

                static inline int mask_popcount(mask_type a) {
                    // Safely extract lanes to scalar. Each true lane is guaranteed to be 0xFFFFFFFF.
                    int count = 0;
                    if (vgetq_lane_u32(a, 0)) count++;
                    if (vgetq_lane_u32(a, 1)) count++;
                    if (vgetq_lane_u32(a, 2)) count++;
                    if (vgetq_lane_u32(a, 3)) count++;
                    return count;
                }

                static inline int mask_find_first_set(mask_type a) {
                    // Extract lanes to scalar and find the first non-zero lane
                    if (vgetq_lane_u32(a, 0)) return 0;
                    if (vgetq_lane_u32(a, 1)) return 1;
                    if (vgetq_lane_u32(a, 2)) return 2;
                    if (vgetq_lane_u32(a, 3)) return 3;
                    return -1;
                }

                template <int i0, int i1, int i2, int i3>
                static inline register_type shuffle(register_type a) {
                    register_type res = vdupq_n_s32(0); 
                    res = vsetq_lane_s32(vgetq_lane_s32(a, i0), res, 0);
                    res = vsetq_lane_s32(vgetq_lane_s32(a, i1), res, 1);
                    res = vsetq_lane_s32(vgetq_lane_s32(a, i2), res, 2);
                    res = vsetq_lane_s32(vgetq_lane_s32(a, i3), res, 3);
                    return res;
                }
                static inline register_type gather(const int32_t* base_addr, uint32x4_t indices) {
                    register_type res = vdupq_n_s32(0);
                    res = vsetq_lane_s32(base_addr[vgetq_lane_u32(indices, 0)], res, 0);
                    res = vsetq_lane_s32(base_addr[vgetq_lane_u32(indices, 1)], res, 1);
                    res = vsetq_lane_s32(base_addr[vgetq_lane_u32(indices, 2)], res, 2);
                    res = vsetq_lane_s32(base_addr[vgetq_lane_u32(indices, 3)], res, 3);
                    return res;
                }
                template <typename Target>
                static inline float32x4_t cast_to(register_type a) {
                    static_assert(std::is_same_v<Target, float>, "int32_t-to-float cast is implemented.");
                    return vcvtq_f32_s32(a);
                }
            };

            // --- NEON INT16 TRAITS (8 Elements per Register) ---
            template <> struct simd_traits<int16_t, simd_abi::neon> {
                using register_type = int16x8_t;
                using mask_type     = uint16x8_t;
                static constexpr int size = 8;
                
                static inline register_type broadcast(int16_t v) { return vdupq_n_s16(v); }
                static inline register_type load(const int16_t* mem) { return vld1q_s16(mem); }
                static inline void store(int16_t* mem, register_type v) { vst1q_s16(mem, v); }

                static inline register_type add(register_type a, register_type b) { return vaddq_s16(a, b); }
                static inline register_type sub(register_type a, register_type b) { return vsubq_s16(a, b); }
                static inline register_type mul(register_type a, register_type b) { return vmulq_s16(a, b); }
                
                // Saturating Hardware Arithmetic
                static inline register_type add_sat(register_type a, register_type b) { return vqaddq_s16(a, b); }
                static inline register_type sub_sat(register_type a, register_type b) { return vqsubq_s16(a, b); }

                static inline register_type min(register_type a, register_type b) { return vminq_s16(a, b); }
                static inline register_type max(register_type a, register_type b) { return vmaxq_s16(a, b); }
                
                static inline register_type bit_or(register_type a, register_type b) { return vorrq_s16(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return vandq_s16(a, b); }
                static inline register_type bit_xor(register_type a, register_type b) { return veorq_s16(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return vshlq_s16(a, vdupq_n_s32(imm)); }
                static inline register_type shift_r(register_type a, int imm) { return vshlq_s16(a, vdupq_n_s32(-imm)); }
                
                // Relational
                static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_s16(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_s16(a, b); }
                static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_s16(a, b); }

                static inline mask_type mask_not(mask_type a) { return vmvnq_u16(a); }
                static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u16(a, b); }
                static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u16(a, b); }

                static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                    return vbslq_s16(mask, true_v, false_v);
                }

                static inline bool mask_any(mask_type a) { return vmaxvq_u32(vreinterpretq_u32_u16(a)) > 0; }
                static inline bool mask_all(mask_type a) { return vminvq_u32(vreinterpretq_u32_u16(a)) > 0; }

                template <typename Target>
                static inline float32x4_t cast_to(register_type a) { static_assert(sizeof(Target) == 0, "SIMD Mismatch: Cannot cast 8x16-bit to 4x32-bit directly."); return vdupq_n_f32(0.0f); }
            };
        #endif // ENGINE_ARCH_NEON

        // ========================================================
        // --- SCALAR BACKEND (The Absolute Fallback) ---
        // ========================================================
        /*
            - Used when vector hardware is unavailable or SIMD flag is disabled (ENGINE_ARCH_SCALAR).
            - Ensures the exact same code will silently transition into standard scalar operations without compilation errors.
        */
        template <typename T> struct simd_traits<T, simd_abi::scalar> {
            using register_type = T;
            using mask_type     = bool;
            static constexpr int size = 1;

            static inline register_type broadcast(T v) { return v; }
            static inline register_type load(const T* mem) { return *mem; }
            static inline void store(T* mem, register_type v) { *mem = v; }

            static inline register_type add(register_type a, register_type b) { return a + b; }
            static inline register_type sub(register_type a, register_type b) { return a - b; }
            static inline register_type mul(register_type a, register_type b) { return a * b; }
            static inline register_type div(register_type a, register_type b) { return a / b; }

            static inline register_type min(register_type a, register_type b) { return std::min(a, b); }
            static inline register_type max(register_type a, register_type b) { return std::max(a, b); }

            // Math (Mapped entirely to custom hardware)
            static inline register_type rsqrt(register_type a) { return Engine::Math::ScalarFunctions::rsqrt(a); }
            static inline register_type rcp(register_type a)   { return Engine::Math::ScalarFunctions::rcp(a); }
            static inline register_type abs(register_type a)   { return Engine::Math::ScalarFunctions::abs(a); }
            static inline register_type floor(register_type a) { return Engine::Math::ScalarFunctions::floor(a); }
            static inline register_type ceil(register_type a)  { return Engine::Math::ScalarFunctions::ceil(a); }
            static inline register_type sqrt(register_type a)  { return Engine::Math::ScalarFunctions::sqrt(a); }
            static inline register_type fmadd(register_type a, register_type b, register_type c) { return (a * b) + c; }
            
            // Relational
            static inline mask_type cmp_gt(register_type a, register_type b) { return a > b; }
            static inline mask_type cmp_lt(register_type a, register_type b) { return a < b; }
            static inline mask_type cmp_eq(register_type a, register_type b) { return a == b; }

            // Mask Logic
            static inline mask_type mask_not(mask_type a) { return !a; }
            static inline mask_type mask_and(mask_type a, mask_type b) { return a && b; }
            static inline mask_type mask_or(mask_type a, mask_type b) { return a || b; }

            static inline int mask_popcount(mask_type a) { return a ? 1 : 0; }
            static inline int mask_find_first_set(mask_type a) { return a ? 0 : -1; }

            // Bitwise (Integers only via SFINAE)
            static inline register_type bit_xor(register_type a, register_type b) { return a ^ b; }
            static inline register_type bit_or(register_type a, register_type b)  { return a | b; }
            static inline register_type bit_and(register_type a, register_type b) { return a & b; }
            
            static inline register_type negate(register_type a) { return -a; }

            // --- The Missing API Endpoints ---
            
            // Blend: Standard ternary operator
            static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                return mask ? true_v : false_v;
            }

            // Reduction: Scalar is already reduced
            static inline T reduce_add(register_type a) { return a; }
            static inline T reduce_min(register_type a) { return a; }
            static inline T reduce_max(register_type a) { return a; }

            // Gather: Direct array access using scalar index
            static inline register_type gather(const T* base_addr, uint32_t index) {
                return base_addr[index];
            }

            // Mask Evaluation
            static inline bool mask_any(mask_type a) { return a; }
            static inline bool mask_all(mask_type a) { return a; }

            // Shuffle: Scalar has only one lane, return itself
            template <int i0, int i1, int i2, int i3>
            static inline register_type shuffle(register_type a) { return a; }

            // Casting
            template <typename Target>
            static inline Target cast_to(register_type a) { return static_cast<Target>(a); }
        };
    }

    // ==========================================
    // C++26 MASK FRONTEND
    // ==========================================
    template <typename T, typename Abi = simd_abi::native<T>> class simd;

    // C++26 strictly requires a separate mask type to safely handle SIMD conditionals
    template <typename T, typename Abi = simd_abi::native<T>>
    class simd_mask {
    private:
        using Traits = detail::simd_traits<T, Abi>;
        typename Traits::mask_type m_mask;

        // Allow masks of different types to access each other's private data for Mixed-Type casting
        template <typename U, typename UAbi> friend class simd_mask;

        // Only intrinsics can construct a mask directly
        friend class simd<T, Abi>;
        explicit inline simd_mask(typename Traits::mask_type mask) : m_mask(mask) {}

    public:
        simd_mask() = default;

        // Allows safe construction from raw hardware masks
        static inline simd_mask from_native(typename Traits::mask_type mask) { return simd_mask(mask); }

        // Public hardware accessor
        FORCE_INLINE typename Traits::mask_type native() const { return m_mask; }
        FORCE_INLINE typename Traits::mask_type native_handle() const { return m_mask; }

        // --- MIXED-TYPE MASK CASTING ---
        template <typename U>
        inline simd_mask<U, Abi> cast_to() const {
            if constexpr (std::is_same_v<T, U>) {
                return *this;
            } 
            // INT32 / UINT32 to FLOAT
            else if constexpr ((std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) && std::is_same_v<U, float>) {
                if constexpr (std::is_same_v<Abi, simd_abi::avx2>) {
                    return simd_mask<U, Abi>(_mm256_castsi256_ps(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::sse41>) {
                    return simd_mask<U, Abi>(_mm_castsi128_ps(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::neon> || std::is_same_v<Abi, simd_abi::avx512>) {
                    return simd_mask<U, Abi>(m_mask); // AVX-512 and NEON bypass casting
                }
            } 
            // FLOAT to INT32 / UINT32
            else if constexpr (std::is_same_v<T, float> && (std::is_same_v<U, uint32_t> || std::is_same_v<U, int32_t>)) {
                if constexpr (std::is_same_v<Abi, simd_abi::avx2>) {
                    return simd_mask<U, Abi>(_mm256_castps_si256(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::sse41>) {
                    return simd_mask<U, Abi>(_mm_castps_si128(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::neon> || std::is_same_v<Abi, simd_abi::avx512>) {
                    return simd_mask<U, Abi>(m_mask); // AVX-512 and NEON bypass casting
                }
            }
            // INT32 <-> UINT32 (Identical bit representation, safely pass through)
            else if constexpr ((std::is_same_v<T, uint32_t> && std::is_same_v<U, int32_t>) || 
                               (std::is_same_v<T, int32_t> && std::is_same_v<U, uint32_t>)) {
                return simd_mask<U, Abi>(m_mask); 
            }
            else {
                static_assert(sizeof(U) == 0, "Unsupported mask cast.");
            }
        }

        // --- LOGICAL OPERATORS ---
        friend inline simd_mask operator!(const simd_mask& a) {
            return simd_mask(Traits::mask_not(a.m_mask));
        }

        template <typename U>
        inline simd_mask operator&&(const simd_mask<U, Abi>& b) const {
            return simd_mask(Traits::mask_and(m_mask, b.template cast_to<T>().m_mask));
        }

        template <typename U>
        inline simd_mask operator||(const simd_mask<U, Abi>& b) const {
            return simd_mask(Traits::mask_or(m_mask, b.template cast_to<T>().m_mask));
        }

        friend inline bool any_of(const simd_mask& m) { return Traits::mask_any(m.m_mask); }
        friend inline bool all_of(const simd_mask& m) { return Traits::mask_all(m.m_mask); }
        friend inline bool none_of(const simd_mask& m) { return !Traits::mask_any(m.m_mask); }

        // =======================================================
        // MASK INTROSPECTION (ATA COMPACTION / STREAM COMPACTION)
        // =======================================================
        /*
            - When a particle dies, its mask evaluates to true for "dead".
            - popcount(...) lets us know how many particles died in that batch so you can decrement the activeCount.
            - find_first_set(...) lets us know exactly hich array index needs to be overwritten by the particle at the end of the array.
        */

        friend inline int popcount(const simd_mask& m) { return Traits::mask_popcount(m.m_mask); }
        friend inline int find_first_set(const simd_mask& m) { return Traits::mask_find_first_set(m.m_mask); }

    };

    // ==========================================
    // C++26 SIMD FRONTEND
    // ==========================================
    template <typename T, typename Abi>
    class simd {
    private:
        using Traits = detail::simd_traits<T, Abi>;
        typename Traits::register_type m_data;

        // An internal tag type to completely isolate raw registers from numbers (i.e., hides the raw hardare register constructor from the public API).
        struct native_tag {};
        explicit inline simd(typename Traits::register_type data, native_tag) : m_data(data) {}
    public:
        // C++26 Type Definitions
        using value_type = T;
        using abi_type   = Abi;
        using mask_type  = simd_mask<T, Abi>;

        static constexpr int size() { return Traits::size; }

        // ===================================================================
        // --- HARDWARE INTEROP FACTORY & ARGUMENT DEPENDENT LOOKUP (ADL) ---
        // ===================================================================
        /*
            - This from_native is a wrapper that wraps the raw silicon register (__m256, etc.) into an object purely for the function calls.
            - Allows the compiler to find the hidden friend functions (e.g., reduce).
        */
        static inline simd from_native(typename Traits::register_type raw) { return simd(raw, native_tag{}); }
        
        // Public Accessors to prevent proxy layer access violations
        FORCE_INLINE typename Traits::register_type native() const { return m_data; }
        FORCE_INLINE typename Traits::register_type native_handle() const { return m_data; }

        // Explicit Typecast Gateway (Lets the class masquerade as an intrinsic register for mathematical expressions) without letting the compiler bypass the class wrappers properties.
        [[nodiscard]] explicit FORCE_INLINE operator typename Traits::register_type() const { return m_data; }

        // --- PUBLIC CONSTRUCTORS ---

        // 1. Default Constructor (initialized, just like raw floats)
        simd() = default; 

        // 2. Universal Broadcast Constructor (e.g., WideFloat(5.0f)) is strictly implicit. Converts any number into a full vector path cleanly across all architectures.
        template <typename U> 
        requires std::is_arithmetic_v<U>
        inline simd(U value) : m_data(Traits::broadcast(static_cast<T>(value))) {}
        
        // 3. Memory Load Constructor for memory operations (e.g., WideFloat(&array[0])) is strictly explicit.
        explicit inline simd(const T* mem) : m_data(Traits::load(mem)) {}

        // 4. Non-Contiguous Memory Load Constructor (Gather). Takes a base pointer and a SIMD batch of array indices.
        explicit inline simd(const T* base_addr, const simd<uint32_t, Abi>& indices) 
            : m_data(Traits::gather(base_addr, indices.native_handle())) {}

        // --- MEMORY STORE ---
        void copy_to(T* mem) const { Traits::store(mem, m_data); }

        friend inline simd min(const simd& a, const simd& b) { return simd::from_native(Traits::min(a.m_data, b.m_data)); }
        friend inline simd max(const simd& a, const simd& b) { return simd::from_native(Traits::max(a.m_data, b.m_data)); }
        friend inline simd clamp(const simd& v, const simd& lo, const simd& hi) { return min(max(v, lo), hi); }

        // requires std::is_floating_point_v<T> means this is a float only intrinsic instruction (i.e., floating-point constraint).
        friend inline simd rsqrt(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::rsqrt(a.m_data)); }
        friend inline simd rcp(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::rcp(a.m_data)); }
        friend inline simd abs(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::abs(a.m_data)); }
        friend inline simd floor(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::floor(a.m_data)); }
        friend inline simd ceil(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::ceil(a.m_data)); }
        friend inline simd sqrt(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::sqrt(a.m_data)); }
        friend inline simd fmadd(const simd& a) requires std::is_floating_point_v<T> { return simd::from_native(Traits::fmadd(a.m_data)); }


        // Float16 Memory Gateway
        static simd load_f16(const uint16_t* mem) requires std::is_floating_point_v<T> {
            return simd::from_native(Traits::load_half(mem));
        }
        void store_f16(uint16_t* mem) const requires std::is_floating_point_v<T> {
            Traits::store_half(mem, m_data);
        }

        // --- SATURATING ARITHMETIC (Clamps instead of wrapping) ---
        // Unlocked for 8-bit integers (Colors) and 16-bit integers (Audio)
        friend inline simd add_sat(const simd& a, const simd& b) requires (sizeof(T) == 1 || sizeof(T) == 2) {
            return simd::from_native(Traits::add_sat(a.m_data, b.m_data));
        }

        friend inline simd sub_sat(const simd& a, const simd& b) requires (sizeof(T) == 1 || sizeof(T) == 2) {
            return simd::from_native(Traits::sub_sat(a.m_data, b.m_data));
        }

        // =======================================================================
        // VECTORIZED TRIGONOMETRY (MINIMAX POLYNOMIAL APPROX & HORNER'S METHOD)
        // =======================================================================
        /*
            - Estimates a curve using a 9th degree polynomial.
            - Used to animate thousands of skeletal meshes, calculate camera FOV projections, or generate procedural wind for foliage.
            - Used to simulate ocean waves, or project 3D coordinates onto a 2D screen.
            - Sine and Cosine are needed.
        */
        
        // Note: This implementation is suppose to be in the backend, not the front end.
        friend inline simd sin(const simd& x) requires std::is_floating_point_v<T> {
            // 1. Core Constants
            const simd INV_TWO_PI(0.159154943f);
            const simd TWO_PI(6.283185307f);
            
            // Minimax Polynomial Coefficients for [-PI, PI]
            const simd C1(-0.1666666716f); // -1/3!
            const simd C2(0.0083333310f);  //  1/5!
            const simd C3(-0.0001984087f); // -1/7!
            const simd C4(0.0000027525f);  //  1/9!

            // 2. Range Reduction
            // Map the arbitrary angle 'x' perfectly into the [-PI, PI] window.
            // Math: cycles = floor((x / 2PI) + 0.5)
            simd cycles = floor(fma(x, INV_TWO_PI, simd(0.5f)));

            // ==========================
            // FLOATING-POINT PRECISION
            // ==========================
            /*
                - All floating-point numbers lose precision as they grow.
                - Use Cody-Waite Range Reduction to preserve the mantissa bits no matter the scale (i.e., open-world games with large distances).
                - Split the 2PI contant into two parts to preserve precision.
                - Prevents catastrophic floating-point cancellations when generating waves or wind at massive distances/times.
            */
            
            // 1. Core Constants
            const simd INV_TWO_PI(0.159154943f);
            const simd TWO_PI_A(-6.28318501f);    // High precision chunk
            const simd TWO_PI_B(-2.96690463e-7f); // Low precision tail

            // 2. Range Reduction
            simd cycles = floor(fma(x, INV_TWO_PI, simd(0.5f)));

            // Cody-Waite 2-part FMA subtraction preserves mantissa precision perfectly!
            // Math: x_wrapped = x - (cycles * 2PI)
            // We use FMA to subtract without losing precision: x_wrapped = fma(cycles, -2PI, x)
            simd x_wrapped = fma(cycles, TWO_PI_A, x);    // simd x_wrapped = fma(cycles, simd(-6.283185307f), x);
            x_wrapped = fma(cycles, TWO_PI_B, x_wrapped);

            // 3. Prepare x^2 for the polynomial
            simd x2 = x_wrapped * x_wrapped;

            // 4. Horner's Method Evaluation (4 FMA instructions!)
            // Evaluates: P = C1 + x^2(C2 + x^2(C3 + x^2 * C4))
            simd poly = fma(x2, C4, C3);
            poly = fma(x2, poly, C2);
            poly = fma(x2, poly, C1);

            // 5. Final Assembly: sin(x) = x + (x * x^2 * P)
            return fma(x_wrapped * x2, poly, x_wrapped);
        }

        // Cosine is just a phase-shifted Sine wave! 
        // We add PI/2 and feed it right back into our vectorized Sine function.
        friend inline simd cos(const simd& x) requires std::is_floating_point_v<T> {
            const simd HALF_PI(1.570796326f);
            return sin(x + HALF_PI);
        }

        // --- TRANSCENDENTALS (TANGENT & ARCCOSINE) ---
        friend inline simd tan(const simd& x) requires std::is_floating_point_v<T> {
            return simd::from_native(Traits::FastTan(x.m_data));
        }

        friend inline simd acos(const simd& x) requires std::is_floating_point_v<T> {
            return simd::from_native(Traits::FastACos(x.m_data));
        }

        // --- C++26 OPERATORS (HIDDEN FRIENDS) ---
        // Using friends prevents ambiguous overload resolution and ensures identical inline compilation

        // --- SCALAR LANE ACCESS ---
        // Dynamically reads or writes an individual scalar lane within the SIMD register.
        FORCE_INLINE T& operator[](size_t index) {
            return reinterpret_cast<T*>(&m_data)[index];
        }

        // FORCE_INLINE ensures the compiler is strictly forbidden from ever emitting an actual function call overhead.
        FORCE_INLINE const T& operator[](size_t index) const {
            return reinterpret_cast<const T*>(&m_data)[index];
        }

        // --- ARITHMETIC OPERATORS ---
        friend inline simd operator+(const simd& a, const simd& b) {
            return simd::from_native(Traits::add(a.m_data, b.m_data));
        }

        friend inline simd operator-(const simd& a, const simd& b) {
            return simd::from_native(Traits::sub(a.m_data, b.m_data));
        }

        friend inline simd operator*(const simd& a, const simd& b) {
            return simd::from_native(Traits::mul(a.m_data, b.m_data));
        }

        // Division explicitly restricted to floating point traits
        friend inline simd operator/(const simd& a, const simd& b) requires std::is_floating_point_v<T> {
            return simd::from_native(Traits::div(a.m_data, b.m_data));
        }

        inline simd& operator+=(const simd& rhs) { m_data = Traits::add(m_data, rhs.m_data); return *this; }
        inline simd& operator-=(const simd& rhs) { m_data = Traits::sub(m_data, rhs.m_data); return *this; }
        inline simd& operator*=(const simd& rhs) { m_data = Traits::mul(m_data, rhs.m_data); return *this; }
        inline simd& operator/=(const simd& rhs) requires std::is_floating_point_v<T> { m_data = Traits::div(m_data, rhs.m_data); return *this; }

        // --- RELATIONAL & CONDITIONALS ---
        // In C++26, comparison returns a mask, not a boolean
        friend inline mask_type operator>(const simd& a, const simd& b) {
            return mask_type::from_native(Traits::cmp_gt(a.m_data, b.m_data));
        }
        
        friend inline mask_type operator<(const simd& a, const simd& b) { 
            return mask_type::from_native(Traits::cmp_lt(a.m_data, b.m_data)); 
        }
        
        friend inline mask_type operator==(const simd& a, const simd& b) { 
            return mask_type::from_native(Traits::cmp_eq(a.m_data, b.m_data)); 
        }

        // Google Highway's "IfThenElse" equivalent
        // The C++26 proposal suggests letting `mask ? a : b` work natively, but 
        // functionally it requires a blend operation.
        static inline simd choose(const mask_type& mask, const simd& true_val, const simd& false_val) {
            return simd::from_native(Traits::blend(mask.m_mask, true_val.m_data, false_val.m_data));
        }

        // --- MATH FUNCTIONS ---

        // Replicates: std::fma(a, b, c)
        friend inline simd fma(const simd& a, const simd& b, const simd& c) {
            return simd::from_native(Traits::fmadd(a.m_data, b.m_data, c.m_data));
        }

        // Replicates: std::sqrt(a)
        friend inline simd sqrt(const simd& a) {
            return simd::from_native(Traits::sqrt(a.m_data));
        }

        // C++26 Portable Horizontal Reduction (Hidden Friend)
        // By declaring this as a friend inside the class, the compiler knows exactly 
        // which 'reduce' to call when you pass it a simd<float>.
        friend inline T reduce(const simd& a) {
            return Traits::reduce_add(a.m_data);
        }

        // ===================================================
        // ADVANCED HORIZONTAL REDUCTIUON
        // ===================================================
        /*
            - Used for frustum culling and AABB generation.
            - hmin() gives left most edge of the box.
            - hmax() gives right most edge.
        */

        friend inline T hmin(const simd& a) requires std::is_floating_point_v<T> { return Traits::reduce_min(a.m_data); }
        friend inline T hmax(const simd& a) requires std::is_floating_point_v<T> { return Traits::reduce_max(a.m_data); }

        // --- UNARY OPERATORS ---
        friend inline simd operator-(const simd& a) {
            return simd::from_native(Traits::negate(a.m_data));
        }

        // --- BITWISE MATH (floats for IEEE-754 manipulation) ---
        friend inline simd operator^(const simd& a, const simd& b) {
            return simd::from_native(Traits::bit_xor(a.m_data, b.m_data));
        }

        // UPDATE: Remove the 'requires std::is_integral_v' from AND and OR
        friend inline simd operator|(const simd& a, const simd& b) {
            return simd::from_native(Traits::bit_or(a.m_data, b.m_data));
        }

        friend inline simd operator&(const simd& a, const simd& b) {
            return simd::from_native(Traits::bit_and(a.m_data, b.m_data));
        }

        // C++20 (Concpets): Add 'requires std::is_integral_v<T>' to the bitwise hidden friends to restrict bitwise math exclusively to integer types.
        // KEEP the restriction on bit-shifts! Shifting a float destroys the exponent (ruins IEEE 754 layout).
        friend inline simd operator<<(const simd& a, T shift) requires std::is_integral_v<T> {
            return simd::from_native(Traits::shift_l(a.m_data, static_cast<int>(shift)));
        } 

        friend inline simd operator>>(const simd& a, T shift) requires std::is_integral_v<T> {
            return simd::from_native(Traits::shift_r(a.m_data, static_cast<int>(shift)));
        }

        // ============================================
        // VECTOR INITIALIZATION
        // ============================================
        /*  
            - Allows us to generate a vector containing an ascending sequence of numbers (e.g., [0, 1, 2, 3, ...]).
            - Instantly generates staggered offsets and indices.
        */

        // Generates [0, 1, 2, 3...] inside a single vector
        static inline simd iota() {
            // Can be implemented using a fast static constant array load in the traits, or by broadcasting a base value and adding a predefined sequence register.
            alignas(64) T seq[size()];
            for (int i = 0; i < size(); ++i) seq[i] = static_cast<T>(i);
            return simd(seq); // Calls your explicit load constructor
        }

        // Generates [base, base+step, base+step*2...]
        static inline simd iota(T base, T step) {
            return simd(base) + (iota() * simd(step));
        }

        // ============================================
        // VECTOR RESIZING (SPLIT & CONCAT)
        // ============================================
        
        // Glues two half-width vectors into one full-width vector
        // SFINAE ensures this only compiles if the current Abi is AVX2 or AVX-512
        template <typename HalfAbi>
        friend inline simd concat(const simd<T, HalfAbi>& a, const simd<T, HalfAbi>& b) 
        requires (
            std::is_same_v<T, float> && 
            (
                (std::is_same_v<Abi, simd_abi::avx2> && std::is_same_v<HalfAbi, simd_abi::sse41>) ||
                (std::is_same_v<Abi, simd_abi::avx512> && std::is_same_v<HalfAbi, simd_abi::avx2>)
            )
        ) {
            return simd::from_native(Traits::concat(a.native_handle(), b.native_handle()));
        }
        
        // Splits this vector into a pair of half-width vectors
        // SFINAE ensures this only compiles if the target HalfAbi makes sense for the current silicon
        template <typename HalfAbi>
        inline std::pair<simd<T, HalfAbi>, simd<T, HalfAbi>> split() const 
        requires (
            std::is_same_v<T, float> && 
            (
                (std::is_same_v<Abi, simd_abi::avx2> && std::is_same_v<HalfAbi, simd_abi::sse41>) ||
                (std::is_same_v<Abi, simd_abi::avx512> && std::is_same_v<HalfAbi, simd_abi::avx2>)
            )
        ) {
            typename detail::simd_traits<T, HalfAbi>::register_type low, high;
            Traits::split(m_data, low, high);
            
            return {
                simd<T, HalfAbi>::from_native(low),
                simd<T, HalfAbi>::from_native(high)
            };
        }

        // ============================================
        // VECTOR SWIZZLING & SHUFFLING (SOA -> AOS)
        // ============================================
        /*
            - GPUs demands the AOS (Array of Structs) layout.
            - This allows us to freely translate between SOA (Struct of Arrays) or physics and AOS (Array of Structs) for GPU buffer uploads without stalling the CPU.
            - swizzle allows us to create a AOS vector and run horizontal dot products inside a single register, bypassing expensive memory extractions entirely.
        */

        // Usage: mySimd.swizzle<1, 0, 3, 2>();
        template <int i0, int i1, int i2, int i3>
        inline simd swizzle() const {
            static_assert(i0 >= 0 && i0 < 4 && i1 >= 0 && i1 < 4 && i2 >= 0 && i2 < 4 && i3 >= 0 && i3 < 4, 
                "Swizzle indices must be 0, 1, 2, or 3");

            // Moves data horizontally across lanes (e.g., [x, y, z, w] to [y, x, w, z]) to perform matrix multiplication or dot products inside a single SIMD register.
            return simd::from_native(Traits::template shuffle<i0, i1, i2, i3>(m_data));
        }

        // 2-Lane Swizzle (Required for double precision on SSE4.1 / NEON)
        template <int i0, int i1>
        inline simd swizzle() const requires (Traits::size == 2) {
            static_assert(i0 >= 0 && i0 < 2 && i1 >= 0 && i1 < 2, 
                "Swizzle indices for 2-lane vectors must be 0 or 1");
            return simd::from_native(Traits::template shuffle<i0, i1>(m_data));
        }

        // Common Engine Splats (Broadcast a single lane across the entire register)
        inline simd splat_x() const { return swizzle<0, 0, 0, 0>(); }
        inline simd splat_y() const { return swizzle<1, 1, 1, 1>(); }
        inline simd splat_z() const { return swizzle<2, 2, 2, 2>(); }
        inline simd splat_w() const { return swizzle<3, 3, 3, 3>(); }
    };

    // =========================================================
    // PROXY OBJECT & SIMD CASTING 
    // =========================================================

    // A lightweight proxy object that executes masked operations, overloads operators to perform a branchless hardware blend.
    template <typename T, typename Abi>
    struct WhereExpression {
        const simd_mask<T, Abi>& mask;
        simd<T, Abi>& target; // Reference to the batch we are modifying

        // SIMD batch assignments

        // Replicates: std::simd::where(mask, x) = 0.0f;
        inline void operator=(T scalar) {
            target = simd<T, Abi>::choose(mask, simd<T, Abi>(scalar), target);
        }
        inline void operator+=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target + rhs, target);
        }
        inline void operator-=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target - rhs, target);
        }

        // Replicates: std::simd::where(mask, x) *= invLen;
        inline void operator*=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target * rhs, target);
        }

        inline void operator/=(const simd<T, Abi>& rhs) requires std::is_floating_point_v<T> { 
            target = simd<T, Abi>::choose(mask, target / rhs, target); 
        }

        // Assign another SIMD Batch
        inline void operator=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, rhs, target);
        }
    };

    // The free function that mirrors std::simd::where
    template <typename T, typename Abi>
    [[nodiscard]] FORCE_INLINE WhereExpression<T, Abi> where(const simd_mask<T, Abi>& mask, simd<T, Abi>& target) {
        return WhereExpression<T, Abi>{mask, target};
    }

    // The C++26 SIMD Casting.
    template <typename ToType, typename FromType, typename Abi>
    inline simd<ToType, Abi> simd_cast(const simd<FromType, Abi>& from) {
        // Fetch the raw hardware cast from the traits, and explicitly construct the new SIMD type.
        // (Assuming from_data is accessed via a getter or friend declaration).
        auto raw_cast = detail::simd_traits<FromType, Abi>::template cast_to<ToType>(from.native_handle());
        return simd<ToType, Abi>::from_native(raw_cast);
    }

    // ========================================================
    // TIER 1: SOA (Struct of Arrays) - "The Number Cruncher"
    // ========================================================
    // Maps to the widest available CPU register natively (e.g., 256-bit on AVX2, 128-bit on NEON).
    // Particle physics, Job Systems, Audio Processing, Culling loops.
    // WARNING: Do NOT use inside structs meant for network serialization or GPU buffers.
    template <typename T>
    using WideBatch = simd<T, simd_abi::native<T>>;
    
    // Engine-wide typedefs for data processing
    using WideFloat  = WideBatch<float>;          // High-throughput SOA physics and ECS iteration.
    using WideDouble = WideBatch<double>;         // 64-bit Large World Coordinates (e.g., space simulator 10,000 asteroids orbit the sun at massive coordinates)
    using WideInt16  = WideBatch<int16_t>;        // 16 lanes audio mixing (e.g., combine multiple sound effects into a master bus, mix 16 audio samples per clock cycle).
    using WideInt32  = WideBatch<int32_t>;        // High-throughput SOA physics and ECS iteration.
    using WideUInt8  = WideBatch<uint8_t>;        // 32 lanes voxel/color processing (e.g., image processing, post processing, particle color lifecycles, 8 RGBA pixels in a single clock cycle).
    using WideUInt32 = WideBatch<uint32_t>;       

    // ========================================================
    // TIER 2: AOS (Array of Structs) - "The Geometric Standard"
    // ========================================================
    // Strictly locked to 128-bit (4 lanes) across ALL platforms.
    // On AVX2 systems, this deliberately steps down to SSE4.1 ABI.
    // Transform Matrices, Vectors, Quaternions, GPU Uniform Buffers.
    template <typename T>
    #if ENGINE_ARCH_NEON
        using FixedBatch4 = simd<T, simd_abi::neon>;
    #elif ENGINE_ARCH_AVX512 || ENGINE_ARCH_AVX2 || ENGINE_ARCH_SSE41
        // ALL x86/x64 PC and Console platforms drop to SSE for 128-bit geometry
        using FixedBatch4 = simd<T, simd_abi::sse41>; 
    #else
        // Absolute fallback (scalar 4-float array)
        using FixedBatch4 = simd<T, simd_abi::scalar>; 
    #endif
    
    // Engine-wide typedefs for geometry
    using FixedFloat4  = FixedBatch4<float>;   // Guaranteed 16-byte aligned geometry/matrices for GPU uniform buffers.
    using FixedDouble  = FixedBatch4<double>;
    using FixedInt16_8 = FixedBatch4<int16_t>; // Note: A 128-bit 'Fixed' register holds EIGHT 16-bit values, not 4!
    using FixedInt32   = FixedBatch4<int32_t>;
    using FixedUInt8   = FixedBatch4<uint8_t>;
    using FixedUInt32  = FixedBatch4<uint32_t>;

    /*
        // 1. PERFECT GPU VEC4 (Guaranteed 16 Bytes on all consoles/PC)
        CACHE_CHUNK_ALIGN_16 struct GPUVector4 {
            Engine::ISAArch::FixedFloat4 data;  // Explicitly locked to 128-bit (4 lanes)
            
            FORCE_INLINE Engine::ISAArch::FixedFloat4 dot(const GPUVector4& other) const {
                auto xyzw = data * other.data;
                auto yxwz = xyzw.swizzle<1, 0, 3, 2>();
                auto sum1 = xyzw + yxwz;
                auto zwxy = sum1.swizzle<2, 3, 0, 1>();
                return sum1 + zwxy;
            }
        };

        // 2. PERFECT BATCH PROCESSOR (Scales natively to AVX2/NEON width)
        void ApplyGravityToTriggerBox(float gravity, float dt, float* RESTRICT globalPosZ, const uint32_t* RESTRICT activeEntityIDs, size_t activeCount) {
            Engine::ISAArch::WideFloat gravityStep = gravity * dt; // Uses up to 256-bit AVX2

            for (size_t i = 0; i < activeCount; i += Engine::ISAArch::WideFloat::size()) {
                Engine::ISAArch::WideUInt entityIndices(&activeEntityIDs[i]);

                Engine::ISAArch::WideFloat posX(globalPosX, entityIndices);
                Engine::ISAArch::WideFloat posY(globalPosY, entityIndices);
                Engine::ISAArch::WideFloat posZ(globalPosZ, entityIndices);

                posZ -= gravityStep;

                CACHE_CHUNK_ALIGN_32 float tempZ[Engine::ISAArch::WideFloat::size()];
                CACHE_CHUNK_ALIGN_32 uint32_t tempIndices[Engine::ISAArch::WideUInt::size()];
                
                posZ.copy_to(tempZ);
                entityIndices.copy_to(tempIndices);

                for (int lane = 0; lane < Engine::ISAArch::WideFloat::size(); ++lane) {
                    globalPosZ[tempIndices[lane]] = tempZ[lane];
                }
            }
        }
    */
}

/*
// Use the custom C++26-compliant wrapper
using NativeFloatSIMDBatch = Engine::ISAArch::FixedFloat4;
using NativeUIntBatch      = Engine::ISAArch::simd<uint32_t>;

constexpr std::size_t NATIVE_BATCH_SIZE = NativeFloatSIMDBatch::size();
constexpr std::size_t NATIVE_SIMD_BATCH_ALIGN = alignof(NativeFloatSIMDBatch);

struct alignas(NATIVE_SIMD_BATCH_ALIGN) SIMDVector3D {
    NativeFloatSIMDBatch x, y, z;

    // Dot Product with explicit hardware FMA mapping
    FORCE_INLINE NativeFloatSIMDBatch dot_fma(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
        NativeFloatSIMDBatch res = x * bx;
        res = Engine::ISAArch::fma(y, by, res);
        res = Engine::ISAArch::fma(z, bz, res);
        return res;
    }

    FORCE_INLINE void normalize() {
        NativeFloatSIMDBatch sqLen = length_sq();
        NativeFloatSIMDBatch epsilon = 1e-8f;
        
        auto validMask = sqLen > epsilon;

        // Custom where-proxy seamlessly executing branchless assignment
        NativeFloatSIMDBatch safeSqLen = sqLen;
        Engine::ISAArch::where(!validMask, safeSqLen) = 1.0f; 

        // Division is too expensive (10-15 clock cycles)
        // NativeFloatSIMDBatch invLen = 1.0f / Engine::ISAArch::sqrt(safeSqLen);

        // Fast, refined, division-free inverse square root! Solves the expensive division problem, provides a performance boost.
        NativeFloatSIMDBatch invLen = Engine::ISAArch::rsqrt(safeSqLen);

        // Masked multiplications execute cleanly via the proxy overload
        Engine::ISAArch::where(validMask, x) *= invLen;
        Engine::ISAArch::where(validMask, y) *= invLen;
        Engine::ISAArch::where(validMask, z) *= invLen;

        Engine::ISAArch::where(!validMask, x) = 0.0f;
        Engine::ISAArch::where(!validMask, y) = 0.0f;
        Engine::ISAArch::where(!validMask, z) = 0.0f;
    }
};

using NativeFloat = Engine::ISAArch::simd<float>;

struct GPUVector4 {
    NativeFloat data; // Holds [X, Y, Z, W] (and an identical second set on AVX2)
    
    // Fast Horizontal Dot Product inside a single register!
    FORCE_INLINE NativeFloat dot(const GPUVector4& other) const {
        // 1. Multiply the lanes vertically: [x1*x2, y1*y2, z1*z2, w1*w2]
        NativeFloat xyzw = data * other.data;

        // 2. Swizzle to swap adjacent pairs: [y1*y2, x1*x2, w1*w2, z1*z2]
        NativeFloat yxwz = xyzw.swizzle<1, 0, 3, 2>();

        // 3. Add them together: [(x+y), (x+y), (z+w), (z+w)]
        NativeFloat sum1 = xyzw + yxwz;

        // 4. Swizzle the upper half to the lower half: [(z+w), (z+w), (x+y), (x+y)]
        NativeFloat zwxy = sum1.swizzle<2, 3, 0, 1>();

        // 5. Final addition. Every lane now contains the full Dot Product result!
        // [ (x+y+z+w), (x+y+z+w), (x+y+z+w), (x+y+z+w) ]
        return sum1 + zwxy;
    }
};

using NativeFloat = Engine::ISAArch::simd<float>;
using NativeUInt  = Engine::ISAArch::simd<uint32_t>;

// The dense, contiguous arrays containing all transforms in the level
float* globalPosX; 
float* globalPosY;
float* globalPosZ;

// The scattered array of Entity IDs currently standing inside the trigger box
uint32_t* activeEntityIDs;
size_t activeCount;

void ApplyGravityToTriggerBox(float gravity, float dt) {
    NativeFloat gravityStep = gravity * dt;

    // Loop through the active entities in SIMD batches
    for (size_t i = 0; i < activeCount; i += NativeFloat::size()) {
        
        // 1. Load a batch of 8 scattered Entity IDs (e.g., [4, 102, 59, 881...])
        NativeUInt entityIndices(&activeEntityIDs[i]);

        // 2. THE GATHER: Instantly pluck the scattered X, Y, and Z coordinates from memory
        // and pack them perfectly into your SIMD registers.
        NativeFloat posX(globalPosX, entityIndices);
        NativeFloat posY(globalPosY, entityIndices);
        NativeFloat posZ(globalPosZ, entityIndices);

        // 3. Process the math (Apply gravity to the Z axis)
        posZ -= gravityStep;

        // 4. Write back to memory
        // (Note: Hardware "Scatter" is not widely supported on AVX2, so we 
        // flush the results back to the scattered addresses via scalar memory stores).
        alignas(32) float tempZ[NativeFloat::size()];
        alignas(32) uint32_t tempIndices[NativeUInt::size()];
        
        posZ.copy_to(tempZ);
        entityIndices.copy_to(tempIndices);

        for (int lane = 0; lane < NativeFloat::size(); ++lane) {
            globalPosZ[tempIndices[lane]] = tempZ[lane];
        }
    }
}

using NativeFloat = Engine::ISAArch::FixedFloat4;

// By allowing floating-point values to interact with standard (^, &, |) operators, we are effectively letting your gameplay programmers act like compiler engineers, manipulating pure binary with the safety of C++ types.
FORCE_INLINE NativeFloat GetSafeInverseRayDirection(const NativeFloat& dir) {
    // 1. Create a mask of the lanes that are exactly 0.0f
    auto isZero = (dir == NativeFloat(0.0f));

    // 2. Instead of branching, we force the 0.0f lanes to become a tiny number.
    NativeFloat safeDir = dir;
    
    // 3. We use our proxy to assign a tiny non-zero float to the empty lanes.
    // BUT we want to preserve the sign (positive zero vs negative zero).
    // Using bitwise XOR logic, we can flip signs perfectly without multiplication!
    Engine::ISAArch::where(isZero, safeDir) = 1e-8f; 

    // 4. Safe hardware reciprocal (1.0 / safeDir)
    return Engine::ISAArch::rcp(safeDir);
}
*/
