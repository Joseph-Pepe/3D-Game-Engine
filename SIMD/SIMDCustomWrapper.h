#pragma once
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <format>
#include <print> // std::println

// ==================================================
// INSTRUCTION SET ARCHITECTURES (ISA)
// ==================================================
/*
    - x86_64 (AMD Zen 2): Xbox Series X/S & PS5 | AVX2 (256-bit registers)
    - ARM64 (ARM): Nintendo Switch 2 (Nvidia Tegra) & Apple Silicon (M1/M2/M3) | ARM NEON (128-bit registers)
    - Legacy: Baseline Legacy PC | SSE4.1 (128 bit registers)
*/

// --- HARDWARE DETECTION & INCLUDES ---
#if defined(__AVX2__)
    // AVX2: Xbox Series X/S, PS5, Modern PC
    #include <immintrin.h>
    #define ENGINE_ARCH_AVX2 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM NEON: Apple Silicon, Switch 2, Android, Windows on ARM
    #include <arm_neon.h>
    #define ENGINE_ARCH_NEON 1
#elif defined(__SSE4_1__)
    // SSE4.1: Legacy PC Fallback
    #include <immintrin.h> // immintrin handles all x86 SIMD headers
    #define ENGINE_ARCH_SSE41 1
#else
    #error "Engine Compiler Error: Unsupported CPU architecture. AVX2, NEON, or SSE4.1 instruction sets are strictly required."
#endif

// ===================================================
// UNIFORM MEMORY ALIGNMENT MACROS (MSVC, Clang, GCC)
// ===================================================
// Cache line sizes are typically 64 bytes on modern CPUs.
// Vulkan/DirectX require 16-byte alignment for vec4.
// Prevents the compiler from padding our structs differently on a Nintendo Switch vs PC. 
#define CACHE_CHUNK_ALIGN_16 alignas(16)
#define CACHE_CHUNK_ALIGN_32 alignas(32)
#define CACHE_CHUNK_ALIGN_64 alignas(64)

// inline is a suggestion to the compiler, __forceinline will force the compiler to flatten the math directly into the execution path.
#if defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define FORCE_INLINE inline
#endif

// Makes a promise to the compiler that data arrays never overlap.
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
    #define RESTRICT __restrict__
#endif

void LogHardwareArchitecture() {
    #if defined(ENGINE_ARCH_AVX2)
        std::println("[AVX2, X86]: Intel/AMD based architecture detected.");
    #elif defined(ENGINE_ARCH_NEON)
        std::println("[ARM64]: ARM based architecture detected.");
    #elif defined(ENGINE_ARCH_SSE41)
        std::println("[SSE4.1]: Legacy based architecture detected.");
    #endif
}

namespace Engine::ISAArch {
    // ==========================================
    // ABI NAMESPACE (Hardware Tags)
    // ==========================================
    namespace simd_abi {
        struct scalar {};
        struct avx2 {};
        struct neon {};

        // C++26 'native' alias: Automatically deduces the best hardware vector length at compile time based on your compiler flags (e.g., /arch:AVX2).
        template <typename T>
        #if ENGINE_ARCH_AVX2
            using native = avx2;     // Xbox/PS5/PC (256-bit (8 floats))
        #elif ENGINE_ARCH_NEON
            using native = neon;     // Nintendo Switch 2 / Apple Silicon (128-bit (4 floats))
        #else
            using native = scalar;   // Legacy PC
        #endif
    }

    // ==========================================
    // BACKEND INTRINSICS (Traits/Storage)
    // ==========================================
    namespace detail {
        // Primary template (Undefined)
        template <typename T, typename Abi> struct simd_traits;

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

                static inline bool mask_any(mask_type a) { return _mm_movemask_ps(a) != 0; }
                static inline bool mask_all(mask_type a) { return _mm_movemask_ps(a) == 0x0F; }

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
                
                // Relational
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm_cmpgt_epi32(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm_cmpgt_epi32(b, a); } // Flipped for Less-Than
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

                // Horizontal Reduction
                // AVX2 doesn't have a single-instruction horizontal add across 256 bits, so we fold it in half repeatedly.
                static inline float reduce_add(register_type a) {
                    // Step 1: Fold 256-bit into 128-bit
                    __m128 lo = _mm256_castps256_ps128(a);
                    __m128 hi = _mm256_extractf128_ps(a, 1);
                    lo = _mm_add_ps(lo, hi);
                    
                    // Step 2: Fold 128-bit down to 64-bit, then down to 32-bit scalar
                    __m128 shuf = _mm_movehdup_ps(lo);
                    __m128 sums = _mm_add_ps(lo, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    
                    return _mm_cvtss_f32(sums);
                }

                static inline bool mask_any(mask_type a) { 
                    return _mm256_movemask_ps(a) != 0; 
                }
                static inline bool mask_all(mask_type a) { 
                    return _mm256_movemask_ps(a) == 0xFF; 
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
                static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
                static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
                static inline register_type shift_l(register_type a, int imm) { return _mm256_slli_epi32(a, imm); }
                
                static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmpgt_epi32(a, b); }
                static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmpgt_epi32(b, a); } // Flipped operands for Less-Than
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
                    // Both MSVC and Clang will aggressively fold this into a single instruction.
                    register_type res = vdupq_n_f32(0.0f); 
                    res = vsetq_lane_f32(vgetq_lane_f32(a, i0), res, 0);
                    res = vsetq_lane_f32(vgetq_lane_f32(a, i1), res, 1);
                    res = vsetq_lane_f32(vgetq_lane_f32(a, i2), res, 2);
                    res = vsetq_lane_f32(vgetq_lane_f32(a, i3), res, 3);
                    return res;
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
        #endif // ENGINE_ARCH_NEON
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
        static inline simd_mask from_native(typename Traits::mask_type mask) { 
            return simd_mask(mask); 
        }

        // --- MIXED-TYPE MASK CASTING ---
        template <typename U>
        inline simd_mask<U, Abi> cast_to() const {
            if constexpr (std::is_same_v<T, U>) {
                return *this;
            } // UINT32 to FLOAT
            else if constexpr (std::is_same_v<T, uint32_t> && std::is_same_v<U, float>) {
                if constexpr (std::is_same_v<Abi, simd_abi::avx2>) {
                    return simd_mask<U, Abi>(_mm256_castsi256_ps(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::sse41>) {
                    return simd_mask<U, Abi>(_mm_castsi128_ps(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::neon>) {
                    return simd_mask<U, Abi>(m_mask);
                }
            } // FLOAT to UINT32
            else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, uint32_t>) {
                if constexpr (std::is_same_v<Abi, simd_abi::avx2>) {
                    return simd_mask<U, Abi>(_mm256_castps_si256(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::sse41>) {
                    return simd_mask<U, Abi>(_mm_castps_si128(m_mask));
                } else if constexpr (std::is_same_v<Abi, simd_abi::neon>) {
                    return simd_mask<U, Abi>(m_mask);
                }
            } else {
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
    };

    // ==========================================
    // C++26 SIMD FRONTEND
    // ==========================================
    template <typename T, typename Abi>
    class simd {
    private:
        using Traits = detail::simd_traits<T, Abi>;
        typename Traits::register_type m_data;

        // Internal constructor for operations to bypass memory
        explicit inline simd(typename Traits::register_type data) : m_data(data) {}

    public:
        // C++26 Type Definitions
        using value_type = T;
        using abi_type   = Abi;
        using mask_type  = simd_mask<T, Abi>;

        static constexpr int size() { return Traits::size; }

        // --- HARDWARE INTEROP ---
        typename Traits::register_type native_handle() const { return m_data; }
        static simd from_native(typename Traits::register_type raw) { return simd(raw); }

        // --- CONSTRUCTORS ---
        // C++26 guarantees default simds are uninitialized, just like raw floats!
        simd() = default; 
        
        // Broadcast Constructor
        simd(T value) : m_data(Traits::broadcast(value)) {} 
        
        // C++26 Memory Load (P1928 allows implicit load from memory)
        explicit simd(const T* mem) : m_data(Traits::load(mem)) {}

        // --- NON-CONTIGUOUS MEMORY LOAD (GATHER) ---
        // C++26 Hardware Gather Constructor.
        // Takes a base pointer and a SIMD batch of array indices.
        explicit simd(const T* base_addr, const simd<uint32_t, Abi>& indices) 
            : m_data(Traits::gather(base_addr, indices.native_handle())) {}

        // --- MEMORY STORE ---
        void copy_to(T* mem) const { Traits::store(mem, m_data); }

        friend inline simd min(const simd& a, const simd& b) { return simd(Traits::min(a.m_data, b.m_data)); }
        friend inline simd max(const simd& a, const simd& b) { return simd(Traits::max(a.m_data, b.m_data)); }
        friend inline simd clamp(const simd& v, const simd& lo, const simd& hi) { return min(max(v, lo), hi); }

        friend inline simd rsqrt(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::rsqrt(a.m_data)); }
        friend inline simd rcp(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::rcp(a.m_data)); }

        friend inline simd abs(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::abs(a.m_data)); }
        friend inline simd floor(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::floor(a.m_data)); }
        friend inline simd ceil(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::ceil(a.m_data)); }

        // =======================================================================
        // VECTORIZED TRIGONOMETRY (MINIMAX POLYNOMIAL APPROX & HORNER'S METHOD)
        // =======================================================================
        /*
            - Estimates a curve using a 9th degree polynomial.
            - Used to animate thousands of skeletal meshes, calculate camera FOV projections, or generate procedural wind for foliage.
            - Used to simulate ocean waves, or project 3D coordinates onto a 2D screen.
            - Sine and Cosine are needed.
        */
        
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
            
            // Math: x_wrapped = x - (cycles * 2PI)
            // We use FMA to subtract without losing precision: x_wrapped = fma(cycles, -2PI, x)
            simd x_wrapped = fma(cycles, simd(-6.283185307f), x);

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

        // --- C++26 OPERATORS (HIDDEN FRIENDS) ---
        // Using friends prevents ambiguous overload resolution and ensures identical inline compilation

        // --- ARITHMETIC OPERATORS ---
        friend inline simd operator+(const simd& a, const simd& b) {
            return simd(Traits::add(a.m_data, b.m_data));
        }

        friend inline simd operator-(const simd& a, const simd& b) {
            return simd(Traits::sub(a.m_data, b.m_data));
        }

        friend inline simd operator*(const simd& a, const simd& b) {
            return simd(Traits::mul(a.m_data, b.m_data));
        }

        // Division explicitly restricted to floating point traits
        friend inline simd operator/(const simd& a, const simd& b) requires std::is_floating_point_v<T> {
            return simd(Traits::div(a.m_data, b.m_data));
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
            return simd(Traits::blend(mask.m_mask, true_val.m_data, false_val.m_data));
        }

        // --- MATH FUNCTIONS ---

        // Replicates: std::fma(a, b, c)
        friend inline simd fma(const simd& a, const simd& b, const simd& c) {
            return simd(Traits::fmadd(a.m_data, b.m_data, c.m_data));
        }

        // Replicates: std::sqrt(a)
        friend inline simd sqrt(const simd& a) {
            return simd(Traits::sqrt(a.m_data));
        }

        // C++26 Portable Horizontal Reduction (Hidden Friend)
        // By declaring this as a friend inside the class, the compiler knows exactly 
        // which 'reduce' to call when you pass it a simd<float>.
        friend inline T reduce(const simd& a) {
            return Traits::reduce_add(a.m_data);
        }

        // --- UNARY OPERATORS ---
        friend inline simd operator-(const simd& a) {
            return simd(Traits::negate(a.m_data));
        }

        // --- BITWISE MATH (floats for IEEE-754 manipulation) ---
        friend inline simd operator^(const simd& a, const simd& b) {
            return simd(Traits::bit_xor(a.m_data, b.m_data));
        }

        // UPDATE: Remove the 'requires std::is_integral_v' from AND and OR
        friend inline simd operator|(const simd& a, const simd& b) {
            return simd(Traits::bit_or(a.m_data, b.m_data));
        }

        friend inline simd operator&(const simd& a, const simd& b) {
            return simd(Traits::bit_and(a.m_data, b.m_data));
        }

        // C++20 (Concpets): Add 'requires std::is_integral_v<T>' to the bitwise hidden friends to restrict bitwise math exclusively to integer types.
        // KEEP the restriction on bit-shifts! Shifting a float destroys the exponent (ruins IEEE 754 layout).
        friend inline simd operator<<(const simd& a, int shift) requires std::is_integral_v<T> {
            return simd(Traits::shift_l(a.m_data, shift));
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
            return simd(Traits::template shuffle<i0, i1, i2, i3>(m_data));
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
    using WideFloat = WideBatch<float>;
    using WideUInt  = WideBatch<uint32_t>;

    // ========================================================
    // TIER 2: AOS (Array of Structs) - "The Geometric Standard"
    // ========================================================
    // Strictly locked to 128-bit (4 lanes) across ALL platforms.
    // On AVX2 systems, this deliberately steps down to SSE4.1 ABI.
    // Transform Matrices, Vectors, Quaternions, GPU Uniform Buffers.
    template <typename T>
    #if ENGINE_ARCH_NEON
        using FixedBatch4 = simd<T, simd_abi::neon>;
    #else
        using FixedBatch4 = simd<T, simd_abi::sse41>; 
    #endif
    
    // Engine-wide typedefs for geometry
    using FixedFloat4 = FixedBatch4<float>;
    using FixedUInt4  = FixedBatch4<uint32_t>;

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
