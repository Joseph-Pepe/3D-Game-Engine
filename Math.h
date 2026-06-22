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

#if __has_include(<inplace_vector>)
    /*
        // Replaces (std::vector). Zero heap allocations. Data is perfectly contiguous on the stack.
        // Extremely cache friendly for your SIMD wrappers.
        std::inplace_vector<Vector3D, 64> localCluster;
    */
    #include <inplace_vector> // C++26 API provides a vector that stores data locally without ever touching the heap allocator.
#endif 



// --- COMPILER MACROS ---
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode
#if __has_include(<simd>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    // C++26 features are unlocked (Optional) Include <simd> if you want the portable vector typedefs
    #include <simd>
    #define ENGINE_HAS_CXX26_SIMD 1
#else
    // Fallback for C++23 and older
    #define ENGINE_HAS_CXX26_SIMD 0
#endif

// C++26 linear algebra (compiler dependent availability)
#if __has_include(<linear_algebra>)
    #include <linear_algebra>
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
/*
    - XorShift32 has statistical flaws, low-dimensional equidistribution errors.
    - For things like procedural generation, terrain noise, or monte carlo raytracing it will eventuallly produce visible banding or artifacting.

    - PCG32 (Permuted Congruential Generator) uses an LCG (Linear Congruential Generator), but applies a bitwise output permutation ot destroy the predictability.
    - Its just as fast and better than XorShift32.
*/

// --- FAST STATELESS PRNG ---
// Executes in ~1-2 clock cycles entirely inside the ALU registers.
FORCE_INLINE uint32_t XorShift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// --- FAST STATELESS PRNG (PCG32) ---
// Requires a 64-bit state, returns a perfectly distributed 32-bit random number.
FORCE_INLINE uint32_t PCG32(uint64_t& state) {
    uint64_t oldState = state;
    
    // 1. Advance internal state (Multiplier and Increment are PCG standards)
    state = oldState * 6364136223846793005ULL + 1ULL;
    
    // 2. Calculate output function (XSH RR - Xorshift High bits, Random Rotate)
    uint32_t xorshifted = static_cast<uint32_t>(((oldState >> 18u) ^ oldState) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldState >> 59u);
    
    // 3. Bitwise rotation 
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// --- DIVISION-LESS RANGE MAPPING ---
// Maps a random 32-bit integer to [0, range) using multiplication instead of division.
FORCE_INLINE uint32_t MapToRange(uint32_t randomVal, uint32_t range) {
    return static_cast<uint32_t>((static_cast<uint64_t>(randomVal) * static_cast<uint64_t>(range)) >> 32);
}

// ====================================================
// 2D & 3D COORDINATE SPACE
// ====================================================

/* [3D SPACE GRID SIZE]
    - Maximum grid bounds in the engine is based on [WORLD_SIZE (2000) / CELL_SIZE (4) = 500] + 1 (safety buffer)
    - 3D Space = Grid (501 x 501 x 501)
    - The highest coordinate value we hash is 501 which easily fits within 10 bits (0 to 1023).
    - InitMortonLUT: Pre-calculate the bit shifts for every possible coordinate from 0 to 1023 at engine startup and permanently store them in L1 cache.
*/

/* APPROACH #1: [2D Array Flattening]
    - Simulate a 2D universe (2D grid) inside a 1D memory stick (i.e., line of memory addresses).
    - Slice the 2D space and lay it in row-major order (2D array flattening).
    - gridX is a baseline dimension with a stride of 1 from [0] to [500].
    - Y-Slice: gridY access requires skipping the entire row of gridX cells (x-axis), so we need to jump by GRID_WIDTH addresses in memory (e.g., [501], gridY + 1 = [1002]).
    - This hash lays stores the data in addresses linearly, so that the hardware prefetcher instantly recognizes the pattern and loads them in the ultra-fast L1 cache behind the scenes (i.e., improves frame rate by 7-10fps).
    - Good hashing matters because it can improve performance if data is spaced out linearly. 
    - This allows them to look to the right at there neighbor to check for a collision.
*/

/* APPROACH #2: [3D Space Hash]
    - Create a 3D spacial hash with [GRID_HEIGHT * GRID_WIDTH], e.g., [1MB = 250,000 cells (int) = 500 x 500]
    - Avoids eating up RAM just for the histogram. 
    - Avoid using [GRID_DEPTH].
    - Never use [GRID_DEPTH * GRID_HEIGHT * GRID_WIDTH], e.g., [500MB = 125,000,000 cells (int) = 500 x 500 x 500]
    - Linear array flattening works perfectly for 2D, but not 3D because the multiplier becomes 251,101
    - Z-Slice: gridZ access requires skipping an entire 2D plane (gridX, gridY rows), so we need to jump forward by GRID_WIDTH * GRID_HEIGHT addresses in memory (e.g., [251001], gridZ + 1 = [502002], jumps by ~1MB which violently ejects the currently loaded cache). 
    - hash = gridX + (gridY * GRID_WIDTH) + (gridZ * GRID_WIDTH * GRID_HEIGHT), where GRID_WIDTH * GRID_HEIGHT = 501 * 501 = 251001u.
*/

/* BEST SOLUTION: [Z-Order Curve (Morton Encoding)]
    - Interleaves the binary bits of X, Y, Z
    - If two particles are physically close in 3D space, Morton codes guarantees they are physically close in 1D memory array.
    - The memory addresses are woven together like a zipper to form [x1 y1 z1 x2 y2 z2 ...]
    - Hashing with prime numbers destroys cache locality because it scatters data in random memory addresses resulting in undetectable patterns, and starved registers waiting hundreds of cycles waiting on main memory. 
*/

// ====================================================
// SPATIAL HASHING (MORTON CODES / Z-ORDER CURVE)
// ====================================================

// --- C++26 COMPILE-TIME MORTON LUT GENERATION ---
// This function executes entirely during compilation.
consteval std::array<uint32_t, 1024> GenerateMortonTable(uint32_t shift) {
    std::array<uint32_t, 1024> table{};
    for (uint32_t i = 0; i < 1024; ++i) {
        uint32_t v = i;
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v <<  8)) & 0x0300F00F;
        v = (v | (v <<  4)) & 0x030C30C3;
        v = (v | (v <<  2)) & 0x09249249;
        table[i] = v << shift;
    }
    return table;
}

// 12KB off data embedded natively into the Read-Only Data (.rodata) segment of the compiled binary.
// Zero heap allocations. Zero runtime initialization, tightly packed in memory for hardware prefetcher to stream it into L1 cache instantly.
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableX = GenerateMortonTable(0);
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableY = GenerateMortonTable(1);
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableZ = GenerateMortonTable(2);

// // --- PHYSICAL COORDINATE GRID LOOKUP TABLE (LUT) ---
// // 1024 entries * 4 bytes = 4KB each. Easily fits in 32KB L1 Data Cache. 4KB arrays that will live permanently in the L1 CPU Cache
// // 'inline' prevents Multiple Definition linker errors in header files.
// inline std::vector<uint32_t> g_MortonTableX(1024); // Physical Grid Coordinate X, Global Memory
// inline std::vector<uint32_t> g_MortonTableY(1024); // Physical Grid Coordinate Y, Global Memory
// inline std::vector<uint32_t> g_MortonTableZ(1024); // Physical Grid Coordinate Z, Global Memory

// --- RUN-TIME MORTON LUT GENERATION ---
// void InitMortonLUT() {
//     // --- OLDER INTEL, AMD CPUs DON'T HAVE _pdep_u32 IMPLEMENTED IN THE HARDWARE (18-50 CLOCK CYCLES) ---
//     auto expandBits = [](uint32_t v) -> uint32_t {
//         v = (v | (v << 16)) & 0x030000FF;
//         v = (v | (v <<  8)) & 0x0300F00F;
//         v = (v | (v <<  4)) & 0x030C30C3;
//         v = (v | (v <<  2)) & 0x09249249;
//         return v;
//     };

//     // Solution: Bit shift version uses standard ALU instructions that can be used on every CPU Architecture.
//     for (uint32_t i = 0; i < 1024; ++i) {
//         uint32_t expanded = expandBits(i); 
//         g_MortonTableX[i] = expanded;        // Bits shifted for X (Standard)
//         g_MortonTableY[i] = expanded << 1;   // Bits shifted for Y
//         g_MortonTableZ[i] = expanded << 2;   // Bits shifted for Z
//     }
// }

// The new blindingly fast scalar fallback
// Make this constexpr for C++26 so it can be evaluated entirely at compile-time for static level geometry
FORCE_INLINE constexpr uint32_t getMortonCodeLUT(uint32_t x, uint32_t y, uint32_t z) {

    // 1. Bitwise mask to strictly enforce the [0, 1023] limit (0x3FF is 1023 in hex).
    // This executes in a fraction of a cycle and prevents catastrophic engine crashes.
    x &= 0x3FF; 
    y &= 0x3FF; 
    z &= 0x3FF;

    // 2. C++26 [[assume]] attribute.
    // Tells the compiler "I guarantee these are under 1024, do not generate bounds-checking assembly."
    [[assume(x < 1024 && y < 1024 && z < 1024)]];

    // 3 memory fetches and 2 bitwise ORs!
    return g_MortonTableX[x] | g_MortonTableY[y] | g_MortonTableZ[z];
}

// --- HARDWARE MORTON ENCODING (BMI2) ---
// Interleaves the bits of X, Y, and Z to preserve 3D spatial cache locality (3D array flattening), optimizes L1/L2 cache
// By passing IsLegacy as a template parameter, the compiler resolves the branch at compile-time (i.e., compiler generates two completely separate, branchless versions of the function at compile-time saving millions of evaluations per frame).
template <bool IsLegacy>
FORCE_INLINE uint32_t getMortonCode(uint32_t x, uint32_t y, uint32_t z) {
    // x bits go to slots 0, 3, 6, 9...
    // y bits go to slots 1, 4, 7, 10...
    // z bits go to slots 2, 5, 8, 11...

    // --- INTEL HASWELL+, AMD ZEN 3+ CPUs NATIVE HARDWARE SUPPORT (~3 CLOCK CYCLES) HAVE _pdep_u32 INSTRUCTION DIRECTLY WIRED INTO THE SILICON ---
    /* [!g_EngineSettings.isLegacyCPU]
        - Global memory space being used in C++ means that the compiler cannot guarantee that another thread won't change it mid-execution.
        - The compiler injects an L1 data cache to check if the boolean has changed.
        - This burns roughly 10 million clock cycles per frame just reading a boolean that never changes during gameplay [100,000 particles, 2.7 million calls per frame, ~4 clock cycles].
        
        - Solution is to move the boolean global memory to const stack variable.
        - Cache it into a local, const stack variable at the moment the thread wakes up and pass them explicitly.
        - Passing a const variable into an inline function, the compiler will optimize it and make it a pure branchless execution path.
    */
    // C++17/26 'if constexpr' ensures the false path is entirely deleted from the compiled binary.
    if constexpr (!IsLegacy) {
        // We use "masks" to define where the bits should land, bit manipulation instruction 2, parallel bits deposit.
        uint32_t mx = _pdep_u32(x, 0x09249249);
        uint32_t my = _pdep_u32(y, 0x12492492);
        uint32_t mz = _pdep_u32(z, 0x24924924);
        
        return mx | my | mz;
    } 
    // --- LEGACY CPU FALLBACK: UNIVERSAL LUT (~4 CLOCK CYCLES) BECAUSE OLDER INTEL, AMD CPUs DON'T HAVE _pdep_u32 IMPLEMENTED INTO THE SILICON ---
    else {
        // USE THE LUT! (Virtually instant) NEARLY MATCHES THE HARDWARE-LEVEL "_pdep_u32" INSTRUCTION, BUT IT WORKS FAST ON EVERY CPU EVER MADE.
        return getMortonCodeLUT(x, y, z);
    }
}

// ====================================================
// MORTON CODE - AVX2 INTEGER VECTORIZATION (FAST PATH)
// ====================================================

// --- CHEWS THROUGH THE MASSIVE 100,000+ PARTICLE ARRAYS, 8 PARTICLES AT A TIME ---
FORCE_INLINE __m256i expandBits_AVX2(__m256i v) {
    // Apply the magic bits to 8 integers simultaneously 
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 16)), _mm256_set1_epi32(0x030000FF));
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v,  8)), _mm256_set1_epi32(0x0300F00F));
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v,  4)), _mm256_set1_epi32(0x030C30C3));
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v,  2)), _mm256_set1_epi32(0x09249249));
    return v;
}

// --- VECTORIZE THE MAGIC BITS AND CALCULATE THE MORTON CODES FOR 8 PARTICLES (BULK DATA PROCESSING)---
FORCE_INLINE __m256i getMortonCode_AVX2(__m256i x, __m256i y, __m256i z) {
    __m256i ex = expandBits_AVX2(x);
    __m256i ey = _mm256_slli_epi32(expandBits_AVX2(y), 1);
    __m256i ez = _mm256_slli_epi32(expandBits_AVX2(z), 2);

    // Combine X, Y, and Z
    return _mm256_or_si256(_mm256_or_si256(ex, ey), ez);
}

// ======================================================================
// C++26: NATIVE SIMD ARCHITECTURE for MSVC build v14.51 and newer.
// ======================================================================
/*
    - This now scales to ARM (Apple Silicon M1/M2/M3), NEON, AMD (Playstation/Xbox) and Intel (AVX-512) automatically once compiled without a total rewrite of code (i.e., seamlessly maps those registers, cross-platform).
    - No need to use bitwise mask hacks (_mm512_cmp_ps_mask, _mm512_maskz_mul_ps) anymore b/c the compiler automatically translates these logical operators into hardware masks.
    - No more Intel-specific _mm256! 
    - Never hardcode instruction sets into your datastructures. Write once and rely on the compiler to traslate it into the widest register the hardware supports.
    - No longer need to manually hardcode intrinsics for __mm256, __mm512 versions, the C++ compiler will figure out what the native hardware is based on the build flag (e.g., -march=native, -mavx512f, -mcpu=apple-1).
    - Decouples the engine from Intel by changing the compiler target flag in CMake.
    - std::simd generated assembly is identical to manual intrinsics (1:1 match). You lose zero performance.
*/

#if ENGINE_HAS_CXX26_SIMD
    // --- 1. THE C++26 MATH LAYER (Portable SIMD) --- 
    // Use the official C++26 P1928 syntax based on the silicon it detects at compile time.
    using NativeFloatSIMDBatch = std::simd<float, std::simd_abi::native<float>>; // Let the compiler decide the widest register available on the target hardware

    // Automatically scales: 4 (SSE/NEON), 8 (AVX2), or 16 (AVX-512)
    constexpr std::size_t NATIVE_BATCH_SIZE = NativeFloatSIMDBatch::size();
    constexpr std::size_t NATIVE_SIMD_BATCH_ALIGN = alignof(NativeFloatSIMDBatch);     // Use this constant to dynamically align your memory allocators and structs! Ask the C++26 standard exactly how many bytes the current hardware needs

    // ==================================================================================
    // BULK DATA PROCESSING (SOA) - SCALES TO ANY CPU AUTOMATICALLY
    // ==================================================================================

    // Dynamic Alignment Wrapper:
    // If compiling for AVX-512, this guarantees 64-byte alignment. 
    // If compiling for ARM NEON, it guarantees 16-byte alignment.
    struct alignas(NATIVE_SIMD_BATCH_ALIGN) SIMDVector3D {
        // [NativeFloatSIMDBatch]: Instead of manual __m256 or __m512 loads, you use a template that automatically picks the widest register the hardware supports.
        // Under the hood, this is __m128, __m256, __m512, or float32x4_t. Your code no longer cares.
        NativeFloatSIMDBatch x, y, z;

        // Standard Addition
        FORCE_INLINE void add(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) {
            x += bx; // C++26 SIMD supports standard operators!
            y += by;
            z += bz;
        }

        // Standard Subtraction
        FORCE_INLINE void sub(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) {
            x -= bx; 
            y -= by; 
            z -= bz;
        }

        // Scalar Multiplication
        FORCE_INLINE void mul(const NativeFloatSIMDBatch& scalar) {
            x *= scalar;
            y *= scalar;
            z *= scalar;
        }

        // Dot Product with FMA (Fused Multiply-Add)
        // C++26 automatically fuses (a * b + c) into a single clock cycle if compiler flags allow it, or you can explicitly use std::fma overloaded for simd.
        FORCE_INLINE NativeFloatSIMDBatch dot_fma(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
            NativeFloatSIMDBatch res = x * bx;
            res = std::fma(y, by, res);
            res = std::fma(z, bz, res);
            return res;
        }

        // Standard Dot Product
        FORCE_INLINE NativeFloatSIMDBatch dot(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
            return (x * bx) + (y * by) + (z * bz);
        }

        // SOA Cross Product
        // Fused Multiply-Add cross product. Compiles to _mm512_fmsub_ps on AVX-512, and vfmaq_f32 on ARM automatically.
        FORCE_INLINE void cross(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) {
            // Explicitly using std::fma to guarantee hardware Fused Multiply-Subtract (FMS) 
            // Example: (y * bz) - (z * by) -> fma(y, bz, -(z * by))
            NativeFloatSIMDBatch rx = std::fma(y, bz, -(z * by));
            NativeFloatSIMDBatch ry = std::fma(z, bx, -(x * bz));
            NativeFloatSIMDBatch rz = std::fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }

        // Magnitude Squared
        FORCE_INLINE NativeFloatSIMDBatch length_sq() const {
            // return (x * x) + (y * y) + (z * z);

            // Nudging compiler to use FMA
            NativeFloatSIMDBatch sq = x * x;
            sq = std::fma(y, y, sq);
            sq = std::fma(z, z, sq);
            return sq;
        }

        // Magnitude
        FORCE_INLINE NativeFloatSIMDBatch length() const {
            return std::sqrt(length_sq()); // std::sqrt is overloaded for simd types!
        }

        // --- C++26 PORTABLE OPMASK LOGIC ---
        FORCE_INLINE void normalize() {
            NativeFloatSIMDBatch sqLen = length_sq();
            NativeFloatSIMDBatch epsilon = 1e-8f;
            
            // 1. Generate the Hardware Mask. 
            // -> If compiling for AVX-512, this generates an `__mmask16`.
            // -> If compiling for AVX2, this generates a `__m256` bitmask.
            auto validMask = sqLen > epsilon;

            // 2. Prevent NaN/Inf generation by patching invalid lengths to 1.0f BEFORE division.
            // If sqLen is 0, we temporarily pretend it is 1.0f so division succeeds gracefully.
            NativeFloatSIMDBatch safeSqLen = sqLen;
            std::simd::where(!validMask, safeSqLen) = 1.0f; 

            // 3. Fast-math will translate this to a hardware reciprocal square root instruction.
            // -> Emits _mm512_rsqrt14_ps on AVX-512
            // -> Emits _mm256_rsqrt_ps on AVX2
            // -> Emits vrsqrteq_f32 on ARM NEON
            NativeFloatSIMDBatch invLen = 1.0f / std::sqrt(safeSqLen);

            // 4. Masked Assignment (Multiplication).
            std::simd::where(validMask, x) *= invLen;
            std::simd::where(validMask, y) *= invLen;
            std::simd::where(validMask, z) *= invLen;

            // 5. Zero out the invalid lanes
            std::simd::where(!validMask, x) = 0.0f;
            std::simd::where(!validMask, y) = 0.0f;
            std::simd::where(!validMask, z) = 0.0f;
        }
    };

    // Let the compiler dynamically pick 4-wide (NEON), 8-wide (AVX2), or 16-wide (AVX-512) integers
    using NativeUIntBatch = std::simd<uint32_t, std::simd_abi::native<uint32_t>>;

    // --- C++26 PORTABLE MORTON CODE VECTORIZATION (CROSS-PLATFORM) ---
    FORCE_INLINE NativeUIntBatch expandBits_SIMD(NativeUIntBatch v) {
        // The compiler automatically translates these bitwise operators into vector instructions
        // e.g., _mm256_slli_epi32 and _mm256_and_si256 on Intel!
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v <<  8)) & 0x0300F00F;
        v = (v | (v <<  4)) & 0x030C30C3;
        v = (v | (v <<  2)) & 0x09249249;
        return v;
    }

    FORCE_INLINE NativeUIntBatch getMortonCode_SIMD(const NativeUIntBatch& x, const NativeUIntBatch& y, const NativeUIntBatch& z) {
        NativeUIntBatch ex = expandBits_SIMD(x);
        NativeUIntBatch ey = expandBits_SIMD(y) << 1;
        NativeUIntBatch ez = expandBits_SIMD(z) << 2;

        return ex | ey | ez;
    }
#endif



// ================================================================================
// AVX-512: This represents 16 vectors. It doesn't own memory; it just processes it.
// ================================================================================

// --- 1. THE MATH LAYER (Intel-specific Intrinsics SIMD) ---
struct SIMDVector16 {
    // --- AVX-512: 16-Wide Batch ---
    __m512 x, y, z;

    FORCE_INLINE void add(const __m512& bx, const __m512& by, const __m512& bz) {
        x = _mm512_add_ps(x, bx);
        y = _mm512_add_ps(y, by);
        z = _mm512_add_ps(z, bz);
    }

    FORCE_INLINE __m512 dot_fma(const __m512& bx, const __m512& by, const __m512& bz) const {
        __m512 res = _mm512_mul_ps(x, bx);
        res = _mm512_fmadd_ps(y, by, res);
        res = _mm512_fmadd_ps(z, bz, res);
        return res;
    }

    FORCE_INLINE __m512 dot(const __m512& bx, const __m512& by, const __m512& bz) const {
        __m512 mx = _mm512_mul_ps(x, bx);
        __m512 my = _mm512_mul_ps(y, by);
        __m512 mz = _mm512_mul_ps(z, bz);
        return _mm512_add_ps(_mm512_add_ps(mx, my), mz);
    }

    FORCE_INLINE void cross(const __m512& bx, const __m512& by, const __m512& bz) {
        // OPTIMIZATION: Fused Multiply-Subtract reduces 6 instructions to 3.
        // Formula: (y * bz) - (z * by)
        __m512 rx = _mm512_fmsub_ps(y, bz, _mm512_mul_ps(z, by));
        __m512 ry = _mm512_fmsub_ps(z, bx, _mm512_mul_ps(x, bz));
        __m512 rz = _mm512_fmsub_ps(x, by, _mm512_mul_ps(y, bx));
        x = rx; y = ry; z = rz;
    }

    FORCE_INLINE void sub(const __m512& bx, const __m512& by, const __m512& bz) {
        x = _mm512_sub_ps(x, bx);
        y = _mm512_sub_ps(y, by);
        z = _mm512_sub_ps(z, bz);
    }

    FORCE_INLINE __m512 length_sq() const {
        // OPTIMIZATION: Fused Multiply-Add
        __m512 xx = _mm512_mul_ps(x, x);
        __m512 xx_yy = _mm512_fmadd_ps(y, y, xx);
        return _mm512_fmadd_ps(z, z, xx_yy);
        // __m512 xx = _mm512_mul_ps(x, x);
        // __m512 yy = _mm512_mul_ps(y, y);
        // __m512 zz = _mm512_mul_ps(z, z);
        // return _mm512_add_ps(_mm512_add_ps(xx, yy), zz);
    }

    FORCE_INLINE __m512 length() const {
        return _mm512_sqrt_ps(length_sq());
    }

    // --- AVX-512 EXCLUSIVE OPMASK LOGIC ---
    FORCE_INLINE void normalize() {
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

// ==================================================================================
// 2. AVX2 BARE-METAL SIMD STRUCTURES (8-Wide)
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

// 96 Bytes total: Fits perfectly into two 64-byte L1 Cache lines.
struct alignas(32) ParticleBlock8 {
    __m256 x;
    __m256 y;
    __m256 z;
};

// ==================================================================================
// 3. SSE Accelerated Vectors 
// ==================================================================================
/*  
    - The w only matters when multiplying a vector with a matrix.
    - w = 0.0f (Direction): Represents a vector (like gravity or camera's forward axis). When multiplied by a matrix it ignores translation (i.e., you cannot move gravity).
    - w = 1.0f (Point): Represents a position in space (like a player or vertex). When multiplied by a matrix, the translation is applied.
*/

// [Vector3D]: Use this version if your creating a very large, persistent buffer where you don't want to blow out the stack. 
// alignas(16) guarantees that whenever this struct is created, it starts on a 16-byte boundary. No malloc required!
class alignas(16) Vector3D {
public:
    /*
         - Using a float array of 4 to align with 128-bit SSE registers.
         - x, y, z, and a padding/w element.
         - Aligning the pointer itself is good practice.
         - But the actual memory it points to is aligned by std::aligned_alloc.

           // This dynamically allocates the memory on the heap, perfectly aligned.
           // There is zero pointer-chasing. The CPU prefetcher will chew through this instantly.
           std::vector<Vector3D> largePersistentBuffer(1'000'000); 

           // Usage is clean and readable:
           Vector3D a(1.0f, 0.0f, 0.0f);
           Vector3D b(0.0f, 1.0f, 0.0f);
           Vector3D c = a + (b * 5.0f); // Completely optimized into registers by the compiler
    */
    // Anonymous union allows you to access data via names (x,y,z) OR directly as an SSE register (__m128), OR as a float array.
    union {
        __m128 reg;
        struct { float x, y, z, w; };
    };

    // --- HOMOGENEOUS COORDINATE ENFORCEMENT ---

    // Forces W = 0.0f (Treats the vector as a Direction/Normal)
    // Mask 0x08 (binary 1000) tells the hardware: 
    // "Take X, Y, Z from 'reg', take W from the zero vector."
    FORCE_INLINE Vector3D asDirection() const {
        return Vector3D(_mm_blend_ps(reg, _mm_setzero_ps(), 0x08));
    }

    // Forces W = 1.0f (Treats the vector as a Position/Point in space)
    // We blend our register with a vector containing 1.0f in the W lane.
    FORCE_INLINE Vector3D asPoint() const {
        __m128 wOne = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // Setps takes (W, Z, Y, X)
        return Vector3D(_mm_blend_ps(reg, wOne, 0x08));
    }

    // Default constructor (Zero initialization)
    FORCE_INLINE Vector3D() : reg(_mm_setzero_ps()) {}

    // Constructor from floats
    FORCE_INLINE Vector3D(float _x, float _y, float _z, float _w = 0.0f) 
        : reg(_mm_set_ps(_w, _z, _y, _x)) { 
        // Note: _mm_set_ps takes arguments in reverse order (w, z, y, x)
    }

    // Constructor directly from SSE register (Crucial for fast operators)
    FORCE_INLINE Vector3D(__m128 m) : reg(m) {}

    // --- MATHEMATICAL OPERATORS ---
    // By returning by value, the compiler uses Return Value Optimization (RVO).
    // The data never touches the stack; it stays perfectly inside the CPU registers.

    // Addition: result = this + other
    FORCE_INLINE Vector3D operator+(const Vector3D& other) const {
        return Vector3D(_mm_add_ps(reg, other.reg));
    }

    FORCE_INLINE Vector3D operator-(const Vector3D& other) const {
        return Vector3D(_mm_sub_ps(reg, other.reg));
    }

    // Scales the current vector in place
    FORCE_INLINE Vector3D operator*(float scalar) const {
        return Vector3D(_mm_mul_ps(reg, _mm_set1_ps(scalar)));
    }

    // --- DOT & CROSS PRODUCT ---

    // Dot Product: returns (x1*x2 + y1*y2 + z1*z2)
    FORCE_INLINE float dot(const Vector3D& other) const {
        // 0x7F mask: 0111 (read first 3) | 1111 (write to all 4 for safety, or 0001 for just lowest)
        __m128 res = _mm_dp_ps(reg, other.reg, 0x71); 
        return _mm_cvtss_f32(res); // Faster than _mm_store_ss to a stack variable
    }

    // SHUFFLE: Rearranges the (x, y, z) values inside the register, so we can multiply them all at once.
    FORCE_INLINE Vector3D cross(const Vector3D& other) const {
        __m128 tmp0 = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 tmp1 = _mm_shuffle_ps(other.reg, other.reg, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp2 = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp3 = _mm_shuffle_ps(other.reg, other.reg, _MM_SHUFFLE(3, 0, 2, 1));

        return Vector3D(_mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3)));
    }
};

// SSE Accelerated Stack Vector (Use for 99% of general game logic and bulk data processing).
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
        _mm_store_ps(result.data, _mm_add_ps(v1, v2));

        return result;
    }

    // Subtraction: C = A - B
    FORCE_INLINE Vector3DStack operator-(const Vector3DStack& other) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);
        _mm_store_ps(result.data, _mm_sub_ps(v1, v2));

        return result;
    }

    // Scalar Multiplication: B = A * scalar
    FORCE_INLINE Vector3DStack operator*(float scalar) const {
        Vector3DStack result;
        
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar); // Broadcasts scalar to all 4 slots        
        _mm_store_ps(result.data, _mm_mul_ps(v1, s));

        return result;
    }

    // In-place Scalar Multiplication: A *= scalar
    FORCE_INLINE void operator*=(float scalar) {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 s = _mm_set1_ps(scalar);
        _mm_store_ps(this->data, _mm_mul_ps(v1, s)); // Store directly back into itself
    }

    // Dot Product
    FORCE_INLINE float dot(const Vector3DStack& other) const {
        __m128 v1 = _mm_load_ps(this->data);
        __m128 v2 = _mm_load_ps(other.data);

        // ===================================
        // DOT PRODUCT INSTRUCTION (_mm_dp_ps)
        // ===================================
        /*
            - On modern Intel/AMD architectures, _mm_dp_ps is implemented in slow microcode (~9-14 clock cycles).
            - It tries to do too many things (multiply, horizontal add, and mask) simultaneously.
            - Hogs CPU execution ports because the silicilon has to internally decode it into a sequence of multiplies, adds, and bitwise masks.
        */

        // // 0x71 mask: calculates dots for first 3 elements, stores in first element
        // __m128 res = _mm_dp_ps(v1, v2, 0x71);

        // float result;
        // _mm_store_ss(&result, res); 
        // return result;

        // ===================================
        // DOT PRODUCT (MANUAL HORIZONTAL REDUCTION)
        // ===================================
        /*
            - Operates on separate execution ports and can overlap these manual instructions.
            - [_mm_mul_ps] ~4 cycles
            - [_mm_movehl_ps, _mm_shuffle_ps] ~1 cycle
            - [_mm_add_ps, _mm_add_ss] ~3-4 cycles
        */
        // Bypassing the _mm_dp_ps hardware trap using fast manual reduction (3-4 clock cycles, 2x performance improvement in dot product speed)
        __m128 mul = _mm_mul_ps(v1, v2);

        // Since constructor guarantees w = 0.0f, we can safely horizontal sum all 4 lanes
        __m128 shuf = _mm_movehl_ps(mul, mul); 
        mul = _mm_add_ps(mul, shuf);           
        shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1)); 
        mul = _mm_add_ss(mul, shuf);           
        
        return _mm_cvtss_f32(mul);
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

    // --- HOMOGENEOUS COORDINATE ENFORCEMENT ---
    /*
        - Allow addition, cross product, and dot products to generate garbage in the W lane (let the math be dirty). 
        - Clean it at the boundary where w actually matters when multiplying a vector by a matrix (force it to either a point [w =1], or a direction [w = 0] right before the multiplication).
    */

    // Forces W = 0.0f (Treats the vector as a Direction/Normal)
    FORCE_INLINE Vector3DStack asDirection() const {
        Vector3DStack result;
        __m128 reg = _mm_load_ps(this->data);
        
        // Blend in a 0.0f to the W lane (mask 0x08 = 1000 binary)
        reg = _mm_blend_ps(reg, _mm_setzero_ps(), 0x08);
        
        _mm_store_ps(result.data, reg);
        return result;
    }

    // Forces W = 1.0f (Treats the vector as a Position/Point in space)
    FORCE_INLINE Vector3DStack asPoint() const {
        Vector3DStack result;
        __m128 reg = _mm_load_ps(this->data);
        
        // Blend in a 1.0f to the W lane
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
// SCALAR VECTORS (NON-SIMD)
// ==================================================================================

// Pure Scalar Fallback: Used for one-off calculations like the camera and compile time calculations.
class Vector3DScalar {
public:
    float x, y, z, w; // Same 16-byte memory footprint for fairness

    constexpr Vector3DScalar(float x = 0.0f, float y = 0.0f, float z = 0.0f) 
        : x(x), y(y), z(z), w(0.0f) {}

    // Traditional element-by-element addition
    FORCE_INLINE constexpr Vector3DScalar operator+(const Vector3DScalar& other) const {
        return Vector3DScalar(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction: C = A - B
    FORCE_INLINE constexpr Vector3DScalar operator-(const Vector3DScalar& other) const {
        return Vector3DScalar(x - other.x, y - other.y, z - other.z);
    }

    // Scalar Multiplication: B = A * scalar
    FORCE_INLINE constexpr Vector3DScalar operator*(float scalar) const {
        return Vector3DScalar(x * scalar, y * scalar, z * scalar);
    }

    // In-place Scalar Multiplication: A *= scalar (constexpr in C++20 allows modifying member variables)
    FORCE_INLINE constexpr void operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
    }

    // Traditional dot product
    FORCE_INLINE constexpr float dot(const Vector3DScalar& other) const {
        return (x * other.x) + (y * other.y) + (z * other.z);
    }

    // Traditional cross product
    FORCE_INLINE constexpr Vector3DScalar cross(const Vector3DScalar& other) const {
        return Vector3DScalar(
            (y * other.z) - (z * other.y),
            (z * other.x) - (x * other.z),
            (x * other.y) - (y * other.x)
        );
    }
};

// ==================================================================================
// LARGE WORLD COORDINATES (LWC)
// ==================================================================================
/*
    - The GPU only ever sees 32-bit floats and SIMD registers only process 32-bit floats.
    - World Space (64-bit):Entities, Transforms, and the camera track their absolute positions in the universe using doubles.
    - Local Space (32-bit): Before rendering or running local physics (like collision subtraction), you subtract the camera's world position form the object's world position.
    - Result: The camera becomes the center of the universe (0, 0, 0).
    - The object is now a 32-bit float offset relative to the camera, safely within the zone of high-precision floating-point math (i.e., camera-relative rendering).
    - Allows SIMD to process it at maximum speed.
    - This will replace Vector3DStack only for the absolute position property of the entities and the camera.
    - Absolute world positions are tracked in 64-bit space, physics and rendering are done in 32-bit floats relative to the camera.
*/

struct Vector3DWorld {
    // Dedicated 64-bit scalar vector.
    double x, y, z;

    constexpr Vector3DWorld(double x = 0.0, double y = 0.0, double z = 0.0) 
        : x(x), y(y), z(z) {}

    // Standard addition for moving objects in the world
    FORCE_INLINE constexpr Vector3DWorld operator+(const Vector3DWorld& other) const {
        return Vector3DWorld(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction is the most important operator in LWC.
    // It returns the difference between two massive world coordinates.
    FORCE_INLINE constexpr Vector3DWorld operator-(const Vector3DWorld& other) const {
        return Vector3DWorld(x - other.x, y - other.y, z - other.z);
    }

    // --- THE LWC BRIDGE ---
    // Safely casts a 64-bit world difference down to your ultra-fast 32-bit SIMD vector.
    FORCE_INLINE Vector3DStack toFloatVector() const {
        return Vector3DStack(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }
};



// ==================================================================================
// 4. MATRICES & INTERPOLATION
// ==================================================================================
/*
    - Vector3DScalar (NO SSE-Accelerated)
    - Matrix generation requires sequential FPU (Floating Point Unit) math.
    - When dealing with a single entity, packing data into 128 bit SSE registers actually hurts performance.
    - CPU wastes clock cycles shuffling the data from the FPU, into the SSE registers for the cross product and then unpack it again for the Matrix.
    - To solve this, never leave SIMD registers (AAA Engines: a 4x4 matrix and a 3D vector are always 128-bit SIMD registers).
    - Load the camera data into an SSE register, perform all LookAt, Projection and View Matrix math inside SSE, and only extract the data when pushing it to the GPU via uniform buffers.
*/

// ==================================================================================
// SIMD 4x4 MATRIX (COLUMN-MAJOR)
// ==================================================================================
/*
    - A 4x4 matrix should not be a raw array of 16 floats.
    - It should be an array of four 128-bit SIMD registers. 
    - This is how AA engines keep camera math on the silicon.
*/
struct alignas(64) Matrix4x4_SIMD {
    // 4 columns, each taking up exactly one 128-bit register
    __m128 col[4];

    // Creates an Identity Matrix entirely inside the registers
    static FORCE_INLINE Matrix4x4_SIMD Identity() {
        Matrix4x4_SIMD mat;
        mat.col[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f); // { 1, 0, 0, 0 }
        mat.col[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f); // { 0, 1, 0, 0 }
        mat.col[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f); // { 0, 0, 1, 0 }
        mat.col[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // { 0, 0, 0, 1 }
        return mat;
    }

    // --- __MM_TRNASPOSE4_PS ---
    /*
        - Rendering APIs require matrices to be formatted in column-major order.
        - Instead of extracting floats sequentially to flip the rows into columns, SSE has a built in macro to transpose a 4x4 matrix across four registers in a few clock cycles.
    */
    static FORCE_INLINE Matrix4x4_SIMD LookAtLWC_SIMD(const Vector3DWorld& eye, const Vector3DWorld& target, const Vector3D& upVec) {
        
        // 1. Calculate World Difference & Cast to 32-bit SIMD (Vector3D is your SSE wrapper class)
        Vector3DWorld worldDiff = target - eye;
        Vector3D f = Vector3D(static_cast<float>(worldDiff.x), 
                            static_cast<float>(worldDiff.y), 
                            static_cast<float>(worldDiff.z), 
                            0.0f); // Ensure W is 0.0f for directional vectors
        
        // Normalize Forward
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) f = f * (1.0f / std::sqrt(fLenSq));

        // 2. Right Vector (X) - SIMD Cross Product
        Vector3D r = f.cross(upVec);
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) r = r * (1.0f / std::sqrt(rLenSq));

        // 3. Up Vector (Y) - SIMD Cross Product
        Vector3D u = r.cross(f);

        // 4. Negate the Forward vector (Required for Right-Handed Coordinate Systems)
        // Flip the sign bit in hardware without multiplication: XOR with -0.0f
        __m128 negZero = _mm_set1_ps(-0.0f);
        __m128 negF = _mm_xor_ps(f.reg, negZero);

        // 5. Load our rows. 
        // We force the W component of these row vectors to 0.0f, except for the bottom row.
        __m128 row0 = r.reg;     // { Rx, Ry, Rz, 0 }
        __m128 row1 = u.reg;     // { Ux, Uy, Uz, 0 }
        __m128 row2 = negF;      // {-Fx,-Fy,-Fz, 0 }
        __m128 row3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // { 0, 0, 0, 1 }

        // 6. THE MAGIC TRICK: Hardware Transpose
        // This flips our rows into Column-Major format in-place!
        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        // 7. Store the transposed registers directly into the matrix columns.
        Matrix4x4_SIMD mat;
        mat.col[0] = row0;
        mat.col[1] = row1;
        mat.col[2] = row2;
        mat.col[3] = row3;

        // Notice we do NOT calculate translation (-r.dot(eye), etc.). 
        // Because we are using Large World Coordinates (LWC), the camera is ALWAYS at (0,0,0)!
        
        return mat;
    }
};

// --- SIMD MATRIX OPERATORS ---
// Multiplies a SIMD Vector against a SIMD Matrix (i.e., use the broadcast and multiply-add)
FORCE_INLINE Vector3D operator*(const Matrix4x4_SIMD& mat, const Vector3D& v) {
    // Broadcast a single vertex component into all four lanes of a register, and multiply it by the first column, and accumulate.

    // 1. Broadcast vertex X, Y, Z, W into four separate registers
    __m128 vx = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(0, 0, 0, 0)); // {x, x, x, x}
    __m128 vy = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(1, 1, 1, 1)); // {y, y, y, y}
    __m128 vz = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(2, 2, 2, 2)); // {z, z, z, z}
    __m128 vw = _mm_shuffle_ps(v.reg, v.reg, _MM_SHUFFLE(3, 3, 3, 3)); // {w, w, w, w}

    // 2. Multiply each broadcasted component by its corresponding matrix column
    __m128 res = _mm_mul_ps(vx, mat.col[0]);
    
    // 3. Fused Multiply-Add the rest of the columns
    // res = (vy * col1) + res
    res = _mm_fmadd_ps(vy, mat.col[1], res);  // _mm_fmadd_ps: requirees FMA3 instructions, which is part of AVX2, supported by almost all CPUs made after 2013.
    res = _mm_fmadd_ps(vz, mat.col[2], res);
    res = _mm_fmadd_ps(vw, mat.col[3], res);

    return Vector3D(res);
}

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

    // Creates a View Matrix (LookAt)
    static Matrix4 LookAt(const Vector3DScalar& eye, const Vector3DScalar& target, const Vector3DScalar& upVec) {

        // 1. Forward Vector (Z)
        Vector3DScalar f = Vector3DScalar(target.x - eye.x, target.y - eye.y, target.z - eye.z);
        float fLen = std::sqrt(f.dot(f));
        f.x *= (1.0f / fLen); f.y *= (1.0f / fLen); f.z *= (1.0f / fLen);

        // 2. Right Vector (X)
        Vector3DScalar r = f.cross(upVec);
        float rLen = std::sqrt(r.dot(r));
        r.x *= (1.0f / rLen); r.y *= (1.0f / rLen); r.z *= (1.0f / rLen);

        // 3. Up Vector (Y)
        Vector3DScalar u = r.cross(f);

        // 4. Build Column-Major Matrix
        Matrix4 mat = Identity();
        mat.m[0] = r.x;  mat.m[4] = r.y;  mat.m[8] = r.z;
        mat.m[1] = u.x;  mat.m[5] = u.y;  mat.m[9] = u.z;
        mat.m[2] = -f.x; mat.m[6] = -f.y; mat.m[10] = -f.z;
        
        // Translation offsets
        mat.m[12] = -r.dot(eye);
        mat.m[13] = -u.dot(eye);
        mat.m[14] = f.dot(eye);
        return mat;
    }

    // --- LWC CAMERA-RELATIVE RENDERING LOOK-AT ---
    // View Matrix: No longer needs translation with camera-relative rendering, only handles rotation.
    // Notice the inputs are Vector3DWorld (double), but the matrix is float.
    static Matrix4 LookAtLWC(const Vector3DWorld& eye, const Vector3DWorld& target, const Vector3DStack& upVec) {
        
        // 1. Calculate the forward vector in 64-bit space to prevent jitter at massive distances
        Vector3DWorld worldForward = target - eye;
        
        // 2. Cast down to 32-bit float for the math. 
        // Because it's a directional vector (difference), the cast is perfectly safe!
        Vector3DStack f = worldForward.toFloatVector();
        
        // Use your fast SIMD dot product to normalize
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) {
            f *= (1.0f / std::sqrt(fLenSq));
        }

        // 3. Right Vector (X)
        Vector3DStack r = f.cross(upVec);
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) {
            r *= (1.0f / std::sqrt(rLenSq));
        }

        // 4. Up Vector (Y)
        Vector3DStack u = r.cross(f);

        // 5. Build Column-Major Matrix
        Matrix4 mat = Identity();
        mat.m[0] = r.data[0];  mat.m[4] = r.data[1];  mat.m[8] = r.data[2];
        mat.m[1] = u.data[0];  mat.m[5] = u.data[1];  mat.m[9] = u.data[2];
        mat.m[2] = -f.data[0]; mat.m[6] = -f.data[1]; mat.m[10] = -f.data[2];
        
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
        Vector3DStack relativeLocalPos = relativeWorldPos.toFloatVector();

        // 5. Now, you build your Model Matrix using `relativeLocalPos` and send it to the GPU!
        Matrix4 treeModelMatrix = BuildTranslationMatrix(relativeLocalPos);
    */
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

// ==================================================================================
// 5. ENGINE SUBSYSTEMS & PHYSICS
// ==================================================================================
/*
    - SIMD registers cannot be directly sorted.
    - SIMD execution ports are designed for parallel vertical math, not horizontal cross-lane sorting.
    - std::span is used to make the physics math not care if the memory came from a std::vector, a custom memory arena, or the stack, it just chews through the span.
    - std::span enables us to not have to change a single line of code in the physics or sorting math. 
    - If we want to change std::vector to a custom allocator, you create std::span, over the new memory and feed it into the pipeline.
    - VFX: When a player's car crashes into a crate, it spawns 10,000 particles.
    - Don't need the 64-bit world coordinates. Spawned directly into the camera relative buffers (32-bit buffers), immediately Morton sorted, and processed in bulk using radix sort and linear collision scan.
    
    1. Use getMortonCode_SIMD to calculate grid cells in bulk.
    2. Flush those results out of the SIMD registers into a flat, scalar array of 64-bit [MortonCode, OriginalIndex] pairs.
    3. Run an O(n) integer radix sort on that flat array.
    4. Use the sorted indices to physically rewrite the positions and velocities arrays into perfect z-order curve alignment.
*/
#if ENGINE_HAS_CXX26_SIMD

    // --- SPATIAL HASHING BUFFERS ---
    // 64-bit struct: 32 bits for the Morton Code, 32 bits for the physical particle index.
    struct ParticleSortKey {
        uint32_t mortonCode;
        uint32_t particleIndex;
    };

    // --- DATA-ORIENTED PARTICLE SYSTEM (AoSoA PIPELINE) ---
    // Pure Data Container (No Logic)
    struct ParticleMemoryBlock {
        // Every element in this vector represents a BATCH of particles 
        // (4 on ARM, 8 on AVX2, 16 on AVX-512).
        // Because SIMDVector3D is aligned to NATIVE_SIMD_BATCH_ALIGN, std::vector
        // will perfectly pack these into sequential CPU cache lines.
        std::vector<SIMDVector3D> positions;
        std::vector<SIMDVector3D> velocities;
        
        std::vector<ParticleSortKey> sortKeys;
        std::vector<ParticleSortKey> sortKeysBuffer;  // Required for Radix Sort "Ping-Ponging"

        // We track the exact number of active particles, not just the batch count.
        size_t activeParticleCount = 0;

        // Pre-allocate memory to avoid heap fragmentation
        void Initialize(size_t maxParticles) {
            // Divide by the hardware's native batch size, rounding up.
            size_t batchCount = (maxParticles + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

            positions.resize(batchCount);
            velocities.resize(batchCount);
            
            // Keys are scalar, so we allocate exactly maxParticles
            sortKeys.resize(maxParticles);
            sortKeysBuffer.resize(maxParticles);
        }
    };

    // --- 1. PHYSICS UPDATE ---
    // This loop will chew through millions of particles per millisecond.
    FORCE_INLINE void UpdateParticles(std::span<SIMDVector3D> positions, 
                                    std::span<SIMDVector3D> velocities, 
                                    size_t activeCount, 
                                    float deltaTime) {
        // 1. Broadcast the scalar delta time into a hardware SIMD register ONCE.                                
        NativeFloatSIMDBatch dtBatch = deltaTime; 

        // 2. Iterate over the batches (NOT individual particles)
        size_t activeBatches = (activeCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

        // We can iterate directly over the span size up to the active batch count
        for (size_t i = 0; i < activeBatches; ++i) {
            // 3. Load velocities into registers
            NativeFloatSIMDBatch velX = velocities[i].x;
            NativeFloatSIMDBatch velY = velocities[i].y;
            NativeFloatSIMDBatch velZ = velocities[i].z;

            // 4. Calculate movement: Velocity * DeltaTime
            velX *= dtBatch;
            velY *= dtBatch;
            velZ *= dtBatch;

            // 5. Apply Fused Multiply-Add (Position = Position + (Velocity * dt))
            // Because you built `add()` to accept NativeFloatSIMDBatch, this automatically 
            // maps to hardware vector addition.
            positions[i].add(velX, velY, velZ);

            // Notice there are NO if-statements, NO branching, and NO function overhead.
            // The hardware prefetcher will detect this linear memory access pattern 
            // immediately and stream the L1 cache ahead of the CPU's execution ports.
        }
    }

    // --- 2. GRID QUANTIZATION ---
    // Turns a floating-point world position into a Morton code by mapping it to a strictly positive integer grid [0, 1023].
    FORCE_INLINE void GenerateMortonKeys(std::span<const SIMDVector3D> positions, 
                                        std::span<ParticleSortKey> outKeys, 
                                        size_t activeCount) {
                                            
        // 1. Grid Parameters
        // Offset ensures all particles are pushed into positive coordinate space.
        NativeFloatSIMDBatch offset = 50000.0f; 

        // Grid Cell Size = 4.0 units. Multiplying by 0.25 is drastically faster than dividing by 4.0.
        NativeFloatSIMDBatch invCellSize = 0.25f; 
        size_t activeBatches = (activeCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

        for (size_t i = 0; i < activeBatches; ++i) {
            // 2. Shift and Scale (FMA)
            NativeFloatSIMDBatch gridX_f = (positions[i].x + offset) * invCellSize;
            NativeFloatSIMDBatch gridY_f = (positions[i].y + offset) * invCellSize;
            NativeFloatSIMDBatch gridZ_f = (positions[i].z + offset) * invCellSize;

            // 3. Cast SIMD Floats to SIMD Integers
            // C++26 <simd> syntax for parallel type conversion
            NativeUIntBatch gridX = std::simd_cast<uint32_t>(gridX_f);
            NativeUIntBatch gridY = std::simd_cast<uint32_t>(gridY_f);
            NativeUIntBatch gridZ = std::simd_cast<uint32_t>(gridZ_f);

            // 4. Generate Morton Codes across all lanes simultaneously (BMI2 / LUT)
            NativeUIntBatch mortonBatch = getMortonCode_SIMD(gridX, gridY, gridZ);

            // 5. The Bridge: Flush SIMD register into a temporary aligned array
            alignas(NATIVE_SIMD_BATCH_ALIGN) uint32_t tempMortons[NATIVE_BATCH_SIZE];
            mortonBatch.copy_to(tempMortons, std::element_aligned);

            // 6. Write to our flat scalar sorting array
            for (size_t lane = 0; lane < NATIVE_BATCH_SIZE; ++lane) {
                uint32_t absoluteIdx = static_cast<uint32_t>(i * NATIVE_BATCH_SIZE + lane);
                if (absoluteIdx < activeCount) {
                    outKeys[absoluteIdx] = { tempMortons[lane], absoluteIdx };
                }
            }
        }
    }

    // --- 3. 8-bit RADIX SORT ---
    // It looks at the binary bits of the Morton code and drops them into buckets. 
    // By processing 8-bits at a time, we only need 4 passes to perfectly sort 32-bit integers.
    FORCE_INLINE void RadixSortKeys(std::span<ParticleSortKey> keys, 
                                    std::span<ParticleSortKey> buffer) {
                                        
        ParticleSortKey* src = keys.data();
        ParticleSortKey* dst = buffer.data();
        size_t count = keys.size();

        // 4 passes: 0, 8, 16, 24 (to cover all 32 bits of the Morton Code)
        for (int shift = 0; shift < 32; shift += 8) {
            uint32_t counts[256] = {0}; // 8 bits = 256 possible buckets

            // 1. HISTOGRAM PASS: Count how many particles fall into each bucket
            for (size_t i = 0; i < count; ++i) {
                uint8_t bucket = (src[i].mortonCode >> shift) & 0xFF;
                counts[bucket]++;
            }

            // 2. PREFIX SUM PASS: Calculate the starting memory address for each bucket
            uint32_t offsets[256];
            offsets[0] = 0;
            for (int i = 1; i < 256; ++i) {
                offsets[i] = offsets[i - 1] + counts[i - 1];
            }

            // 3. SCATTER PASS: Move the keys to their sorted locations
            for (size_t i = 0; i < count; ++i) {
                uint8_t bucket = (src[i].mortonCode >> shift) & 0xFF;
                dst[offsets[bucket]++] = src[i];
            }

            // 4. PING-PONG: Swap source and destination buffers for the next pass
            std::swap(src, dst);
        }

        // If the final sorted data ended up in the buffer, copy it back to the main array.
        if (src != keys.data()) {
            std::copy(buffer.begin(), buffer.end(), keys.begin());
        }
    }

    // --- 4. REORDER MEMORY ---
    // Permutes the AOSOA arrays to match the sorted keys.
    FORCE_INLINE void ReorderParticleData(std::span<const SIMDVector3D> srcPos, 
                                        std::span<const SIMDVector3D> srcVel,
                                        std::span<SIMDVector3D> dstPos,
                                        std::span<SIMDVector3D> dstVel,
                                        std::span<const ParticleSortKey> sortedKeys) {
                                            
        size_t activeCount = sortedKeys.size();

        // Map the sorted data into perfectly contiguous memory
        for (size_t sortedIdx = 0; sortedIdx < activeCount; ++sortedIdx) {

            // Where did this particle come from?
            uint32_t originalIdx = sortedKeys[sortedIdx].particleIndex;

            // Calculate exact Batch and Lane indices
            size_t oldBatch = originalIdx / NATIVE_BATCH_SIZE;
            size_t oldLane  = originalIdx % NATIVE_BATCH_SIZE;
            size_t newBatch = sortedIdx / NATIVE_BATCH_SIZE;
            size_t newLane  = sortedIdx % NATIVE_BATCH_SIZE;

            // Move the Position data
            dstPos[newBatch].x[newLane] = srcPos[oldBatch].x[oldLane];
            dstPos[newBatch].y[newLane] = srcPos[oldBatch].y[oldLane];
            dstPos[newBatch].z[newLane] = srcPos[oldBatch].z[oldLane];

            // Move the Velocity data
            dstVel[newBatch].x[newLane] = srcVel[oldBatch].x[oldLane];
            dstVel[newBatch].y[newLane] = srcVel[oldBatch].y[oldLane];
            dstVel[newBatch].z[newLane] = srcVel[oldBatch].z[oldLane];
        }
    }

    // Generates a constant SIMD batch representing the layout lanes (e.g., {0, 1, 2, 3...})
    // Ensures we only check particles "forward" in the array, so they don't collide with themsleves or apply collision forces twice.
    FORCE_INLINE NativeUIntBatch GetLaneIndices() {
        alignas(NATIVE_SIMD_BATCH_ALIGN) uint32_t indices[NATIVE_BATCH_SIZE];
        for (uint32_t i = 0; i < NATIVE_BATCH_SIZE; ++i) {
            indices[i] = i;
        }
        
        NativeUIntBatch batch;
        batch.copy_from(indices, std::element_aligned);
        return batch;
    }

    // --- 5. LINEAR COLLISION SCAN ---
    // Extracts a single "reference particle", broadcasts it across a full 128/256/512-bit register, and tests it against entire batches of its neighbors simultaneously.
    FORCE_INLINE void ResolveCollisions(std::span<SIMDVector3D> positions, 
                                        size_t activeCount, 
                                        float particleRadius) {
        
        float diameter = particleRadius * 2.0f;
        NativeFloatSIMDBatch diameterSq = diameter * diameter;
        NativeFloatSIMDBatch epsilon = 1e-8f;

        // Cache the lane sequence {0, 1, 2, 3...}
        NativeUIntBatch laneIndices = GetLaneIndices();

        size_t activeBatches = (activeCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

        // [THE SCAN WINDOW]
        // Because the array is sorted by spatial proximity (Morton Codes), 
        // we only need to look ahead a few batches to find all physical neighbors.
        // Adjust this based on your CELL_SIZE and particle density (2 to 4 is usually plenty).
        const size_t SCAN_WINDOW = 2; 

        // Outer Loop: Iterate through every particle linearly
        for (size_t i = 0; i < activeCount; ++i) {
            
            // 1. Locate the Reference Particle
            size_t batchI = i / NATIVE_BATCH_SIZE;
            size_t laneI  = i % NATIVE_BATCH_SIZE;

            // 2. Broadcast the scalar (X, Y, Z) into full SIMD registers
            // E.g., refX becomes { X, X, X, X, X, X, X, X }
            NativeFloatSIMDBatch refX = positions[batchI].x[laneI];
            NativeFloatSIMDBatch refY = positions[batchI].y[laneI];
            NativeFloatSIMDBatch refZ = positions[batchI].z[laneI];

            // We accumulate the push-back forces for the reference particle locally
            float moveX = 0.0f, moveY = 0.0f, moveZ = 0.0f;

            // 3. Inner Loop: Scan Forward through neighbor batches
            size_t endBatch = std::min(batchI + SCAN_WINDOW + 1, activeBatches);

            for (size_t batchJ = batchI; batchJ < endBatch; ++batchJ) {
                
                // --- SIMD DISTANCE CALCULATION ---
                NativeFloatSIMDBatch dx = positions[batchJ].x - refX;
                NativeFloatSIMDBatch dy = positions[batchJ].y - refY;
                NativeFloatSIMDBatch dz = positions[batchJ].z - refZ;

                // FMA Distance Squared
                NativeFloatSIMDBatch distSq = dx * dx;
                distSq = std::fma(dy, dy, distSq);
                distSq = std::fma(dz, dz, distSq);

                // --- THE MASK GENERATOR ---
                // 1. Are they touching? (distSq < diameterSq)
                // 2. Prevent division by zero (distSq > epsilon)
                auto spatialMask = (distSq < diameterSq) && (distSq > epsilon);

                // 1. Calculate the scalar offset and strictly cast it to uint32_t
                uint32_t batchOffset = static_cast<uint32_t>(batchJ * NATIVE_BATCH_SIZE);

                // 3. Prevent double-resolving and self-resolving! Calculate the absolute index of every particle in Batch J
                // Broadcast the scalar into a SIMD batch
                NativeUIntBatch absoluteJ = NativeUIntBatch(batchOffset) + laneIndices;
                
                // Only apply physics if the neighbor's index is strictly greater than i
                auto validCollisionMask = spatialMask && (absoluteJ > static_cast<uint32_t>(i));

                // --- SIMD PENETRATION RESOLUTION ---
                // If validCollisionMask is false, we set distSq to diameterSq so 
                // the penetration depth becomes exactly 0.0f (preventing NaN math).
                std::simd::where(!validCollisionMask, distSq) = diameterSq;

                // Math: penetration = diameter - sqrt(distSq)
                NativeFloatSIMDBatch actualDist = std::sqrt(distSq);
                NativeFloatSIMDBatch penetration = diameter - actualDist;

                // Math: pushFactor = (penetration * 0.5f) / actualDist
                // We push each particle 50% of the way out of the collision.
                NativeFloatSIMDBatch pushFactor = (penetration * 0.5f) / actualDist;

                // Calculate the exact positional offset
                NativeFloatSIMDBatch pushX = dx * pushFactor;
                NativeFloatSIMDBatch pushY = dy * pushFactor;
                NativeFloatSIMDBatch pushZ = dz * pushFactor;

                // --- APPLY FORCES (NEWTON'S THIRD LAW) ---
                
                // 1. Push Batch J away (Masked addition to prevent moving non-colliding lanes)
                std::simd::where(validCollisionMask, positions[batchJ].x) += pushX;
                std::simd::where(validCollisionMask, positions[batchJ].y) += pushY;
                std::simd::where(validCollisionMask, positions[batchJ].z) += pushZ;

                // 2. Accumulate the opposite push for our Reference Particle
                // We use the portable C++26 reduction (summing up the valid lanes)
                // std::reduce automatically ignores the lanes where validCollisionMask was false
                // because pushX/Y/Z are exactly 0.0f in those lanes due to our penetration math.
                moveX -= std::reduce(pushX);
                moveY -= std::reduce(pushY);
                moveZ -= std::reduce(pushZ);
            }

            // 4. Finally, apply the accumulated movement to the Reference Particle
            positions[batchI].x[laneI] += moveX;
            positions[batchI].y[laneI] += moveY;
            positions[batchI].z[laneI] += moveZ;
        }
    }

    // Grab the raw memory from std::vector, cast it into a std::span representing only the active particles, and pass it into pure functions.
    void EngineTick(ParticleMemoryBlock& memory, float deltaTime) {
        // 1. Create Views (Spans) for the active data. 
        // This costs zero CPU cycles; it's just pointer arithmetic.
        size_t activeBatches = (memory.activeParticleCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;
        
        std::span<SIMDVector3D> posSpan(memory.positions.data(), activeBatches);
        std::span<SIMDVector3D> velSpan(memory.velocities.data(), activeBatches);
        
        std::span<ParticleSortKey> keySpan(memory.sortKeys.data(), memory.activeParticleCount);
        std::span<ParticleSortKey> bufferSpan(memory.sortKeysBuffer.data(), memory.activeParticleCount);

        // 2. Execute the Pipeline
        UpdateParticles(posSpan, velSpan, memory.activeParticleCount, deltaTime);
        GenerateMortonKeys(posSpan, keySpan, memory.activeParticleCount);
        RadixSortKeys(keySpan, bufferSpan);

        // 3. For the reorder, we need a temporary buffer. 
        // In a real engine, you'd pull this from a frame-allocator or memory ring buffer.
        std::vector<SIMDVector3D> tempPos(activeBatches);
        std::vector<SIMDVector3D> tempVel(activeBatches);
        
        ReorderParticleData(posSpan, velSpan, tempPos, tempVel, keySpan);

        // 4. Commit the sorted data back to main memory
        std::copy(tempPos.begin(), tempPos.end(), memory.positions.begin());
        std::copy(tempVel.begin(), tempVel.end(), memory.velocities.begin());

        // 5. NOW resolve collisions on the perfectly sorted posSpan!
        ResolveCollisions(posSpan, memory.activeParticleCount, 2.0f);
    }
#endif
