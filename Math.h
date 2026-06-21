#pragma once

#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).
#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <print>       // Formatting
#include <cstdint>
#include <array>
#include <mdspan>
#include <cstddef>

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
*/

#if ENGINE_HAS_CXX26_SIMD
    // --- 1. THE C++26 MATH LAYER (Portable SIMD) --- 
    // Use the official C++26 P1928 syntax based on the silicon it detects at compile time.
    using NativeFloatSIMD = std::simd<float, std::simd_abi::native<float>>;

    // Use this constant to dynamically align your memory allocators and structs!
    // Ask the C++26 standard exactly how many bytes the current hardware needs
    constexpr std::size_t NATIVE_SIMD_ALIGN = alignof(NativeFloatSIMD);

    // Dynamic Alignment Wrapper:
    // If compiling for AVX-512, this guarantees 64-byte alignment. 
    // If compiling for ARM NEON, it guarantees 16-byte alignment.
    struct alignas(NATIVE_SIMD_ALIGN) SIMDVectorP {
        // [NativeFloatSIMD]: Instead of manual __m256 or __m512 loads, you use a template that automatically picks the widest register the hardware supports.
        NativeFloatSIMD x, y, z;

        // Standard Addition
        FORCE_INLINE void add(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            x += bx; // C++26 SIMD supports standard operators!
            y += by;
            z += bz;
        }

        // Standard Subtraction
        FORCE_INLINE void sub(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            x -= bx; 
            y -= by; 
            z -= bz;
        }

        // Scalar Multiplication
        FORCE_INLINE void mul(const NativeFloatSIMD& scalar) {
            x *= scalar;
            y *= scalar;
            z *= scalar;
        }

        // Dot Product with FMA (Fused Multiply-Add)
        // C++26 automatically fuses (a * b + c) into a single clock cycle if compiler flags allow it, or you can explicitly use std::fma overloaded for simd.
        FORCE_INLINE NativeFloatSIMD dot_fma(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) const {
            NativeFloatSIMD res = x * bx;
            res = std::fma(y, by, res);
            res = std::fma(z, bz, res);
            return res;
        }

        // Standard Dot Product
        FORCE_INLINE NativeFloatSIMD dot(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) const {
            return (x * bx) + (y * by) + (z * bz);
        }

        // SOA Cross Product
        FORCE_INLINE void cross(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            // Explicitly using std::fma to guarantee hardware Fused Multiply-Subtract (FMS) 
            // Example: (y * bz) - (z * by) -> fma(y, bz, -(z * by))
            NativeFloatSIMD rx = std::fma(y, bz, -(z * by));
            NativeFloatSIMD ry = std::fma(z, bx, -(x * bz));
            NativeFloatSIMD rz = std::fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }

        // Magnitude Squared
        FORCE_INLINE NativeFloatSIMD length_sq() const {
            // return (x * x) + (y * y) + (z * z);

            // Nudging compiler to use FMA
            NativeFloatSIMD sq = x * x;
            sq = std::fma(y, y, sq);
            sq = std::fma(z, z, sq);
            return sq;
        }

        // Magnitude
        FORCE_INLINE NativeFloatSIMD length() const {
            return std::sqrt(length_sq()); // std::sqrt is overloaded for simd types!
        }

        // --- C++26 PORTABLE OPMASK LOGIC ---
        FORCE_INLINE void normalize() {
            NativeFloatSIMD sqLen = length_sq();
            NativeFloatSIMD epsilon = 1e-8f;
            
            // 1. Create the hardware mask
            auto validMask = sqLen > epsilon;

            // 2. Prevent NaN/Inf generation by patching invalid lengths to 1.0f BEFORE division.
            // If sqLen is 0, we temporarily pretend it is 1.0f so division succeeds gracefully.
            NativeFloatSIMD safeSqLen = sqLen;
            std::simd::where(!validMask, safeSqLen) = 1.0f; 

            // 3. Fast-math will translate this to a hardware rsqrt instruction.
            NativeFloatSIMD invLen = 1.0f / std::sqrt(safeSqLen);

            // 4. Apply math
            x *= invLen;
            y *= invLen;
            z *= invLen;

            // 5. Zero-mask the invalid lanes
            std::simd::where(!validMask, x) = 0.0f;
            std::simd::where(!validMask, y) = 0.0f;
            std::simd::where(!validMask, z) = 0.0f;
        }
    };
#endif

// ================================================================================
// VECTOR3D STRUCTS
// ================================================================================

struct alignas(16) Vector3DStackAligned {
    float x, y, z, w; // Total 16 bytes
};

// This represents 4 vectors at once
struct Vector3D_SOA_Batch {
    __m128 x; // [v1.x, v2.x, v3.x, v4.x]
    __m128 y; // [v1.y, v2.y, v3.y, v4.y]
    __m128 z; // [v1.z, v2.z, v3.z, v4.z]
};

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
        float data[4];
    };

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

    // ==========================================
    // Vector3DScalar (NO SSE-Accelerated)
    // ==========================================
    /*
        - Matrix generation requires sequential FPU (Floating Point Unit) math.
        - When dealing with a single entity, packing data into 128 bit SSE registers actually hurts performance.
        - CPU wastes clock cycles shuffling the data from the FPU, into the SSE registers for the cross product and then unpack it again for the Matrix.
        - Best to use Vector3DScalar instead of Vector3DStack for camera.
    */

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

    // --- LWC CAMERA-RELATIVE LOOK-AT ---
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
