// ==========================================================================
// 1. THE GLOBAL MODULE FRAGMENT
// Must contain ALL legacy C-headers and macros. Modules cannot import these.
// ==========================================================================
module;


// #pragma once

#include <array>
#include <cstdint>

// Only include Intel intrinsics if compiling on an x86_64 architecture
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).
#endif

// --- COMPILER MACROS ---
#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

// ==========================================================================
// 2. THE MODULE DECLARATION & C++26 IMPORTS
// ==========================================================================
export module Engine.Physics.MortonCode;

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

// ====================================================
// SPATIAL HASHING (MORTON CODES / Z-ORDER CURVE)
// ====================================================

/* APROACH #3 (BEST SOLUTION): [Z-Order Curve (Morton Encoding)]
    - Interleaves the binary bits of X, Y, Z
    - If two particles are physically close in 3D space, Morton codes guarantees they are physically close in 1D memory array.
    - The memory addresses are woven together like a zipper to form [x1 y1 z1 x2 y2 z2 ...]
    - Hashing with prime numbers destroys cache locality because it scatters data in random memory addresses resulting in undetectable patterns, and starved registers waiting hundreds of cycles waiting on main memory. 
*/

// --- C++26 COMPILE-TIME MORTON LUT GENERATION ---
consteval std::array<uint32_t, 1024> GenerateMortonTable(uint32_t shift) {
    // This function executes entirely during compilation (i.e., consteval).
    std::array<uint32_t, 1024> table{};

    // We use "masks" to define where the bits should land.
    for (uint32_t i = 0; i < 1024; ++i) {
        uint32_t v = i;

        // Bit shift version uses standard ALU instructions that can be used on every CPU Architecture.
        v = (v | (v << 16)) & 0x030000FF;  
        v = (v | (v <<  8)) & 0x0300F00F;
        v = (v | (v <<  4)) & 0x030C30C3;
        v = (v | (v <<  2)) & 0x09249249;

        // X bits (shift = 0) go to slots 0, 3, 6,  9...
        // Y bits (shift = 1) go to slots 1, 4, 7, 10...
        // Z bits (shift = 2) go to slots 2, 5, 8, 11...

        // Interleaves the bits of X, Y, and Z to preserve 3D spatial cache locality (3D array flattening), optimizes L1/L2 cache
        table[i] = v << shift;
    }
    return table;
}

// ===================================================
// --- PHYSICAL COORDINATE GRID LOOKUP TABLE (LUT) ---
// ===================================================
/*
    - 12KB of data embedded natively into the Read-Only Data (.rodata) segment of the compiled binary.
    - Zero heap allocations. Zero runtime initialization, tightly packed in memory for hardware prefetcher to stream it into L1 cache instantly.
    - [1024 entries * 4 bytes] = (4KB) each, where (3 * (4KB)) = (12KB) total.
*/

// 12KB arrays for X, Y, Z that will live permanently and easily fit in the 32KB L1 CPU Data Cache. 'inline' prevents Multiple Definition linker errors in header files.
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableX = GenerateMortonTable(0); // Physical Grid Coordinate X, Global Memory
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableY = GenerateMortonTable(1); // Physical Grid Coordinate Y, Global Memory
alignas(64) inline constexpr std::array<uint32_t, 1024> g_MortonTableZ = GenerateMortonTable(2); // Physical Grid Coordinate Z, Global Memory

// --- RUN-TIME MORTON LUT GENERATION ---
/*
void InitMortonLUT() {
    auto expandBits = [](uint32_t v) -> uint32_t {
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v <<  8)) & 0x0300F00F;
        v = (v | (v <<  4)) & 0x030C30C3;
        v = (v | (v <<  2)) & 0x09249249;
        return v;
    };

    
    for (uint32_t i = 0; i < 1024; ++i) {
        uint32_t expanded = expandBits(i); 
        g_MortonTableX[i] = expanded;        // Bits shifted for X (Standard)
        g_MortonTableY[i] = expanded << 1;   // Bits shifted for Y
        g_MortonTableZ[i] = expanded << 2;   // Bits shifted for Z
    }
}
*/

// Blindingly fast scalar fallback is evaluated entirely at compile-time (i.e., constexpr) for static level geometry.
FORCE_INLINE constexpr uint32_t getMortonCodeLUT(uint32_t x, uint32_t y, uint32_t z) {
    // x bits are located in slots 0, 3, 6, 9...
    // y bits are located in slots 1, 4, 7, 10...
    // z bits are located in slots 2, 5, 8, 11...

    // 1. Bitwise mask to strictly enforce the [0, 1023] limit (0x3FF is 1023 in hex), executes in a fraction of a cycle and prevents catastrophic engine crashes.
    x &= 0x3FF; 
    y &= 0x3FF; 
    z &= 0x3FF;

    // Tells the compiler "I guarantee these are under 1024, do not generate bounds-checking assembly."
    #if defined(__cpp_constexpr) && __cplusplus >= 202302L
        // C++23 [[assume]] attribute.
        [[assume(x < 1024 && y < 1024 && z < 1024)]];
    #elif defined(_MSC_VER)
        // MSVC specific assume
        __assume(x < 1024 && y < 1024 && z < 1024);
    #elif defined(__clang__) || defined(__GNUC__)
        // Clang/GCC specific assume (using __builtin_unreachable)
        if (!(x < 1024 && y < 1024 && z < 1024)) __builtin_unreachable();
    #endif

    // (3) Memory Fetches and (2) Bitwise ORs!
    return g_MortonTableX[x] | g_MortonTableY[y] | g_MortonTableZ[z];
}

// =======================================================
// --- HARDWARE MORTON ENCODING (BMI2 / LUT FALLBACK)  ---
// =======================================================

template <bool IsLegacy>
FORCE_INLINE uint32_t getMortonCode(uint32_t x, uint32_t y, uint32_t z) {
    // --- x86_64 INTEL/AMD ARCHITECTURE (PC / PS5 / Xbox Series X/S) ---
    #if defined(__x86_64__) || defined(_M_X64)
        // C++17/26: 'if constexpr' ensures the false path is entirely deleted from the compiled binary.
        if constexpr (!IsLegacy) {
            // Intel Haswell+ / AMD Zen3+ native hardware support (~3 clock cycles); _pdep_u32 instruction (BMI2) directly implemented (or wired) into its silicon (or hardware).
            uint32_t mx = _pdep_u32(x, 0x09249249); 
            uint32_t my = _pdep_u32(y, 0x12492492);
            uint32_t mz = _pdep_u32(z, 0x24924924);
            
            // Return the result after performing bit manipulation instruction 2
            return mx | my | mz;
        } 
        else {
            // Legacy CPU fallback for x86 processors without BMI2 support (~4 clock cycles).
            return getMortonCodeLUT(x, y, z);
        }
    // --- ARM64 ARCHITECTURE (Apple Silicon / Nintendo Switch 2) ---
    #else
        // ARM CPUs do not have a hardware equivalent to Intel's _pdep_u32. Ignore the 'IsLegacy' boolean and seamlessly route execution to the ultra-fast L1 cache table.
        return getMortonCodeLUT(x, y, z);
    #endif    
}

/* 
    - Global memory space being used in C++ means that the compiler cannot guarantee that another thread won't change it mid-execution.
    - This burns roughly 10 million clock cycles per frame just reading a boolean that never changes during gameplay [100,000 particles, 2.7 million calls per frame, ~4 clock cycles].

      struct EngineSettings {
          // We need to check if the boolean has changed every frame.
          bool isLegacyCPU = false; 
      };
      extern EngineSettings g_EngineSettings;
      
    - Solution is to move the boolean global memory to const stack variable.
    
      // By passing IsLegacy as a template parameter, the compiler resolves the branch at compile-time (i.e., compiler generates two completely separate, branchless versions of the function at compile-time saving millions of evaluations per frame).
      template <bool IsLegacy>

    - Cache it into a local, const stack variable at the moment the thread wakes up and pass them explicitly.
    - Passing a const variable into an inline function, the compiler will optimize it and make it a pure branchless execution path.

      // C++20: Templated Lambda Dispatcher. 
      auto processMortonCode = [&]<bool IsLegacy>() {
        hash = getMortonCode<IsLegacy>(gridX, gridY, gridZ) & HASH_MASK;
      };
      
*/
