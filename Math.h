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
#include <bit>         // Required for std::bit_cast

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
// SSE Accelerated Vectors 
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
    __m128 reg;

    // Default constructor (Zero initialization)
    FORCE_INLINE Vector3D() : reg(_mm_setzero_ps()) {}

    // Constructor from floats
    FORCE_INLINE Vector3D(float _x, float _y, float _z, float _w = 0.0f) 
        : reg(_mm_set_ps(_w, _z, _y, _x)) { 
        // Note: _mm_set_ps takes arguments in reverse order (w, z, y, x)
    }

    // Constructor directly from SSE register (Crucial for fast operators)
    FORCE_INLINE Vector3D(__m128 m) : reg(m) {}

    // ======================================================================
    // 1. HARDWARE GETTERS (Zero Memory Access)
    // ======================================================================
    // Extracts the float directly from the XMM register. 
    // This API change means you will call `v.x()` instead of `v.x`.
    
    // X is in the lowest 32 bits, so we just convert scalar.
    FORCE_INLINE float x() const { 
        return _mm_cvtss_f32(reg); 
    }
    
    // Y, Z, and W require a 1-cycle shuffle to move them to the lowest 32 bits before extraction.
    FORCE_INLINE float y() const { 
        return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(1, 1, 1, 1))); 
    }
    
    FORCE_INLINE float z() const { 
        return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(2, 2, 2, 2))); 
    }
    
    FORCE_INLINE float w() const { 
        return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3))); 
    }

    // ======================================================================
    // 2. HARDWARE SETTERS (SSE4.1)
    // ======================================================================
    // Allows mutation without spilling the register to the stack.
    
    FORCE_INLINE void setX(float val) {
        // _mm_move_ss replaces the lowest 32-bits (X) and keeps the high 96-bits intact.
        reg = _mm_move_ss(reg, _mm_set_ss(val));
    }

    FORCE_INLINE void setY(float val) {
        // _mm_insert_ps takes the source value and inserts it into a specific lane.
        // 0x10 = Source Index 0, Destination Index 1 (Y)
        reg = _mm_insert_ps(reg, _mm_set_ss(val), 0x10);
    }

    FORCE_INLINE void setZ(float val) {
        // 0x20 = Source Index 0, Destination Index 2 (Z)
        reg = _mm_insert_ps(reg, _mm_set_ss(val), 0x20);
    }

    FORCE_INLINE void setW(float val) {
        // 0x30 = Source Index 0, Destination Index 3 (W)
        reg = _mm_insert_ps(reg, _mm_set_ss(val), 0x30);
    }

    // ======================================================================
    // 3. C++20 ZERO-COST MEMORY BRIDGE
    // ======================================================================
    // If you need to interface with OpenGL/Vulkan APIs or loop through the 
    // vector like an array, std::bit_cast is perfectly standard compliant.
    // It guarantees the exact same zero-overhead assembly as the old union hack.
    
    FORCE_INLINE std::array<float, 4> asArray() const {
        return std::bit_cast<std::array<float, 4>>(reg);
    }

    // --- AXIS INDEXING ---
    // Safely extracts X (0), Y (1), Z (2), or W (3) dynamically without breaking strict aliasing.
    FORCE_INLINE float operator[](int axis) const {
        // We use std::bit_cast (C++20) to treat the register as a safe array 
        // entirely on the stack, allowing dynamic indexing without UB.
        auto arr = std::bit_cast<std::array<float, 4>>(reg);
        return arr[axis];
    }

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
// LARGE WORLD COORDINATES (LWC)
// ==================================================================================
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

    // --- PURE SIMD VIEW MATRIX (NO LOAD-HIT-STORE) ---
    // Takes native SSE vectors and keeps all math strictly on the silicon.
    static FORCE_INLINE Matrix4x4_SIMD LookAt_SIMD(const Vector3D& eye, const Vector3D& target, const Vector3D& upVec) {
        
        // 1. Forward Vector (Z)
        Vector3D f = target - eye;
        f = f.asDirection(); // Force W=0.0f
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) f = f * (1.0f / std::sqrt(fLenSq));

        // 2. Right Vector (X)
        Vector3D r = f.cross(upVec).asDirection();
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) r = r * (1.0f / std::sqrt(rLenSq));

        // 3. Up Vector (Y)
        Vector3D u = r.cross(f).asDirection();

        // 4. Negate the Forward vector (Required for Right-Handed Coordinate Systems like OpenGL)
        // Flip the sign bit in hardware without multiplication: XOR with -0.0f
        __m128 negZero = _mm_set1_ps(-0.0f);
        __m128 negF = _mm_xor_ps(f.reg, negZero);

        // 5. Calculate Translation Vector
        // Standard View Matrix translation is: [-dot(R, eye), -dot(U, eye), dot(F, eye)]
        float tx = -r.dot(eye);
        float ty = -u.dot(eye);
        float tz = f.dot(eye); 
        
        // Pack into a single register for the 4th column. (_mm_set_ps takes args in reverse order: w, z, y, x)
        __m128 translation = _mm_set_ps(1.0f, tz, ty, tx);

        // 6. Setup rows for Hardware Transposition
        __m128 row0 = r.reg;                              // { Rx, Ry, Rz, 0 }
        __m128 row1 = u.reg;                              // { Ux, Uy, Uz, 0 }
        __m128 row2 = negF;                               // {-Fx,-Fy,-Fz, 0 }
        __m128 row3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // { 0,  0,  0,  1 }

        // 7. THE MAGIC TRICK: Hardware Transpose
        // Flips the 3x3 rotation axes into Column-Major format instantly.
        _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

        // 8. Store the columns, overwriting the transposed 4th column with our calculated translation.
        Matrix4x4_SIMD mat;
        mat.col[0] = row0;
        mat.col[1] = row1;
        mat.col[2] = row2;
        mat.col[3] = translation; 

        return mat;
    }

    // ==================================================================================
    // MATRICES & INTERPOLATION
    // ==================================================================================
    /*
        - Vector3DScalar (NO SSE-Accelerated)
        - Matrix generation requires sequential FPU (Floating Point Unit) math.
        - When dealing with a single entity, packing data into 128 bit SSE registers actually hurts performance.
        - CPU wastes clock cycles shuffling the data from the FPU, into the SSE registers for the cross product and then unpack it again for the Matrix.
        - To solve this, never leave SIMD registers (AAA Engines: a 4x4 matrix and a 3D vector are always 128-bit SIMD registers).
        - Load the camera data into an SSE register, perform all LookAt, Projection and View Matrix math inside SSE, and only extract the data when pushing it to the GPU via uniform buffers.
    */

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
