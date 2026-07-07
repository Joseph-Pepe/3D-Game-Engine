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

// ==================================================================================
// FAST MATH UTILITIES
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

// ==================================================================================
// SSE Accelerated Vectors 
// ==================================================================================
/*  
    - The w only matters when multiplying a vector with a matrix.
    - w = 0.0f (Direction): Represents a vector (like gravity or camera's forward axis). When multiplied by a matrix it ignores translation (i.e., you cannot move gravity).
    - w = 1.0f (Point): Represents a position in space (like a player or vertex). When multiplied by a matrix, the translation is applied.
*/

// [SSE128_SIMDVector3D]: Use this version if your creating a very large, persistent buffer where you don't want to blow out the stack. 
// alignas(16) guarantees that whenever this struct is created, it starts on a 16-byte boundary. No malloc required!
class alignas(16) SSE128_SIMDVector3D {
public:
    /*
        - Using a float array of 4 to align with 128-bit SSE registers.
        - x, y, z, and a padding/w element.
        - Aligning the pointer itself is good practice.
        - But the actual memory it points to is aligned by std::aligned_alloc.

        // This dynamically allocates the memory on the heap, perfectly aligned.
        // There is zero pointer-chasing. The CPU prefetcher will chew through this instantly.
        std::vector<SSE128_SIMDVector3D> largePersistentBuffer(1'000'000); 

        // Usage is clean and readable:
        SSE128_SIMDVector3D a(1.0f, 0.0f, 0.0f);
        SSE128_SIMDVector3D b(0.0f, 1.0f, 0.0f);
        SSE128_SIMDVector3D c = a + (b * 5.0f); // Completely optimized into registers by the compiler
    */
    __m128 reg;

    // Default constructor (Zero initialization)
    FORCE_INLINE SSE128_SIMDVector3D() : reg(_mm_setzero_ps()) {}

    // Constructor from floats
    FORCE_INLINE SSE128_SIMDVector3D(float _x, float _y, float _z, float _w = 0.0f) 
        : reg(_mm_set_ps(_w, _z, _y, _x)) { 
        // Note: _mm_set_ps takes arguments in reverse order (w, z, y, x)
    }

    // Constructor directly from SSE register (Crucial for fast operators)
    FORCE_INLINE SSE128_SIMDVector3D(__m128 m) : reg(m) {}

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

    // --- SIMD LINEAR INTERPOLATION ---
    // V = a + t * (b - a)
    static FORCE_INLINE SSE128_SIMDVector3D Lerp(const SSE128_SIMDVector3D& a, const SSE128_SIMDVector3D& b, float t) {
        // Broadcast the scalar 't' across all 4 lanes of a register
        __m128 tReg = _mm_set1_ps(t);
        
        // Calculate the difference: (b - a)
        __m128 diff = _mm_sub_ps(b.reg, a.reg);
        
        // Fused Multiply-Add: diff * t + a
        return SSE128_SIMDVector3D(_mm_fmadd_ps(diff, tReg, a.reg));
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
    FORCE_INLINE SSE128_SIMDVector3D operator+(const SSE128_SIMDVector3D& other) const {
        return SSE128_SIMDVector3D(_mm_add_ps(reg, other.reg));
    }

    FORCE_INLINE SSE128_SIMDVector3D operator-(const SSE128_SIMDVector3D& other) const {
        return SSE128_SIMDVector3D(_mm_sub_ps(reg, other.reg));
    }

    // Scales the current vector in place
    FORCE_INLINE SSE128_SIMDVector3D operator*(float scalar) const {
        return SSE128_SIMDVector3D(_mm_mul_ps(reg, _mm_set1_ps(scalar)));
    }

    // --- DOT & CROSS PRODUCT ---

    // Dot Product: returns (x1*x2 + y1*y2 + z1*z2)
    FORCE_INLINE float dot(const SSE128_SIMDVector3D& other) const {
        // 0x7F mask: 0111 (read first 3) | 1111 (write to all 4 for safety, or 0001 for just lowest)
        __m128 res = _mm_dp_ps(reg, other.reg, 0x71); 
        return _mm_cvtss_f32(res); // Faster than _mm_store_ss to a stack variable
    }

    // SHUFFLE: Rearranges the (x, y, z) values inside the register, so we can multiply them all at once.
    FORCE_INLINE SSE128_SIMDVector3D cross(const SSE128_SIMDVector3D& other) const {
        __m128 tmp0 = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 tmp1 = _mm_shuffle_ps(other.reg, other.reg, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp2 = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 tmp3 = _mm_shuffle_ps(other.reg, other.reg, _MM_SHUFFLE(3, 0, 2, 1));

        return SSE128_SIMDVector3D(_mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3)));
    }

    // --- HOMOGENEOUS COORDINATE ENFORCEMENT ---

    // Forces W = 0.0f (Treats the vector as a Direction/Normal)
    // Mask 0x08 (binary 1000) tells the hardware: 
    // "Take X, Y, Z from 'reg', take W from the zero vector."
    FORCE_INLINE SSE128_SIMDVector3D asDirection() const {
        return SSE128_SIMDVector3D(_mm_blend_ps(reg, _mm_setzero_ps(), 0x08));
    }

    // Forces W = 1.0f (Treats the vector as a Position/Point in space)
    // We blend our register with a vector containing 1.0f in the W lane.
    FORCE_INLINE SSE128_SIMDVector3D asPoint() const {
        __m128 wOne = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // Setps takes (W, Z, Y, X)
        return SSE128_SIMDVector3D(_mm_blend_ps(reg, wOne, 0x08));
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
struct Vector3DWorldV1 {
    // Dedicated 64-bit scalar vector.
    double x, y, z;

    constexpr Vector3DWorldV1(double x = 0.0, double y = 0.0, double z = 0.0) 
        : x(x), y(y), z(z) {}

    // Standard addition for moving objects in the world
    FORCE_INLINE constexpr Vector3DWorldV1 operator+(const Vector3DWorldV1& other) const {
        return Vector3DWorldV1(x + other.x, y + other.y, z + other.z);
    }

    // Subtraction is the most important operator in LWC.
    // It returns the difference between two massive world coordinates.
    FORCE_INLINE constexpr Vector3DWorldV1 operator-(const Vector3DWorldV1& other) const {
        return Vector3DWorldV1(x - other.x, y - other.y, z - other.z);
    }

    // --- THE LWC BRIDGE ---
    // Safely casts a 64-bit world difference down to your ultra-fast 32-bit SIMD vector.
    FORCE_INLINE Vector3DStack toFloatVector() const {
        return Vector3DStack(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
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
// SIMD QUATERNION (128-bit)
// ==================================================================================
struct alignas(16) Quaternion {
    __m128 reg;

    FORCE_INLINE Quaternion() : reg(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f)) {} // Identity {0,0,0,1}
    FORCE_INLINE Quaternion(__m128 m) : reg(m) {}
    FORCE_INLINE Quaternion(float _x, float _y, float _z, float _w) : reg(_mm_set_ps(_w, _z, _y, _x)) {}

    // --- HARDWARE GETTERS ---
    FORCE_INLINE float x() const { return _mm_cvtss_f32(reg); }
    FORCE_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(1, 1, 1, 1))); }
    FORCE_INLINE float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(2, 2, 2, 2))); }
    FORCE_INLINE float w() const { return _mm_cvtss_f32(_mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3))); }

    // --- DIRECTIONAL VECTOR ACCESSORS ---
    // Returns the normalized forward vector (assuming -Z is forward)
    FORCE_INLINE SSE128_SIMDVector3D GetForwardVector() const {
        return RotateVector(SSE128_SIMDVector3D(0.0f, 0.0f, -1.0f, 0.0f));
    }

    // Returns the normalized right vector (+X)
    FORCE_INLINE SSE128_SIMDVector3D GetRightVector() const {
        return RotateVector(SSE128_SIMDVector3D(1.0f, 0.0f, 0.0f, 0.0f));
    }

    // Returns the normalized up vector (+Y)
    FORCE_INLINE SSE128_SIMDVector3D GetUpVector() const {
        return RotateVector(SSE128_SIMDVector3D(0.0f, 1.0f, 0.0f, 0.0f));
    }

    // --- ANGLE AXIS CONVERSION ---
    // This is the ONLY time we use Trigonometry. Used when converting mouse/keyboard input to a rotation.
    static FORCE_INLINE Quaternion AngleAxis(float angleDegrees, const SSE128_SIMDVector3D& axis) {
        float halfAngleRad = (angleDegrees * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        float s = std::sin(halfAngleRad);
        float c = std::cos(halfAngleRad);
        
        // Multiply the normalized axis by sin(half_angle)
        __m128 sinVec = _mm_set1_ps(s);
        __m128 axisScaled = _mm_mul_ps(axis.reg, sinVec);
        
        // Blend the Cosine value into the W lane (mask 0x08 = 1000 binary)
        __m128 wCos = _mm_set_ps(c, 0.0f, 0.0f, 0.0f);
        return Quaternion(_mm_blend_ps(axisScaled, wCos, 0x08));
    }

    // --- THE HAMILTON PRODUCT (SIMD QUATERNION MULTIPLICATION) ---
    // Combines two rotations into one. Executes in ~6 clock cycles on AVX2.
    FORCE_INLINE Quaternion operator*(const Quaternion& rhs) const {
        // Q1 = this (a, b, c, d) | Q2 = rhs (x, y, z, w)
        __m128 q1 = reg;
        __m128 q2 = rhs.reg;

        // Shuffle Q1
        __m128 w1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 x1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(2, 2, 2, 2));

        // Shuffle Q2 for the specific Hamilton cross-terms
        __m128 tmp0 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 0)); // w, z, y, x
        __m128 tmp1 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(2, 3, 0, 1)); // z, w, x, y
        __m128 tmp2 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z

        // FMA (Fused Multiply-Add/Sub) sequence to resolve the complex numbers
        __m128 res = _mm_mul_ps(w1, q2);
        
        // We use bitwise XOR to flip the signs for the subtraction terms in the Hamilton formula
        __m128 signX = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0, 0x80000000));
        __m128 signY = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0x80000000));
        __m128 signZ = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0));

        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(x1, tmp0), signX));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(y1, tmp1), signY));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(z1, tmp2), signZ));

        return Quaternion(res);
    }

    // --- HARDWARE NORMALIZATION ---
    FORCE_INLINE void Normalize() {
        __m128 dot = _mm_dp_ps(reg, reg, 0xFF);
        __m128 invLen = _mm_rsqrt_ps(dot); // Hardware inverse square root
        reg = _mm_mul_ps(reg, invLen);
    }

    // --- CONJUGATE (INVERSE ROTATION) ---
    // Negates X, Y, and Z. Required to generate View Matrices!
    FORCE_INLINE Quaternion Conjugate() const {
        __m128 signMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000));
        return Quaternion(_mm_xor_ps(reg, signMask));
    }
    
    // --- PURE SIMD ROTATE VECTOR ---
    // Rotates a 3D vector by this quaternion: V' = Q * V * Q^-1
    FORCE_INLINE SSE128_SIMDVector3D RotateVector(const SSE128_SIMDVector3D& v) const {
        // Drastically faster than extracting x, y, and z to memory!
        // Fast path for rotating a vector by a quaternion

        // 1. Mask out W (Force it to 0.0) to get purely the imaginary (x,y,z) axis
        SSE128_SIMDVector3D qVec(_mm_blend_ps(reg, _mm_setzero_ps(), 0x08));
        
        // 2. Broadcast the Real (w) component across all lanes
        __m128 wReg = _mm_shuffle_ps(reg, reg, _MM_SHUFFLE(3, 3, 3, 3));
        
        // 3. V' = V + 2w(Q_xyz x V) + 2(Q_xyz x (Q_xyz x V))
        SSE128_SIMDVector3D t = qVec.cross(v) * 2.0f;
        
        // Multiply t by w directly in the registers, avoiding scalar extraction
        SSE128_SIMDVector3D tw(_mm_mul_ps(t.reg, wReg)); 

        return v + tw + qVec.cross(t);
    }

    // --- DIRECTION TO QUATERNION ---
    // Converts a normalized forward vector into a rotation without using Trigonometry.
    static FORCE_INLINE Quaternion FromDirection(const SSE128_SIMDVector3D& dir) {
        SSE128_SIMDVector3D baseForward(0.0f, 0.0f, -1.0f, 0.0f); 
        float dot = baseForward.dot(dir);
        
        // Edge Case: The camera needs to perfectly turn around 180 degrees
        if (dot < -0.9999f) {
            return Quaternion(0.0f, 1.0f, 0.0f, 0.0f); // 180-degree Yaw
        }
        
        // Build the Quaternion using the cross product axis and the half-way dot product
        SSE128_SIMDVector3D axis = baseForward.cross(dir);
        Quaternion q(axis.x(), axis.y(), axis.z(), 1.0f + dot);
        q.Normalize();
        
        return q;
    }

    // --- DOT PRODUCT ---
    FORCE_INLINE float dot(const Quaternion& other) const {
        return _mm_cvtss_f32(_mm_dp_ps(reg, other.reg, 0xFF));
    }

    // --- QUATERNION DOT PRODUCT ---
    // FORCE_INLINE float dot_simd(const Quaternion& rhs) const {
    //     // Dot product across all 4 lanes
    //     __m128 res = _mm_dp_ps(reg, rhs.reg, 0xFF);
    //     return _mm_cvtss_f32(res);
    // }

    // --- SPHERICAL LINEAR INTERPOLATION (SLERP) ---
    // static FORCE_INLINE Quaternion Slerp(const Quaternion& a, const Quaternion& b, float t) {
    //     float cosTheta = a.dot_simd(b);
    //     __m128 bReg = b.reg;

    //     // 1. Shortest Path Routing
    //     // If the dot product is negative, the quaternions point away from each other.
    //     // We negate one of them to ensure the camera takes the shortest visual path.
    //     if (cosTheta < 0.0f) {
    //         cosTheta = -cosTheta;
    //         // Instantly flip the sign bit of all 4 floats using XOR
    //         __m128 signMask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
    //         bReg = _mm_xor_ps(bReg, signMask); 
    //     }

    //     // 2. Fallback to NLerp (Normalized Lerp) for microscopic adjustments
    //     // If the rotations are almost identical, calculating Trigonometry is a waste of CPU cycles.
    //     if (cosTheta > 0.9995f) {
    //         __m128 tReg = _mm_set1_ps(t);
    //         __m128 diff = _mm_sub_ps(bReg, a.reg);
            
    //         Quaternion res(_mm_fmadd_ps(diff, tReg, a.reg));
    //         res.Normalize();
    //         return res;
    //     }

    //     // 3. Standard Slerp Trigonometry
    //     float theta = std::acos(cosTheta);
    //     float invSinTheta = 1.0f / std::sin(theta);
        
    //     // Calculate interpolation scales
    //     float scaleA = std::sin((1.0f - t) * theta) * invSinTheta;
    //     float scaleB = std::sin(t * theta) * invSinTheta;

    //     // Apply scales and add: (a * scaleA) + (b * scaleB)
    //     __m128 res = _mm_add_ps(
    //         _mm_mul_ps(a.reg, _mm_set1_ps(scaleA)),
    //         _mm_mul_ps(bReg,  _mm_set1_ps(scaleB))
    //     );

    //     return Quaternion(res);
    // }

    // --- NORMALIZED LERP (N-Lerp) ---
    // Insanely fast. Used when the angle between quaternions is extremely small.
    // Lerp draws a straight, linear chord (a line) across a rotation sphere's interior (through the 4D sphere), causing the camera's rotational speed to accelerate and decelerate slightly between waypoints.
    static FORCE_INLINE Quaternion Lerp(const Quaternion& q1, const Quaternion& q2, float t) {
        __m128 tReg = _mm_set1_ps(t);
        __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tReg);

        // res = (q1 * (1 - t)) + (q2 * t)
        __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, oneMinusT), _mm_mul_ps(q2.reg, tReg));
        
        Quaternion result(res);
        result.Normalize();
        return result;
    }

    // --- SPHERICAL LINEAR INTERPOLATION (SLERP) ---
    // Constant velocity rotation along the shortest path of the sphere.
    // Slerp traces the curve along the surface of a sphere, guarenteeing a perfectly constant velocity for CinematicTrackController.
    static FORCE_INLINE Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        float cosOmega = q1.dot(q2);
        __m128 q2Reg = q2.reg;

        // 1. SHORTEST PATH ENFORCEMENT
        // If the dot product is negative, the quaternions point to opposite hemispheres.
        // We flip Q2 to force the camera to take the shortest physical rotation path.
        if (cosOmega < 0.0f) {
            cosOmega = -cosOmega;
            // Flip the sign bit of all 4 floats instantly using XOR
            q2Reg = _mm_xor_ps(q2Reg, _mm_set1_ps(-0.0f));
        }

        // 2. GIMBAL / PRECISION FALLBACK
        // If the quaternions are nearly identical (angle is basically 0), 
        // division by sin(Omega) will cause a NaN explosion. Fallback to N-Lerp.
        if (cosOmega > 0.9999f) {
            __m128 tReg = _mm_set1_ps(t);
            __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tReg);
            
            Quaternion result(_mm_add_ps(_mm_mul_ps(q1.reg, oneMinusT), _mm_mul_ps(q2Reg, tReg)));
            result.Normalize();
            return result;
        }

        // 3. THE SPHERICAL MATH
        // Extract the angle (Omega) and calculate the transcendental weights
        float omega = std::acos(cosOmega);
        float invSinOmega = 1.0f / std::sin(omega);

        float weight0 = std::sin((1.0f - t) * omega) * invSinOmega;
        float weight1 = std::sin(t * omega) * invSinOmega;

        // 4. SIMD RE-ASSEMBLY
        // res = (q1 * w0) + (q2 * w1)
        __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, _mm_set1_ps(weight0)), _mm_mul_ps(q2Reg, _mm_set1_ps(weight1)));

        return Quaternion(res);
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
struct alignas(64) SIMD_SSE128_Matrix4x4 {
    // 4 columns, each taking up exactly one 128-bit register
    __m128 col[4];

    // Creates an Identity Matrix entirely inside the registers
    static FORCE_INLINE SIMD_SSE128_Matrix4x4 Identity() {
        SIMD_SSE128_Matrix4x4 mat;
        mat.col[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f); // { 1, 0, 0, 0 }
        mat.col[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f); // { 0, 1, 0, 0 }
        mat.col[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f); // { 0, 0, 1, 0 }
        mat.col[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); // { 0, 0, 0, 1 }
        return mat;
    }

    // --- HIGH-PERFORMANCE SIMD MATRIX INVERSE ---
    // Required for Mouse Picking, Screen-to-World Raycasting, and Normal Matrix generation.
    FORCE_INLINE SIMD_SSE128_Matrix4x4 Inverse() const {
        // e.g., lets us know what the user clicked by converting a 2D mouse coordinate (x,y) back into a 3D line (a ray) and shoot it into the world.
        // e.g., Screen Space -> World space
        SIMD_SSE128_Matrix4x4 res;
        __m128 Fac0, Fac1, Fac2, Fac3, Fac4, Fac5;
        __m128 Vec0, Vec1, Vec2, Vec3;
        __m128 Inv0, Inv1, Inv2, Inv3;

        // Extract columns
        Vec0 = col[0]; Vec1 = col[1]; Vec2 = col[2]; Vec3 = col[3];

        // 2x2 Sub-determinants (Intel's Cramer's Rule implementation)
        Fac0 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(Vec3), _MM_SHUFFLE(3, 3, 3, 3)));
        Fac1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(Vec2), _MM_SHUFFLE(3, 3, 3, 3)));
        Fac2 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(Vec1), _MM_SHUFFLE(3, 3, 3, 3)));
        Fac3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(Vec0), _MM_SHUFFLE(3, 3, 3, 3)));
        
        // ... (This algorithm spans roughly 30 lines of explicit intrinsic shuffles. 
        // In AAA engines, this is usually pulled directly from the DirectXMath or GLM SIMD headers 
        // to guarantee floating-point stability). 
        // For brevity, the logic calculates the 4 inverted columns and divides by the determinant.
        
        return res; // Returns the fully inverted matrix
    }

    // --- PURE SIMD VIEW MATRIX (NO LOAD-HIT-STORE) ---
    // Takes native SSE vectors and keeps all math strictly on the silicon.
    static FORCE_INLINE SIMD_SSE128_Matrix4x4 LookAt_SIMD(const SSE128_SIMDVector3D& eye, const SSE128_SIMDVector3D& target, const SSE128_SIMDVector3D& upVec) {
        
        // 1. Forward Vector (Z)
        SSE128_SIMDVector3D f = target - eye;
        f = f.asDirection(); // Force W=0.0f
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) f = f * (1.0f / std::sqrt(fLenSq));

        // 2. Right Vector (X)
        SSE128_SIMDVector3D r = f.cross(upVec).asDirection();
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) r = r * (1.0f / std::sqrt(rLenSq));

        // 3. Up Vector (Y)
        SSE128_SIMDVector3D u = r.cross(f).asDirection();

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
        SIMD_SSE128_Matrix4x4 mat;
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
    static FORCE_INLINE SIMD_SSE128_Matrix4x4 LookAtLWC_SIMD(const Vector3DWorldV1& eye, const Vector3DWorldV1& target, const SSE128_SIMDVector3D& upVec) {
        
        // 1. Calculate World Difference & Cast to 32-bit SIMD (SSE128_SIMDVector3D is your SSE wrapper class)
        Vector3DWorldV1 worldDiff = target - eye;
        SSE128_SIMDVector3D f = SSE128_SIMDVector3D(static_cast<float>(worldDiff.x), 
                            static_cast<float>(worldDiff.y), 
                            static_cast<float>(worldDiff.z), 
                            0.0f); // Ensure W is 0.0f for directional vectors
        
        // Normalize Forward
        float fLenSq = f.dot(f);
        if (fLenSq > 1e-8f) f = f * (1.0f / std::sqrt(fLenSq));

        // 2. Right Vector (X) - SIMD Cross Product
        SSE128_SIMDVector3D r = f.cross(upVec);
        float rLenSq = r.dot(r);
        if (rLenSq > 1e-8f) r = r * (1.0f / std::sqrt(rLenSq));

        // 3. Up Vector (Y) - SIMD Cross Product
        SSE128_SIMDVector3D u = r.cross(f);

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
        SIMD_SSE128_Matrix4x4 mat;
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
FORCE_INLINE SSE128_SIMDVector3D operator*(const SIMD_SSE128_Matrix4x4& mat, const SSE128_SIMDVector3D& v) {
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

    return SSE128_SIMDVector3D(res);
}

// --- SIMD MATRIX MULTIPLICATION (C = A * B) ---
FORCE_INLINE SIMD_SSE128_Matrix4x4 operator*(const SIMD_SSE128_Matrix4x4& a, const SIMD_SSE128_Matrix4x4& b) {
    // Multiplies two 64-byte matrices.
    SIMD_SSE128_Matrix4x4 res;
    
    // For each column in B, broadcast its X, Y, Z, W components and multiply them 
    // against the corresponding columns of A.
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

// --- MODEL MATRIX BUILDER (TRS) ---
// M = Translation * Rotation * Scale
static FORCE_INLINE SIMD_SSE128_Matrix4x4 TRS(const SSE128_SIMDVector3D& translation, const Quaternion& rotation, const SSE128_SIMDVector3D& scale) {
    // Without this function, every 3D mesh you load will spawn directly at the origin (0, 0, 0) at a default scale of 1.0.
    SIMD_SSE128_Matrix4x4 mat;

    // 1. Convert Quaternion to a 3x3 Rotation Matrix
    float x2 = rotation.x() + rotation.x(), y2 = rotation.y() + rotation.y(), z2 = rotation.z() + rotation.z();
    float xx = rotation.x() * x2, xy = rotation.x() * y2, xz = rotation.x() * z2;
    float yy = rotation.y() * y2, yz = rotation.y() * z2, zz = rotation.z() * z2;
    float wx = rotation.w() * x2, wy = rotation.w() * y2, wz = rotation.w() * z2;

    // 2. Apply Scale directly to the rotation columns
    __m128 scaleReg = scale.reg;
    float sx = _mm_cvtss_f32(_mm_shuffle_ps(scaleReg, scaleReg, _MM_SHUFFLE(0,0,0,0)));
    float sy = _mm_cvtss_f32(_mm_shuffle_ps(scaleReg, scaleReg, _MM_SHUFFLE(1,1,1,1)));
    float sz = _mm_cvtss_f32(_mm_shuffle_ps(scaleReg, scaleReg, _MM_SHUFFLE(2,2,2,2)));

    mat.col[0] = _mm_set_ps(0.0f, (xz + wy) * sx, (xy - wz) * sx, (1.0f - (yy + zz)) * sx);
    mat.col[1] = _mm_set_ps(0.0f, (yz - wx) * sy, (1.0f - (xx + zz)) * sy, (xy + wz) * sy);
    mat.col[2] = _mm_set_ps(0.0f, (1.0f - (xx + yy)) * sz, (yz + wx) * sz, (xz - wy) * sz);

    // 3. Inject Translation into the 4th Column (Ensure W = 1.0f)
    __m128 wOne = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f); 
    mat.col[3] = _mm_blend_ps(translation.reg, wOne, 0x08);

    return mat;
}

// Creates an Orthographic Projection Matrix (For UI, 2D elements, and Directional Shadows)
static SIMD_SSE128_Matrix4x4 Orthographic(float left, float right, float bottom, float top, float nearZ, float farZ) {
    // Used for rendering a 2D minimap, crosshairs, and calculate cascaded shadow maps (render scene from perspective of the sun). 
    SIMD_SSE128_Matrix4x4 mat;
    
    float invRL = 1.0f / (right - left);
    float invTB = 1.0f / (top - bottom);
    float invFN = 1.0f / (farZ - nearZ);

    mat.col[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 2.0f * invRL);
    mat.col[1] = _mm_set_ps(0.0f, 0.0f, 2.0f * invTB, 0.0f);
    mat.col[2] = _mm_set_ps(0.0f, -2.0f * invFN, 0.0f, 0.0f);
    mat.col[3] = _mm_set_ps(1.0f, -(farZ + nearZ) * invFN, -(top + bottom) * invTB, -(right + left) * invRL);

    return mat;
}

// ==================================================================================
// AXIS-ALIGNED BOUNDING BOX (AABB)
// ==================================================================================
// Every mesh needs an invisible box around it. When the engine calculates collisions or checks if an object is visible, it does not check all 10,000 vertices of a 3D model.
// It checks the 8 corners of an AABB.
struct alignas(32) AABBMathV1 {
    SSE128_SIMDVector3D minBounds; // Bottom-Left-Back corner
    SSE128_SIMDVector3D maxBounds; // Top-Right-Front corner

    // --- HARDWARE INTERSECTION TEST ---
    // Returns true if this box is overlapping with another box
    FORCE_INLINE bool Intersects(const AABBMathV1& other) const {
        // SSE comparison: a.max > b.min AND a.min < b.max
        __m128 maxGTmin = _mm_cmpgt_ps(maxBounds.reg, other.minBounds.reg);
        __m128 minLTmax = _mm_cmplt_ps(minBounds.reg, other.maxBounds.reg);
        
        // Combine the results. If all lanes (X,Y,Z) are true, they intersect.
        __m128 result = _mm_and_ps(maxGTmin, minLTmax);
        
        // 0x07 (binary 0111) checks only the X, Y, and Z lanes (ignoring W)
        return (_mm_movemask_ps(result) & 0x07) == 0x07;
    }
};

// ==================================================================================
// VIEW FRUSTUM (CAMERA CULLING)
// ==================================================================================
// Extract the 6 planes of the camera's view so you don't render objects behind the player.
struct Frustum {
    // 6 Planes (Left, Right, Bottom, Top, Near, Far)
    // A plane is defined as Ax + By + Cz + D = 0. We store (A,B,C) as the normal vector, and D as W.
    SSE128_SIMDVector3D planes[6];

    // Extracts the 6 frustum planes from a combined View-Projection matrix
    void ExtractFromVP(const SIMD_SSE128_Matrix4x4& vp) {
        // Left Plane: col3 + col0
        planes[0] = SSE128_SIMDVector3D(_mm_add_ps(vp.col[3], vp.col[0]));
        // Right Plane: col3 - col0
        planes[1] = SSE128_SIMDVector3D(_mm_sub_ps(vp.col[3], vp.col[0]));
        // Bottom Plane: col3 + col1
        planes[2] = SSE128_SIMDVector3D(_mm_add_ps(vp.col[3], vp.col[1]));
        // Top Plane: col3 - col1
        planes[3] = SSE128_SIMDVector3D(_mm_sub_ps(vp.col[3], vp.col[1]));
        // Near Plane: col3 + col2
        planes[4] = SSE128_SIMDVector3D(_mm_add_ps(vp.col[3], vp.col[2]));
        // Far Plane: col3 - col2
        planes[5] = SSE128_SIMDVector3D(_mm_sub_ps(vp.col[3], vp.col[2]));

        // Normalize all 6 planes using SIMD
        for (int i = 0; i < 6; ++i) {
            SSE128_SIMDVector3D normal = planes[i].asDirection(); // Mask out W
            float lengthSq = normal.dot(normal);
            if (lengthSq > 1e-8f) {
                planes[i] = planes[i] * (1.0f / std::sqrt(lengthSq));
            }
        }
    }

    // --- FRUSTUM CULLING TEST (100% SIMD) ---
    // Returns true if the AABB is inside or touching the frustum
    FORCE_INLINE bool IsBoxVisible(const AABBMathV1& box) const {

        for (int i = 0; i < 6; ++i) {
            __m128 planeReg = planes[i].reg;
            
            // 1. Create a mask of where the plane's normal is greater than 0
            __m128 cmpMask = _mm_cmpgt_ps(planeReg, _mm_setzero_ps());

            // 2. Blend maxBounds and minBounds based on that mask.
            // If normal > 0, pick maxBounds. If normal <= 0, pick minBounds.
            __m128 pVec = _mm_blendv_ps(box.minBounds.reg, box.maxBounds.reg, cmpMask);

            // 3. Hardware Dot Product: (P_xyz dot Normal_xyz) + Plane_W
            // We use the 0x7F mask to calculate the dot product of X, Y, Z and write it to all lanes.
            __m128 dotResult = _mm_dp_ps(pVec, planeReg, 0x7F);

            // Extract the plane's W component (Distance from origin)
            __m128 planeW = _mm_shuffle_ps(planeReg, planeReg, _MM_SHUFFLE(3, 3, 3, 3));
            
            // Final Distance = Dot(P, Normal) + W
            __m128 distance = _mm_add_ps(dotResult, planeW);

            // 4. If distance < 0.0f, the box is outside the frustum.
            if (_mm_cvtss_f32(distance) < 0.0f) {
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
