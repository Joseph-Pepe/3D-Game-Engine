#pragma once

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


#if __has_include(<inplace_vector>)
    /*
        // Replaces (std::vector). Zero heap allocations. Data is perfectly contiguous on the stack.
        // Extremely cache friendly for your SIMD wrappers.
        std::inplace_vector<Vector3D, 64> localCluster;
    */
    #include <inplace_vector> // C++26 API provides a vector that stores data locally without ever touching the heap allocator.
#endif 

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

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode
#if __has_include(<simd>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    // C++26 features are unlocked (Optional) Include <simd> if you want the portable vector typedefs
    #include <simd>
    #define ENGINE_HAS_CXX26_SIMD 1
#else
    // Fallback for C++23 and older
    #define ENGINE_HAS_CXX26_SIMD 0
#endif

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
        FORCE_INLINE SIMDVector3D toFloatVector() const {
            return SIMDVector3D(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        }
    };

    // ==================================================================================
    // ENGINE SUBSYSTEMS & PHYSICS
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
                // auto validCollisionMask = spatialMask && (absoluteJ > i).cast_to<float>();

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
        // 1. Calculate active batches
        size_t activeBatches = (memory.activeParticleCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;
        
        std::span<SIMDVector3D> posSpan(memory.positions.data(), activeBatches);
        std::span<SIMDVector3D> velSpan(memory.velocities.data(), activeBatches);
        
        std::span<ParticleSortKey> keySpan(memory.sortKeys.data(), memory.activeParticleCount);
        std::span<ParticleSortKey> bufferSpan(memory.sortKeysBuffer.data(), memory.activeParticleCount);

        // 2. Execute the Pipeline (Updates & Sorting)
        UpdateParticles(posSpan, velSpan, memory.activeParticleCount, deltaTime);
        GenerateMortonKeys(posSpan, keySpan, memory.activeParticleCount);
        RadixSortKeys(keySpan, bufferSpan);

        // =========================================================================
        // ZERO-ALLOCATION TEMPORARY WORKSPACE
        // =========================================================================
        
        // 3. Drop the ArenaMarker. 
        // This takes a snapshot of the bump pointer (e.g., offset = 0). 
        // When this function ends, the marker's destructor will instantly reset the memory.
        ArenaMarker frameMarker(t_PhysicsTransientArena);

        // 4. Ask the arena for aligned memory using your C++26 NativeAlignedArray.
        // This is an O(1) operation. No OS calls are made.
        NativeAlignedArray<SIMDVector3D> tempPos(t_PhysicsTransientArena, activeBatches);
        NativeAlignedArray<SIMDVector3D> tempVel(t_PhysicsTransientArena, activeBatches);

        // 5. Bypass Initialization. 
        // Since ReorderParticleData will immediately overwrite every single byte, 
        // zeroing out the memory first would waste CPU cycles.
        tempPos.ResizeUninitialized(activeBatches);
        tempVel.ResizeUninitialized(activeBatches);

        // =========================================================================

        // 6. Reorder data into our ultra-fast L1 cache aligned transient memory
        ReorderParticleData(posSpan, velSpan, tempPos, tempVel, keySpan);

        // 7. Commit the perfectly sorted data back to main memory
        std::copy(tempPos.begin(), tempPos.end(), memory.positions.begin());
        std::copy(tempVel.begin(), tempVel.end(), memory.velocities.begin());

        // 8. Resolve collisions
        ResolveCollisions(posSpan, memory.activeParticleCount, 2.0f);
        
        // 9. SCOPE ENDS HERE.
        // frameMarker goes out of scope. 
        // t_PhysicsTransientArena.SetOffset(m_savedOffset) is automatically called.
        // The memory used by tempPos and tempVel is instantly "freed" in zero clock cycles.
    }
#endif
