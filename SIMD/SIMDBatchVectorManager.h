#pragma once

#include "../Math.h"

#include <vector>
#include <execution>
#include <ranges>
#include <algorithm>
#include <immintrin.h>

// ======================================================================
// COMPILER PROBING: Check if C++26 SIMD is available
// ======================================================================

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode
#if __has_include(<simd>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    // C++26 features are unlocked
    #include <simd>
    #define ENGINE_HAS_CXX26_SIMD 1
#else
    // Fallback for C++23 and older
    #define ENGINE_HAS_CXX26_SIMD 0
#endif

// ======================================================================
// C++26: SIMD for MSVC build v14.51 and newer.
// ======================================================================

#if ENGINE_HAS_CXX26_SIMD

    // ======================================================================
    // 1. HARDWARE PROBING
    // ======================================================================
    // Let the compiler determine the optimal register width and memory alignment 
    // for the target platform (e.g., 16 bytes for ARM NEON, 32 for AVX2, 64 for AVX-512).
    using NativeFloatSIMD = std::simd<float, std::simd_abi::native<float>>;
    constexpr size_t NATIVE_BATCH_SIZE = NativeFloatSIMD::size();
    constexpr size_t NATIVE_ALIGNMENT = alignof(NativeFloatSIMD);

    // ======================================================================
    // 2. THE AOSOA BLOCK (The Cache-Friendly Data Chunk)
    // ======================================================================
    // This perfectly sizes itself to the hardware. The alignas tag guarantees 
    // it never straddles a cache-line boundary, preventing CPU stalling.
    struct alignas(NATIVE_ALIGNMENT) PortableParticleBlock {
        NativeFloatSIMD x;
        NativeFloatSIMD y;
        NativeFloatSIMD z;
        NativeFloatSIMD w; // Padding: Forces the struct to a power-of-2 size for optimal L1 cache streaming.

        // Built-in Math Functions (Keeps the processing loop clean)
        FORCE_INLINE void add(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            x += bx;
            y += by;
            z += bz;
        }

        FORCE_INLINE NativeFloatSIMD dot_fma(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) const {
            NativeFloatSIMD res = x * bx;
            res = std::fma(y, by, res);
            res = std::fma(z, bz, res);
            return res;
        }

        FORCE_INLINE void cross(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            NativeFloatSIMD rx = std::fma(y, bz, -(z * by));
            NativeFloatSIMD ry = std::fma(z, bx, -(x * bz));
            NativeFloatSIMD rz = std::fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }
    };

    // ======================================================================
    // 3. THE MANAGER (Auto-Vectorizing, Multi-Threaded, Cross-Platform)
    // ======================================================================
    class VectorManagerAoSoA_Portable {
    public:
        // Automatically enforces the hardware-specific memory boundary!
        std::vector<PortableParticleBlock, DynamicAlignedAllocator<PortableParticleBlock, NATIVE_ALIGNMENT>> blocks;

        VectorManagerAoSoA_Portable(size_t particleCount) {
            // Divide total particles by the hardware's native batch size, rounding up.
            size_t blockCount = (particleCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;
            blocks.resize(blockCount);

            // Broadcast initialize (std::simd constructor accepts a single float and copies it to all hardware lanes)
            for (auto& b : blocks) {
                b.x = 1.0f;
                b.y = 2.0f;
                b.z = 3.0f;
                b.w = 0.0f; 
            }
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            // 1. Broadcast the scalar inputs directly into SIMD registers once
            NativeFloatSIMD sX = stepX;
            NativeFloatSIMD sY = stepY;
            NativeFloatSIMD sZ = stepZ;
            NativeFloatSIMD smallVal = 0.00001f;

            // 2. The Execution Loop
            // std::execution::par_unseq maps directly to your CPU's thread pool while 
            // guaranteeing safety for Instruction-Level Parallelism (SIMD).
            std::for_each(std::execution::par_unseq, blocks.begin(), blocks.end(), [=](PortableParticleBlock& batch) {
                
                // Math: Notice there are no manual loads (_mm_load_ps) or stores (_mm_store_ps).
                // Because the struct is perfectly aligned, modifying 'batch.x' triggers the compiler 
                // to generate the exact same aligned-load/store assembly automatically.
                
                batch.add(sX, sY, sZ);
                
                NativeFloatSIMD d = batch.dot_fma(sX, sY, sZ);
                
                batch.x += d * smallVal;
                
                batch.cross(sX, sY, sZ);
            });
        }
    };

    // --- THE SYSTEM LAYER (Portable Vector Manager, Portable SIMD, Any Hardware) ---
    class VectorManagerSOA_Portable {
    public:
        // Memory is perfectly aligned for the target hardware!
        NativeAlignedVector<float> xs, ys, zs;

        VectorManagerSOA_Portable(size_t count) {
            constexpr size_t stride = native_simd::size();
            
            // Dynamic padding: hardware-agnostic boundary alignment (i.e., for ANY hardware architecture)
            size_t remainder = count % stride;

             // Safely pad to the nearest multiple of whatever 'stride' the hardware chose (i.e., engine automatically adapts its memory allocation, so it never crashes at the end of an array).
            size_t paddedCount = (remainder == 0) ? count : count + (stride - remainder);
            
            xs.resize(paddedCount, 1.0f);
            ys.resize(paddedCount, 2.0f);
            zs.resize(paddedCount, 3.0f);
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            // C++26 native_simd::size tells us exactly how many floats fit in one register
            constexpr size_t stride = native_simd::size();
            
            // Broadcast scalars directly into the portable SIMD types
            native_simd sX(stepX);
            native_simd sY(stepY);
            native_simd sZ(stepZ);
            native_simd smallVal(0.00001f);

            // 1. Extract raw pointers to guarantee contiguous memory
            float* ptrX = xs.data();
            float* ptrY = ys.data();
            float* ptrZ = zs.data();

            // Use a standard algorithm with the par_unseq policy
            // In C++26, the compiler is significantly better at "auto-vectorizing" standard containers when using the execution policies.
            auto indices = std::views::iota(0uz, xs.size() / stride);

            // 2. Safely capture raw pointers by value [=]
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=](size_t chunkIdx) {
                size_t i = chunkIdx * stride;

                // LOAD: Vector_aligned portable loads! This emits the fastest possible assembly.
                SIMDVectorP batch = { 
                    native_simd(ptrX + i, simd::vector_aligned), 
                    native_simd(ptrY + i, simd::vector_aligned), 
                    native_simd(ptrZ + i, simd::vector_aligned) 
                };

                // MATH
                batch.add(sX, sY, sZ);
                native_simd d = batch.dot_fma(sX, sY, sZ);
                
                // The old _mm512_add_ps(x, _mm512_mul_ps(d, smallVal)) collapses into standard C++ syntax while maintaining identical silicon execution.
                batch.x += d * smallVal;
                batch.cross(sX, sY, sZ);

                // STORE: Vector_aligned stores write directly to the hardware's Write-Combine buffer.
                batch.x.copy_to(ptrX + i, simd::vector_aligned);
                batch.y.copy_to(ptrY + i, simd::vector_aligned);
                batch.z.copy_to(ptrZ + i, simd::vector_aligned);
            });
        }
    };
#endif
