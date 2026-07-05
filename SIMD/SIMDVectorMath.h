#pragma once

#include "SIMD/SIMDCustomWrapper.h"
#include "../JobSystem.h"
#include "../Memory.h"

#include <vector>
#include <execution>
#include <ranges>

#include <cmath>       // Trigonometry (C++26 constexpr supported)
#include <print>       // Formatting
#include <cstdint>
#include <array>
#include <mdspan>
#include <cstddef>
#include <span>
#include <algorithm>   // Required for std::min, std::copy, std::swap
#include <utility>     // Required for std::swap (in some compilers)
#include <numeric>     // Standard header for std::reduce algorithms
#include <bit>         // Required for std::bit_cast

#include <cstdlib>
#include <new>         // C++17/26 hardware interference sizes
#include <memory>      // C++20/26 std::assume_aligned

// ======================================================================
// TRINITY SIMD PERFORMANCE 
// ======================================================================
/* 
    - Alignment, Intrinsics, Release Mode Optimization
    - Manual AOS SIMD is not always faster than the compiler. 
    - Changing the data layout from AOS to SOA is more powerful than just applying instructions (SIMD) to bad layouts (AOS).
    - Prefer SOA over AOS when dealing with more than 100 objects.
    - Organize your data first, and speed will follow (i.e., data layout dictates performance).
    - AAA Game Engines: Organize data flat (SOA, AoSoA), widen the math (AVX2), wake up the cores (Threads).
*/

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
    - This now scales to ARM NEON (Apple Silicon M1/M2/M3/M4+), AMD (Playstation 5, Xbox Series X) and Intel (SSE, AVX-256, AVX-512) automatically once compiled without a total rewrite of code (i.e., seamlessly maps those registers, cross-platform).
    - No need to use bitwise mask hacks (_mm512_cmp_ps_mask, _mm512_maskz_mul_ps) anymore b/c the compiler automatically translates these logical operators into hardware masks.
    - Never hardcode instruction sets into your datastructures. Write once and rely on the compiler to traslate it into the widest register the hardware supports.
    - No longer need to manually hardcode intrinsics for Intel-specific __mm256, __mm512 versions, the C++ compiler will figure out what the native hardware is based on the build flag (e.g., -march=native, -mavx512f, -mcpu=apple-1).
    - Decouples the engine from Intel by changing the compiler target flag in CMake.
    - Engine::ISAArch::simd generated assembly is identical to manual intrinsics (1:1 match). You lose zero performance.
*/
namespace Engine::ISAArch {

    // ======================================================================
    // HARDWARE PROBING
    // ======================================================================
    /*
        - Let the compiler determine the optimal register width and memory alignment for the target platform.
        - e.g., 16 bytes for ARM NEON, 32 for AVX2, 64 for AVX-512.
    */
    using NativeFloatSIMD = Engine::ISAArch::simd<float, Engine::ISAArch::simd_abi::native<float>>; // (or) its implicit equivalent "Engine::ISAArch::WideFloat", both mean [NativeFloatSIMD =  WideFloat]
    constexpr size_t NATIVE_BATCH_SIZE = NativeFloatSIMD::size();  // (or) its equivalent "WideFloat::size()"
    constexpr size_t NATIVE_HARDWARE_ALIGNMENT = alignof(NativeFloatSIMD); // (or) its equivalent "alignof(WideFloat)"

    // ======================================================================
    // C++26: HARDWARE-AWARE CACHE ALIGNMENT
    // ======================================================================
    // Dynamically queries the silicon compiling the code (Intel, AMD, ARM) for its exact L1 cache chunk size (almost universally 64 bytes).
    #ifdef __cpp_lib_hardware_interference_size
        constexpr size_t L1_CACHE_CHUNK_SIZE = std::hardware_constructive_interference_size;
    #else
        constexpr size_t L1_CACHE_CHUNK_SIZE = 64; // Fallback for older compilers
    #endif

    FORCE_INLINE void clear_registers() {
        #if defined(ENGINE_ARCH_AVX2) || defined(ENGINE_ARCH_AVX512)
            _mm256_zeroupper(); // Or _mm512_zeroupper on some older Xeon architectures
        #endif
        // ARM NEON and SSE4.1 do not suffer from state-transition penalties, so they do nothing.
    }

    // Dynamically requests standard C++17 aligned memory based on the native hardware SIMD width
    template<typename T>
    using NativeAlignedVector = std::vector<T, AlignedAllocator<T, alignof(WideBatch<T>)>>;

    // Dynamically requests standard C++17 aligned memory for 128-bit GPU geometry
    template<typename T>
    using FixedAlignedVector = std::vector<T, AlignedAllocator<T, alignof(FixedBatch4<T>)>>;

    // ======================================================================
    // 1. THE AOSOA BLOCK (The Cache-Friendly Data Chunk)
    // ======================================================================
    // Force the struct itself to align to 32-bytes (3x32 = 96-bytes), not 128-bytes.
    // This perfectly sizes itself to the hardware. The alignas tag guarantees it never straddles a cache-line boundary, preventing CPU stalling.
    struct alignas(NATIVE_HARDWARE_ALIGNMENT) PortableParticleBlock {
        NativeFloatSIMD x; // 32-bytes 
        NativeFloatSIMD y; // 32-bytes
        NativeFloatSIMD z; // 32-bytes
        // NativeFloatSIMD w; // 32-bytes Padding: Forces the struct to a power-of-2 size for optimal L1 cache streaming. By adding w as padding to force a 128-byte alignment, it will tank performance because it needs to move 33% more memory on every single iteration (CPU load and stores).

        // Built-in Math Functions (Keeps the processing loop clean)
        FORCE_INLINE void add(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            x += bx;
            y += by;
            z += bz;
        }
        
        FORCE_INLINE WideFloat dot_fma(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) const {
            NativeFloatSIMD res = x * bx;
            res = fma(y, by, res);
            res = fma(z, bz, res);
            return res;
        }

        FORCE_INLINE void cross(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
            NativeFloatSIMD rx = fma(y, bz, -(z * by));
            NativeFloatSIMD ry = fma(z, bx, -(x * bz));
            NativeFloatSIMD rz = fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }
    };

    // ======================================================================
    // 2. THE MANAGER (Auto-Vectorizing, Multi-Threaded, Cross-Platform)
    // ======================================================================
    class VectorManagerAoSoA_Portable {
    public:
        // Automatically enforces the hardware-specific memory boundary!
        std::vector<PortableParticleBlock, AlignedAllocator<PortableParticleBlock, NATIVE_HARDWARE_ALIGNMENT>> blocks;

        VectorManagerAoSoA_Portable(size_t particleCount) {
            // Divide total particles by the hardware's native batch size, rounding up.
            size_t blockCount = (particleCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;
            blocks.resize(blockCount);

            // Broadcast initialize (WideFloat constructor copies scalar to all hardware lanes)
            for (auto& b : blocks) {
                b.x = NativeFloatSIMD(1.0f);
                b.y = NativeFloatSIMD(2.0f);
                b.z = NativeFloatSIMD(3.0f);
                // b.w = NativeFloatSIMD(0.0f);
            }
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            // 1. Broadcast the scalar inputs directly into SIMD registers once
            NativeFloatSIMD sX(stepX);
            NativeFloatSIMD sY(stepY);
            NativeFloatSIMD sZ(stepZ);
            NativeFloatSIMD smallVal(0.00001f);

            uint32_t blockCount = static_cast<uint32_t>(blocks.size());
            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t targetChunksPerThread = 16;
            
            // Calculate an optimal chunk size to keep L1/L2 caches happy
            uint32_t CHUNK_SIZE = std::max(1024u, blockCount / (threadCount * targetChunksPerThread));

            g_JobSystem.DispatchAndWait(blockCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

                // std::execution::par_unseq maps directly to your CPU's thread pool
                // std::for_each(std::execution::par_unseq, blocks.begin(), blocks.end(), [=](PortableParticleBlock& batch) { 
                //     // Because the struct is perfectly aligned, modifying 'batch.x' triggers the compiler 
                //     // to generate the exact same aligned-load/store assembly automatically.
                //     batch.add(sX, sY, sZ);
                //     WideFloat d = batch.dot_fma(sX, sY, sZ);
                //     batch.x += d * smallVal;
                //     batch.cross(sX, sY, sZ);
                // });

                // 2. The Execution Loop
                for (uint32_t i = start; i < end; ++i) {
                    
                    // 3. LOAD: Copy from heap memory into CPU registers (No '&' reference!)
                    PortableParticleBlock batch = blocks[i]; 
                    
                    // 4. MATH: Pure register execution (Zero RAM access). Notice there are no manual loads (_mm_load_ps) or stores (_mm_store_ps).
                    batch.add(sX, sY, sZ);
                    NativeFloatSIMD d = batch.dot_fma(sX, sY, sZ);
                    batch.x += d * smallVal;
                    batch.cross(sX, sY, sZ);

                    // Emits a generic memory copy rather than aligned AVX stores, better to break it up by parts.
                    // blocks[i] = batch;

                    // 5. STORE: Write the final computed result back to memory, break it apart to match AVX2 speed.
                    blocks[i].x = batch.x;
                    blocks[i].y = batch.y;
                    blocks[i].z = batch.z;
                }
                // Clean the Caller Thread before returning to the SSE benchmark!
                clear_registers();
            });
        }
    };

    // ======================================================================
    // 3. THE SYSTEM LAYER (Portable Vector Manager, SOA Architecture)
    // ======================================================================
    class VectorManagerSOA_Portable {
    public:
        // Memory is perfectly aligned for the target hardware!
        NativeAlignedVector<float> xs, ys, zs;

        // AlignedVector32<float> xs, ys, zs;           // AVX-256: 32-byte aligned vectors
        // AlignedVector64<float> xs, ys, zs;           // AVX-512: 64-byte aligned vectors

        VectorManagerSOA_Portable(size_t count) {
            // size_t paddedCount = (count + 15) & ~15; // Pad to nearest multiple of 16 for AVX-512 boundaries
            // size_t paddedCount = (count + 7) & ~7;   // Pad to nearest multiple of  8 for AVX-256 boundaries
             
            constexpr size_t stride = NativeFloatSIMD::size();

            // Dynamic padding: hardware-agnostic boundary alignment (i.e., for ANY hardware architecture)
            size_t remainder = count % stride;

            // Safely pad to the nearest multiple of whatever 'stride' the hardware chose (i.e., engine automatically adapts its memory allocation, so it never crashes at the end of an array).
            size_t paddedCount = (remainder == 0) ? count : count + (stride - remainder);
            
            xs.resize(paddedCount, 1.0f);
            ys.resize(paddedCount, 2.0f);
            zs.resize(paddedCount, 3.0f);
        }

        struct SOA_Batch {
            NativeFloatSIMD x, y, z;
            FORCE_INLINE void add(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) { x += bx; y += by; z += bz; }
            FORCE_INLINE NativeFloatSIMD dot_fma(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) const {
                NativeFloatSIMD res = x * bx; res = fma(y, by, res); res = fma(z, bz, res); return res;
            }
            FORCE_INLINE void cross(const NativeFloatSIMD& bx, const NativeFloatSIMD& by, const NativeFloatSIMD& bz) {
                NativeFloatSIMD rx = fma(y, bz, -(z * by)); NativeFloatSIMD ry = fma(z, bx, -(x * bz)); NativeFloatSIMD rz = fma(x, by, -(y * bx));
                x = rx; y = ry; z = rz;
            }
        };

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            // NativeFloatSIMD::size tells us exactly how many floats fit in one register.
            constexpr size_t stride = NativeFloatSIMD::size();
        
            // Broadcast the scalars directly into all the portable SIMD types [8 slots of the 256-bit registers (AVX-256)] (or) all [16 slots of the 512-bit registers (AVX-512)].
            // AVX-256: [32-bits (float) x  8 slots (vectors) = 256-bits] 
            // AVX-512: [32-bits (float) x 16 slots (vectors) = 512-bits]
            NativeFloatSIMD sX(stepX);           // AVX-256: __m256 sX       = _mm256_set1_ps(stepX);    AVX-512: __m512 sX       = _mm512_set1_ps(stepX);
            NativeFloatSIMD sY(stepY);           // AVX-256: __m256 sY       = _mm256_set1_ps(stepY);    AVX-512: __m512 sY       = _mm512_set1_ps(stepY);
            NativeFloatSIMD sZ(stepZ);           // AVX-256: __m256 sZ       = _mm256_set1_ps(stepZ);    AVX-512: __m512 sZ       = _mm512_set1_ps(stepZ);
            NativeFloatSIMD smallVal(0.00001f);  // AVX-256: __m256 smallVal = _mm256_set1_ps(0.00001f); AVX-512: __m512 smallVal = _mm512_set1_ps(0.00001f);

            // Extract raw pointers to guarantee contiguous memory
            float* ptrX = xs.data();
            float* ptrY = ys.data();
            float* ptrZ = zs.data();

            // Use a standard algorithm with the par_unseq policy
            // In C++26, the compiler is significantly better at "auto-vectorizing" standard containers when using the execution policies.
            // auto indices = std::views::iota(0uz, xs.size() / stride);

            // Safely capture raw pointers by value [=]
            // std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [=](size_t chunkIdx) {
            //     size_t i = chunkIdx * stride;

            //     // LOAD: Vector_aligned portable loads! This emits the fastest possible assembly.
            //     SIMDVector3D batch = { 
            //         NativeFloatSIMD(ptrX + i, simd::vector_aligned), 
            //         NativeFloatSIMD(ptrY + i, simd::vector_aligned), 
            //         NativeFloatSIMD(ptrZ + i, simd::vector_aligned) 
            //     };

            //     // MATH
            //     batch.add(sX, sY, sZ);
            //     NativeFloatSIMD d = batch.dot_fma(sX, sY, sZ);
                
            //     // The old _mm512_add_ps(x, _mm512_mul_ps(d, smallVal)) collapses into standard C++ syntax while maintaining identical silicon execution.
            //     batch.x += d * smallVal;
            //     batch.cross(sX, sY, sZ);

            //     // STORE: Vector_aligned stores write directly to the hardware's Write-Combine buffer.
            //     batch.x.copy_to(ptrX + i, simd::vector_aligned);
            //     batch.y.copy_to(ptrY + i, simd::vector_aligned);
            //     batch.z.copy_to(ptrZ + i, simd::vector_aligned);
            // });

            // Calculate chunks based on hardware SIMD width
            uint32_t chunkCount = static_cast<uint32_t>(xs.size() / stride);
            
            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t targetChunksPerThread = 16;
            uint32_t CHUNK_SIZE = std::max(1024u, chunkCount / (threadCount * targetChunksPerThread));

            // Dispatch to the Coroutine Work-Stealing Queue
            g_JobSystem.DispatchAndWait(chunkCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
                ENGINE_UNROLL_4 // MSVC specific to force the compiler to pipeline these instructions like it does for raw arrays.
                for (uint32_t chunkIdx = start; chunkIdx < end; ++chunkIdx) {
                    size_t i = chunkIdx * stride;

                    // 1. Load (32 bytes per load) (AVX-256: 8 vectors at once, AVX-512: 16 vectors at once).
                    // WideFloat(ptr) constructor automatically maps to hardware load instructions.
                    SOA_Batch batch = { 
                        NativeFloatSIMD(ptrX + i),  // AVX-256: _mm256_loadu_ps(&xs[i]);  AVX-512: _mm512_loadu_ps(&xs[i]);
                        NativeFloatSIMD(ptrY + i),  // AVX-256: _mm256_loadu_ps(&ys[i]);  AVX-512: _mm512_loadu_ps(&ys[i]);
                        NativeFloatSIMD(ptrZ + i)   // AVX-256: _mm256_loadu_ps(&zs[i]);  AVX-512: _mm512_loadu_ps(&zs[i]);
                    };

                    // 2. Addition (AVX-256: 8 at once, AVX-512: 16 at once)
                    batch.add(sX, sY, sZ);          // AVX-256: x = _mm256_add_ps(x, bx);  AVX-512: x = _mm512_add_ps(x, bx);
                                                    // AVX-256: y = _mm256_add_ps(y, by);  AVX-512: y = _mm512_add_ps(y, by);
                                                    // AVX-256: z = _mm256_add_ps(z, bz);  AVX-512: z = _mm512_add_ps(z, bz);

                    // 3. Dot Product
                    NativeFloatSIMD d = batch.dot_fma(sX, sY, sZ);

                    // 4. Update X, Single element update (x += d * small)
                    batch.x += d * smallVal;  // AVX-256: batch.x = _mm256_add_ps(batch.x, _mm256_mul_ps(d, smallVal));
                                              // AVX-512: batch.x = _mm512_add_ps(batch.x, _mm512_mul_ps(d, smallVal));

                    // 5. Cross Product (AVX-256: 8 simultaneous cross products, AVX-512: 16 simultaneous cross products)
                    batch.cross(sX, sY, sZ);

                    // 6. Store (AVX-256: 8, AVX-512: 16) results back 
                    batch.x.copy_to(ptrX + i);  // AVX-256: _mm256_storeu_ps(&xs[i], batch.x);  AVX-512: _mm512_storeu_ps(&xs[i], batch.x);
                    batch.y.copy_to(ptrY + i);  // AVX-256: _mm256_storeu_ps(&ys[i], batch.y);  AVX-512: _mm512_storeu_ps(&ys[i], batch.y);
                    batch.z.copy_to(ptrZ + i);  // AVX-256: _mm256_storeu_ps(&zs[i], batch.z);  AVX-512: _mm512_storeu_ps(&zs[i], batch.z);
                }

                // If MSVC's /O2 optimizer is being stubborn, you brute-force it by manually unrolling:
                /*
                    for (uint32_t chunkIdx = start; chunkIdx < end; chunkIdx += 4) {
                        size_t i0 = (chunkIdx + 0) * stride;
                        size_t i1 = (chunkIdx + 1) * stride;
                        size_t i2 = (chunkIdx + 2) * stride;
                        size_t i3 = (chunkIdx + 3) * stride;

                        // Load 4
                        SOA_Batch b0 = { WideFloat(ptrX + i0), WideFloat(ptrY + i0), WideFloat(ptrZ + i0) };
                        SOA_Batch b1 = { WideFloat(ptrX + i1), WideFloat(ptrY + i1), WideFloat(ptrZ + i1) };
                        SOA_Batch b2 = { WideFloat(ptrX + i2), WideFloat(ptrY + i2), WideFloat(ptrZ + i2) };
                        SOA_Batch b3 = { WideFloat(ptrX + i3), WideFloat(ptrY + i3), WideFloat(ptrZ + i3) };

                        // Math 4
                        b0.add(sX, sY, sZ); 
                        b1.add(sX, sY, sZ); 
                        b2.add(sX, sY, sZ); 
                        b3.add(sX, sY, sZ);
                        // ... rest of the math ...

                        // Store 4
                        b0.x.copy_to(ptrX + i0); // ... etc
                    }
                */
                // Clean the Caller Thread before returning to the SSE benchmark!
                clear_registers();
            });
        }
    };

    // ======================================================================
    // THE MACRO-CHUNK (The Perfect AoSoA Memory Layout)
    // ======================================================================    
    /*
        - 4 REGISTERS (X) + 4 REGISTERS (Y) + 4 REGISTERS (Z) = 12 REQUIRED VECTOR REGISTERS
        - Fits inside AVX2's 16 available registers, leaving 4 registers left over for temporary math variables (sX, sY, sZ, smallVal)
    */

    // The "Sweet Spot" Macro Chunk (Perfect Cache Line Multiples, No Padding)
    constexpr size_t REGISTERS_PER_CHUNK = 4; 

    // Particles per chunk now scales natively! [AVX2 = (4 * 8) = 32, NEON = (4 * 4) = 16, AVX-512 = (4 * 16) = 64].
    // AVX2 = 32 particles per chunk. AVX-512 = 64 particles per chunk.
    constexpr size_t MACRO_CHUNK_PARTICLES = REGISTERS_PER_CHUNK * WideFloat::size();
    
    struct alignas(L1_CACHE_CHUNK_SIZE) OptimalParticleMacroChunk {
        WideFloat x[REGISTERS_PER_CHUNK];
        WideFloat y[REGISTERS_PER_CHUNK];
        WideFloat z[REGISTERS_PER_CHUNK];

        FORCE_INLINE void process_chunk(const WideFloat& sX, const WideFloat& sY, const WideFloat& sZ, const WideFloat& smallVal) {
            
            // 1. ADDITION (Planar Execution)
            // The CPU executes these linearly, maxing out the ALU without spilling registers.
            x[0] += sX; x[1] += sX; x[2] += sX; x[3] += sX;
            y[0] += sY; y[1] += sY; y[2] += sY; y[3] += sY;
            z[0] += sZ; z[1] += sZ; z[2] += sZ; z[3] += sZ;

            // 2. DOT PRODUCT & SCALE
            // Let the compiler decide unrolling for the complex math
            for(int i=0; i < REGISTERS_PER_CHUNK; i++){
                WideFloat res = fma(z[i], sZ, fma(y[i], sY, x[i] * sX));
                x[i] += res * smallVal;
            }

            // 3. CROSS PRODUCT
            for(int i=0; i < REGISTERS_PER_CHUNK; i++){
                 WideFloat rx = fma(y[i], sZ, -(z[i] * sY));
                 WideFloat ry = fma(z[i], sX, -(x[i] * sZ));
                 WideFloat rz = fma(x[i], sY, -(y[i] * sX));
                 x[i] = rx; y[i] = ry; z[i] = rz;
            }
        }
    };

    class VectorManagerAoSoA_Macro {
    public:
        // Guaranteed allocation on the exact boundaries required by the silicon
        std::vector<OptimalParticleMacroChunk, AlignedAllocator<OptimalParticleMacroChunk, L1_CACHE_CHUNK_SIZE>> chunks;

        VectorManagerAoSoA_Macro(size_t particleCount) {
            // Calculate exactly how many 3072-byte chunks we need
            size_t chunkCount = (particleCount + MACRO_CHUNK_PARTICLES - 1) / MACRO_CHUNK_PARTICLES;
            chunks.resize(chunkCount);

            // Broadcast initialize the chunks
            for (auto& chunk : chunks) {
                for (size_t j = 0; j < REGISTERS_PER_CHUNK; ++j) {
                    chunk.x[j] = WideFloat(1.0f);
                    chunk.y[j] = WideFloat(2.0f);
                    chunk.z[j] = WideFloat(3.0f);
                }
            }
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            uint32_t chunkCount = static_cast<uint32_t>(chunks.size());
            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            
            // Because each chunk is doing more work (e.g. 32 particles on AVX2), 
            // we scale down the target chunks per thread to maintain work-stealing balance.
            uint32_t targetChunksPerThread = 4; 
            uint32_t JOB_CHUNK_SIZE = std::max(256u, chunkCount / (threadCount * targetChunksPerThread));

            // Extract the raw pointer and apply C++20/26 std::assume_aligned.
            // This legally guarantees to the compiler that it is safe to use aligned stores
            // (e.g., vmovaps) instead of unaligned stores (vmovups), unlocking the final 1% of pipelining.
            auto* RESTRICT alignedChunks = std::assume_aligned<L1_CACHE_CHUNK_SIZE>(chunks.data());

            // Capture the raw scalar floats (stepX, stepY, stepZ). 
            // These are 4 bytes each, meaning the lambda struct is tiny and alignment-safe.
            g_JobSystem.DispatchAndWait(chunkCount, JOB_CHUNK_SIZE, [=](uint32_t start, uint32_t end) {

                // Broadcast into SIMD registers natively on the worker thread!
                WideFloat sX(stepX);
                WideFloat sY(stepY);
                WideFloat sZ(stepZ);
                WideFloat smallVal(0.00001f);

                for (uint32_t i = start; i < end; ++i) {
                    
                    // We map a reference directly over the heap memory. 
                    // No copying the struct to the stack! The CPU executes directly against the L1 cache.
                    OptimalParticleMacroChunk& chunk = alignedChunks[i];
                    chunk.process_chunk(sX, sY, sZ, smallVal);
                }
                clear_registers();
            });
        }
    };
}

namespace Engine::Physics {
    // --- 1. MATH LAYER (Custom Portable SIMD) --- 
    // Mapped directly to your custom Tier 1 SOA architecture (i.e., let the compiler decide the widest register available on the target hardware based on the silicon it detects at compile time).
    using NativeFloatSIMDBatch = Engine::ISAArch::simd<float, Engine::ISAArch::simd_abi::native<float>>;

    // Automatically scales: 4 (SSE/NEON), 8 (AVX2), or 16 (AVX-512)
    constexpr std::size_t NATIVE_BATCH_SIZE = NativeFloatSIMDBatch::size();

    // Use this constant to dynamically align your memory allocators and structs! Ask exactly how many bytes the current hardware needs.
    constexpr std::size_t NATIVE_SIMD_BATCH_ALIGN = alignof(NativeFloatSIMDBatch);     

    // ================================================================================
    // VECTOR3D STRUCTS (INTRINSICS)
    // ================================================================================
    /*
        struct alignas(16) Vector3DStackAligned {
            float x, y, z, w; // Total 16 bytes
        };

        // This represents 4 vectors at once
        struct Vector3D_SOA_Batch {
            __m128 x; // [v1.x, v2.x, v3.x, v4.x]
            __m128 y; // [v1.y, v2.y, v3.y, v4.y]
            __m128 z; // [v1.z, v2.z, v3.z, v4.z]
        };
    */

    // ==================================================================================
    // BULK DATA PROCESSING (SOA) - SCALES TO ANY CPU AUTOMATICALLY
    // ==================================================================================
    /*
        - Instead of manual __m256 or __m512 loads, you use a template that automatically picks the widest register the hardware supports.
        - Under the hood, this is __m128, __m256, __m512, or float32x4_t. Your code no longer cares.

          AVX-256 (__m256):

            [v1.x, v2.x, v3.x, v4.x, v5.x, v6.x, v7.x, v8.x]
            [v1.y, v2.y, v3.y, v4.y, v5.y, v6.y, v7.y, v8.y]
            [v1.z, v2.z, v3.z, v4.z, v5.z, v6.z, v7.z, v8.z]

          AVX-512 (__m512): 
          
            [v1.x, v2.x, v3.x, v4.x, v5.x, v6.x, v7.x, v8.x, v9.x, v10.x, v11.x, v12.x, v13.x, v14.x, v15.x, v16.x]
            [v1.y, v2.y, v3.y, v4.y, v5.y, v6.y, v7.y, v8.y, v9.y, v10.y, v11.y, v12.y, v13.y, v14.y, v15.y, v16.y]
            [v1.z, v2.z, v3.z, v4.z, v5.z, v6.z, v7.z, v8.z, v9.z, v10.z, v11.z, v12.z, v13.z, v14.z, v15.z, v16.z]
    */

    // Dynamic Alignment Wrapper: [AVX-512: 64-byte alignment], [AVX-256: 32-byte alignment], [ARM NEON: 16-byte alignment]
    struct alignas(NATIVE_SIMD_BATCH_ALIGN) SIMDVector3D {
        NativeFloatSIMDBatch x; // [v1.x, v2.x, v3.x, v4.x, ...]
        NativeFloatSIMDBatch y; // [v1.y, v2.y, v3.y, v4.y, ...]
        NativeFloatSIMDBatch z; // [v1.z, v2.z, v3.z, v4.z, ...]

        // Standard Addition
        FORCE_INLINE void add(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) {
            // Custom SIMD supports standard operators (overloaded operators)!
            x += bx;  
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
        FORCE_INLINE NativeFloatSIMDBatch dot_fma(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
            // Automatically fuses (a * b + c) into a single clock cycle.
            NativeFloatSIMDBatch res = x * bx;
            res = fma(y, by, res); // ADL automatically resolves Engine::ISAArch::fma
            res = fma(z, bz, res);
            return res;
        }

        // Standard Dot Product
        FORCE_INLINE NativeFloatSIMDBatch dot(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
            return (x * bx) + (y * by) + (z * bz);
        }

        // SOA Cross Product
        FORCE_INLINE void cross(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) {
            // Fused Multiply-Add cross product. Compiles to _mm512_fmsub_ps on AVX-512, and vfmaq_f32 on ARM automatically.
            // Explicitly using fma to guarantee hardware Fused Multiply-Subtract (FMS) 
            // Example: (y * bz) - (z * by) -> fma(y, bz, -(z * by))
            NativeFloatSIMDBatch rx = fma(y, bz, -(z * by));
            NativeFloatSIMDBatch ry = fma(z, bx, -(x * bz));
            NativeFloatSIMDBatch rz = fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }

        // Magnitude Squared
        FORCE_INLINE NativeFloatSIMDBatch length_sq() const {
            // Nudging compiler to use FMA
            NativeFloatSIMDBatch sq = x * x;
            sq = fma(y, y, sq);
            sq = fma(z, z, sq);

            // return (x * x) + (y * y) + (z * z)
            return sq;
        }

        // Magnitude
        FORCE_INLINE NativeFloatSIMDBatch length() const {
            return sqrt(length_sq()); // ADL resolves Engine::ISAArch::sqrt
        }

        // --- PORTABLE OPMASK LOGIC ---
        FORCE_INLINE void normalize() {
            NativeFloatSIMDBatch sqLen = length_sq();
            NativeFloatSIMDBatch epsilon = 1e-8f;
            
            // 1. Generate the Hardware Mask (AVX-512, this generates an `__mmask16`), (AVX2, this generates a `__m256` bitmask)
            auto validMask = sqLen > epsilon;

            // 2. Prevent NaN/Inf generation by patching invalid lengths to 1.0f BEFORE division (i.e., if sqLen is 0, we temporarily pretend it is 1.0f so division succeeds gracefully).
            NativeFloatSIMDBatch safeSqLen = sqLen;
            Engine::ISAArch::where(!validMask, safeSqLen) = 1.0f; 

            // 3. Fast-math will translate this to a hardware reciprocal square root instruction natively via our wrapper (emits _mm512_rsqrt14_ps on AVX-512), (emits _mm256_rsqrt_ps on AVX2), (emits vrsqrteq_f32 on ARM NEON).
            NativeFloatSIMDBatch invLen = rsqrt(safeSqLen);

            // 4. Masked Assignment (Multiplication).
            Engine::ISAArch::where(validMask, x) *= invLen;
            Engine::ISAArch::where(validMask, y) *= invLen;
            Engine::ISAArch::where(validMask, z) *= invLen;

            // 5. Zero out the invalid lanes
            Engine::ISAArch::where(!validMask, x) = 0.0f;
            Engine::ISAArch::where(!validMask, y) = 0.0f;
            Engine::ISAArch::where(!validMask, z) = 0.0f;
        }
    };

    // Let the compiler dynamically pick 4-wide (NEON), 8-wide (AVX2), or 16-wide (AVX-512) integers
    using NativeUIntBatch = Engine::ISAArch::simd<uint32_t, Engine::ISAArch::simd_abi::native<uint32_t>>; // (or) its implicit equivalent "Engine::ISAArch::WideUInt32", both mean [NativeUIntBatch =  WideUInt32]

    // --- PORTABLE MORTON CODE VECTORIZATION (CROSS-PLATFORM) ---
    FORCE_INLINE NativeUIntBatch expandBits_SIMD(NativeUIntBatch v) {
        // The compiler automatically translates these bitwise operators into vector instructions (e.g., _mm256_slli_epi32 and _mm256_and_si256 on Intel)!
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
        - World Space (64-bit): Entities, Transforms, and the camera track their absolute positions in the universe using doubles.
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

        // Subtraction is the most important operator in LWC. It returns the difference between two massive world coordinates.
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
    struct ParticleSortKey {
        // 64-bit struct: 32 bits for the Morton Code, 32 bits for the physical particle index.
        uint32_t mortonCode;
        uint32_t particleIndex;
    };

    // --- DATA-ORIENTED PARTICLE SYSTEM (AoSoA PIPELINE, Pure Data Container (No Logic)) ---
    struct ParticleMemoryBlock {
        // Every element in this vector represents a BATCH of particles (4 on ARM, 8 on AVX2, 16 on AVX-512).
        // Because SIMDVector3D is aligned to NATIVE_SIMD_BATCH_ALIGN, std::vector will perfectly pack these into sequential CPU cache lines.
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

    // --- 1. PHYSICS UPDATE (chew through millions of particles)---
    FORCE_INLINE void UpdateParticles(std::span<SIMDVector3D> positions, 
                                      std::span<SIMDVector3D> velocities, 
                                      size_t activeCount, 
                                      float deltaTime) {
        // 1. Broadcast the scalar delta time into a hardware SIMD register ONCE.                                  
        NativeFloatSIMDBatch dtBatch = deltaTime; 

        // 2. Iterate over the batches (NOT individual particles)
        size_t activeBatches = (activeCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

        // We can iterate directly over the span size up to the active batch count.
        for (size_t i = 0; i < activeBatches; ++i) {
            // 3. Load velocities into registers
            NativeFloatSIMDBatch velX = velocities[i].x;
            NativeFloatSIMDBatch velY = velocities[i].y;
            NativeFloatSIMDBatch velZ = velocities[i].z;

            // 4. Calculate movement: Velocity * DeltaTime
            velX *= dtBatch;
            velY *= dtBatch;
            velZ *= dtBatch;

            // 5. Apply Fused Multiply-Add (Position = Position + (Velocity * dt)),  maps to hardware vector addition.
            positions[i].add(velX, velY, velZ);

            // Notice there are NO if-statements, NO branching, and NO function overhead.
            // The hardware prefetcher will detect this linear memory access pattern immediately and stream the L1 cache ahead of the CPU's execution ports.
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

            // Engine Custom SIMD Casting: cast SIMD Floats to SIMD Integers
            NativeUIntBatch gridX = Engine::ISAArch::simd_cast<uint32_t>(gridX_f);
            NativeUIntBatch gridY = Engine::ISAArch::simd_cast<uint32_t>(gridY_f);
            NativeUIntBatch gridZ = Engine::ISAArch::simd_cast<uint32_t>(gridZ_f);

            // 4. Generate Morton Codes across all lanes simultaneously (BMI2 / LUT)
            NativeUIntBatch mortonBatch = getMortonCode_SIMD(gridX, gridY, gridZ);

            // 5. The Bridge: Flush SIMD register directly to memory and a temporary aligned array.
            alignas(NATIVE_SIMD_BATCH_ALIGN) uint32_t tempMortons[NATIVE_BATCH_SIZE];
            mortonBatch.copy_to(tempMortons /*, std::element_aligned*/);

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
    FORCE_INLINE void RadixSortKeys(std::span<ParticleSortKey> keys, 
                                    std::span<ParticleSortKey> buffer) {
                                        
        ParticleSortKey* src = keys.data();
        ParticleSortKey* dst = buffer.data();
        size_t count = keys.size();

        // By processing 8-bits at a time, we only need 4 passes to perfectly sort 32-bit integers.
        // 4 passes: 0, 8, 16, 24 (to cover all 32 bits of the Morton Code)
        for (int shift = 0; shift < 32; shift += 8) {

            // 8 bits = 256 possible buckets
            uint32_t counts[256] = {0}; 

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
    FORCE_INLINE NativeUIntBatch GetLaneIndices() {
        alignas(NATIVE_SIMD_BATCH_ALIGN) uint32_t indices[NATIVE_BATCH_SIZE];

        // Ensures we only check particles "forward" in the array, so they don't collide with themsleves or apply collision forces twice.
        for (uint32_t i = 0; i < NATIVE_BATCH_SIZE; ++i) {
            indices[i] = i;
        }

        // NativeUIntBatch batch;
        // batch.copy_from(indices, std::element_aligned);
        // return batch;
        
        // Native pointer constructor
        return NativeUIntBatch(indices);
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

        // Because the array is sorted by spatial proximity (Morton Codes), we only need to look ahead a few batches to find all physical neighbors.
        // Adjust this based on your CELL_SIZE and particle density (2 to 4 is usually plenty).
        const size_t SCAN_WINDOW = 2; 

        // Outer Loop: Iterate through every particle linearly
        for (size_t i = 0; i < activeCount; ++i) {
            
            // 1. Locate the Reference Particle
            size_t batchI = i / NATIVE_BATCH_SIZE;
            size_t laneI  = i % NATIVE_BATCH_SIZE;

            // 2. Broadcast the scalar (X, Y, Z) into full SIMD registers
            NativeFloatSIMDBatch refX = positions[batchI].x[laneI]; // refX becomes { X, X, X, X, X, X, X, X }
            NativeFloatSIMDBatch refY = positions[batchI].y[laneI]; // refY becomes { Y, Y, Y, Y, Y, Y, Y, Y }
            NativeFloatSIMDBatch refZ = positions[batchI].z[laneI]; // refZ becomes { Z, Z, Z, Z, Z, Z, Z, Z }

            // We accumulate the push-back forces for the reference particle locally
            float moveX = 0.0f, moveY = 0.0f, moveZ = 0.0f;

            size_t endBatch = std::min(batchI + SCAN_WINDOW + 1, activeBatches);
            
            // 3. Inner Loop: Scan Forward through neighbor batches
            for (size_t batchJ = batchI; batchJ < endBatch; ++batchJ) {
                
                // --- SIMD DISTANCE CALCULATION ---
                NativeFloatSIMDBatch dx = positions[batchJ].x - refX;
                NativeFloatSIMDBatch dy = positions[batchJ].y - refY;
                NativeFloatSIMDBatch dz = positions[batchJ].z - refZ;

                // FMA Distance Squared
                NativeFloatSIMDBatch distSq = dx * dx;
                distSq = fma(dy, dy, distSq);
                distSq = fma(dz, dz, distSq);

                // --- THE MASK GENERATOR ---
                // 1. Are they touching? (distSq < diameterSq)
                auto spatialMask = (distSq < diameterSq);

                // 1. Calculate the scalar offset and strictly cast it to uint32_t
                uint32_t batchOffset = static_cast<uint32_t>(batchJ * NATIVE_BATCH_SIZE);

                // 3. Prevent double-resolving and self-resolving! Calculate the absolute index of every particle in Batch J
                // Broadcast the scalar into a SIMD batch
                NativeUIntBatch absoluteJ = NativeUIntBatch(batchOffset) + laneIndices;
                
                // Only apply physics if the neighbor's index is strictly greater than i
                auto validCollisionMask = spatialMask && (absoluteJ > static_cast<uint32_t>(i));
                // auto validCollisionMask = spatialMask && (absoluteJ > i).cast_to<float>();

                // --- 2. BREAKING PERFECT OVERLAP SYMMETRY ---
                // Detect particles that are occupying the exact same space.
                auto perfectOverlapMask = validCollisionMask && (distSq < epsilon);

                // Artificially nudge overlapping particles along the X-axis by 0.1mm (1e-4f). 
                // This gives the physics solver a non-zero vector to calculate a push direction.
                Engine::ISAArch::where(perfectOverlapMask, dx) = 1e-4f;
                Engine::ISAArch::where(perfectOverlapMask, distSq) = 1e-8f; // (1e-4f)^2

                // --- SIMD PENETRATION RESOLUTION ---
                // Safely overwrite non-colliding lanes to diameterSq (penetration becomes 0.0f)
                Engine::ISAArch::where(!validCollisionMask, distSq) = diameterSq;

                // Math: penetration = diameter - sqrt(distSq), actualDist can never be zero
                NativeFloatSIMDBatch actualDist = sqrt(distSq);
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
                Engine::ISAArch::where(validCollisionMask, positions[batchJ].x) += pushX;
                Engine::ISAArch::where(validCollisionMask, positions[batchJ].y) += pushY;
                Engine::ISAArch::where(validCollisionMask, positions[batchJ].z) += pushZ;

                // 2. Accumulate the opposite push for our Reference Particle
                // We use the portable reduction (summing up the valid lanes)
                // automatically ignores the lanes where validCollisionMask was false
                // because pushX/Y/Z are exactly 0.0f in those lanes due to our penetration math.
                moveX -= reduce(pushX);
                moveY -= reduce(pushY);
                moveZ -= reduce(pushZ);
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

        // 5. Bypass Initialization. Since ReorderParticleData will immediately overwrite every single byte, zeroing out the memory first would waste CPU cycles.
        tempPos.ResizeUninitialized(activeBatches);
        tempVel.ResizeUninitialized(activeBatches);

        // =========================================================================

        // 6. Reorder data into our ultra-fast L1 cache aligned transient memory.
        ReorderParticleData(posSpan, velSpan, tempPos, tempVel, keySpan);

        // 7. Commit the perfectly sorted data back to main memory.
        std::copy(tempPos.begin(), tempPos.end(), memory.positions.begin());
        std::copy(tempVel.begin(), tempVel.end(), memory.velocities.begin());

        // 8. Resolve collisions
        ResolveCollisions(posSpan, memory.activeParticleCount, 2.0f);

        // t_PhysicsTransientArena.SetOffset(m_savedOffset) is automatically called.
        // The memory used by tempPos and tempVel is instantly "freed" in zero clock cycles.
    }
}
