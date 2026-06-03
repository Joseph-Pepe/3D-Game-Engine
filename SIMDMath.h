#pragma once

#include "Math.h"
#include "Memory.h" // Ensure AlignedVector is included!

#include <vector>
#include <execution>
#include <ranges>
#include <algorithm>
#include <immintrin.h>

// A "Manager" class for a large collection of vectors (processes 4 vectors simultaneously).
class VectorManagerSOA {
public:
    // Aligned vectors for X, Y, and Z components
    std::vector<float> xs, ys, zs;

    VectorManagerSOA(size_t count) {
        // Ensure count is a multiple of 4
        size_t paddedCount = (count + 3) & ~3;
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        __m128 sX = _mm_set1_ps(stepX);
        __m128 sY = _mm_set1_ps(stepY);
        __m128 sZ = _mm_set1_ps(stepZ);

        for (size_t i = 0; i < xs.size(); i += 4) {
            // 1. Load 4 vectors at once
            __m128 vX = _mm_load_ps(&xs[i]);
            __m128 vY = _mm_load_ps(&ys[i]);
            __m128 vZ = _mm_load_ps(&zs[i]);

            // 2. Addition (4 vectors at once!)
            vX = _mm_add_ps(vX, sX);
            vY = _mm_add_ps(vY, sY);
            vZ = _mm_add_ps(vZ, sZ);

            // 3. Dot Product (SOA dot product is just 3 muls and 2 adds)
            // d = (x1*x2) + (y1*y2) + (z1*z2)
            __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vX, sX), 
                                               _mm_mul_ps(vY, sY)), 
                                    _mm_mul_ps(vZ, sZ));

            // 4. Update X (A.x += d * small)
            vX = _mm_add_ps(vX, _mm_mul_ps(dot, _mm_set1_ps(0.00001f)));

            // 5. Cross Product (The SOA "Magic")
            // Res.x = (y * b.z) - (z * b.y)
            __m128 resX = _mm_sub_ps(_mm_mul_ps(vY, sZ), _mm_mul_ps(vZ, sY));
            __m128 resY = _mm_sub_ps(_mm_mul_ps(vZ, sX), _mm_mul_ps(vX, sZ));
            __m128 resZ = _mm_sub_ps(_mm_mul_ps(vX, sY), _mm_mul_ps(vY, sX));

            // 6. Store 4 results back
            _mm_store_ps(&xs[i], resX);
            _mm_store_ps(&ys[i], resY);
            _mm_store_ps(&zs[i], resZ);
        }
    }
};

// --- THE SYSTEM LAYER (8-Wide Alignment) Processes 8 vectors simultaneously ---
class VectorManagerSOA_V2_AVX2 {
public:

    // Use the custom 32-byte aligned allocator!
    AlignedVector32<float> xs, ys, zs;

    VectorManagerSOA_V2_AVX2(size_t count) {
        size_t paddedCount = (count + 7) & ~7;
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    // [PROCESSBATCH]: PROCESSES 200 MILLION OPERATIONS IN 83 MILLISECONDS, CRUNCHES OUT ROUGHLY 2.4 BILLION VECTOR OPERATIONS PER SECOND.
    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        // Broadcast the scalars into all 8 slots of the 256-bit registers
        __m256 sX = _mm256_set1_ps(stepX);
        __m256 sY = _mm256_set1_ps(stepY);
        __m256 sZ = _mm256_set1_ps(stepZ);
        __m256 smallVal = _mm256_set1_ps(0.00001f);

        uint32_t dataCount = static_cast<uint32_t>(xs.size());
    
        // 1. Calculate a cache-friendly chunk size (target ~8192 to 32768 elements per job)
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t targetChunksPerThread = 16;
        uint32_t CHUNK_SIZE = dataCount / (threadCount * targetChunksPerThread);
        CHUNK_SIZE = std::clamp(CHUNK_SIZE, 8192u, 32768u);
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7; // Ensure AVX2 8-float alignment

        // 2. Dispatch to the Coroutine Work-Stealing Queue
        g_JobSystem.DispatchAndWait(dataCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Ensure boundaries are strictly aligned to prevent AVX load crashes
            uint32_t alignedStart = start & ~7;
            uint32_t alignedEnd = (end + 7) & ~7;

            for (int i = alignedStart; i < alignedEnd; i += 8) {

                // Prefetch 64 floats (256 bytes / 4 cache lines) ahead. Only use this code strictly for when your jumping around memory randomly.
                // _mm_prefetch((const char*)(xPtr + i + 64), _MM_HINT_T0);
                // _mm_prefetch((const char*)(yPtr + i + 64), _MM_HINT_T0);
                // _mm_prefetch((const char*)(zPtr + i + 64), _MM_HINT_T0);


                // LOAD: Bring data into our math wrapper
                // 1. Load 8 vectors at once (32 bytes per load)
                SIMDVector8 batch = { 
                    // [_mm256_load_ps]: This demands that memory is 32-byte aligned (std::vector only guarantees 16-byte alignment).
                    // We are now guaranteed 32-byte alignment. 
                    // We use _mm256_load_ps (Aligned) instead of _mm256_loadu_ps (Unaligned)
                    _mm256_load_ps(&xs[i]), 
                    _mm256_load_ps(&ys[i]), 
                    _mm256_load_ps(&zs[i]) 
                    // [_mm256_loadu_ps]: The hardware is smart because using it on memory that happens to be aligned incurs zero performance penalty. 
                    // Is the safest way to write SIMD code eithout dealing with custom memory allocators (i.e., prevents any 32-byte memory alignment crashes). 
                    // _mm256_loadu_ps(&xs[i]), 
                    // _mm256_loadu_ps(&ys[i]), 
                    // _mm256_loadu_ps(&zs[i])
                };

                // MATH: Using our clean functions
                 // 2. Addition (8 at once)
                batch.add(sX, sY, sZ);

                // 3. Dot Product (Still pure vertical math)
                __m256 d = batch.dot_fma(sX, sY, sZ);
                
                // Single element update (x += d * small)
                // 4. Update X
                batch.x = _mm256_add_ps(batch.x, _mm256_mul_ps(d, smallVal));
                
                 // 5. Cross Product (8 simultaneous cross products)
                batch.cross(sX, sY, sZ);

                // STORE: Write back to main memory
                // Aligned stores directly to the Write-Combine buffer
                // 6. Store 8 results back
                _mm256_store_ps(&xs[i], batch.x);
                _mm256_store_ps(&ys[i], batch.y);
                _mm256_store_ps(&zs[i], batch.z);
            }

            // Clean the worker CPU registers before handing the thread back to the scheduler
            _mm256_zeroupper(); // this must be the final instruction executed before returning whenever using bare-metal SIMD using _mm256 registers inside lambdas.
        });

        // Clean the Caller Thread before returning to the SSE benchmark!
        _mm256_zeroupper();
    }
};

// --- 2. THE SYSTEM LAYER (16-Wide Alignment) ---
class VectorManagerSOA_V3_AVX512 {
public:
    // Using our 64-byte aligned vectors
    AlignedVector64<float> xs, ys, zs;

    VectorManagerSOA_V3_AVX512(size_t count) {
        // Pad to nearest multiple of 16 for AVX-512 boundaries
        size_t paddedCount = (count + 15) & ~15;
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        __m512 sX = _mm512_set1_ps(stepX);
        __m512 sY = _mm512_set1_ps(stepY);
        __m512 sZ = _mm512_set1_ps(stepZ);
        __m512 smallVal = _mm512_set1_ps(0.00001f);

        uint32_t dataCount = static_cast<uint32_t>(xs.size());

        // 1. Calculate a cache-friendly chunk size
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t targetChunksPerThread = 16;
        uint32_t CHUNK_SIZE = dataCount / (threadCount * targetChunksPerThread);
        
        // Clamp to L1/L2 friendly sizes
        CHUNK_SIZE = std::clamp(CHUNK_SIZE, 8192u, 32768u);
        
        // CRITICAL: Ensure AVX-512 16-float (64-byte) alignment
        CHUNK_SIZE = (CHUNK_SIZE + 15) & ~15;

        // 2. Dispatch to the Coroutine Work-Stealing Queue
        g_JobSystem.DispatchAndWait(dataCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Ensure boundaries are strictly aligned to prevent AVX-512 load crashes
            uint32_t alignedStart = start & ~15;
            uint32_t alignedEnd = (end + 15) & ~15;

            for(uint32_t i = alignedStart; i < alignedEnd; i += 16) {
                // ALIGNED LOAD: We are guaranteed 64-byte alignment (512-bit memory)
                // Custom Allocator: We can now safely use the faster aligned loads (_mm512_load_ps) and stores!
                SIMDVector16 batch = { 
                    _mm512_load_ps(&xs[i]), 
                    _mm512_load_ps(&ys[i]), 
                    _mm512_load_ps(&zs[i]) 
                };

                // MATH
                batch.add(sX, sY, sZ);
                __m512 d = batch.dot_fma(sX, sY, sZ);
                batch.x = _mm512_add_ps(batch.x, _mm512_mul_ps(d, smallVal));
                batch.cross(sX, sY, sZ);

                // ALIGNED STORE: Direct write-back
                _mm512_store_ps(&xs[i], batch.x);
                _mm512_store_ps(&ys[i], batch.y);
                _mm512_store_ps(&zs[i], batch.z);
            }

            // Clean the CPU registers before handing the thread back to the scheduler
            _mm256_zeroupper();
        });
        // Clean the Caller Thread before returning
        _mm256_zeroupper();
    }
};

// ======================================================================
// C++26: SIMD for MSVC build v14.51 and newer.
// ======================================================================

// --- THE SYSTEM LAYER (Portable Vector Manager, Portable SIMD, Any Hardware) ---
class VectorManagerSOA_Portable {
public:
    // Memory is perfectly aligned for the target hardware!
    NativeAlignedVector<float> xs, ys, zs;

    VectorManagerSOA_Portable(size_t count) {
        constexpr size_t stride = native_simd::size();
        
        // Dynamic padding: hardware-agnostic boundary alignment
        size_t remainder = count % stride;
        size_t paddedCount = (remainder == 0) ? count : count + (stride - remainder);
        
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        constexpr size_t stride = native_simd::size();
        
        // Broadcast scalars directly into the portable SIMD types
        native_simd sX(stepX);
        native_simd sY(stepY);
        native_simd sZ(stepZ);
        native_simd smallVal(0.00001f);

        // 1. Extract raw pointers to prevent 'this' aliasing in the parallel loop
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

// ======================================================================================
// ARRAY OF STRUCTS OF ARRAYS (AOSOA)
// ======================================================================================
/*
    - Structure of Arrays (SOA) abuses the CPU's hardware prefetcher. 
    - It has three disconnected arrays vector<float> pX, pY, pZ.
    - For SOA, the CPU's memory controller has to track three separate read streams, and write streams simultaneously causing cache thrashing.
    - AOSOA fixes this by grouping 8 particles into a single 96-byte chunk.
    - The prefetcher now only tracks one continguous memory stream.
    - 8% to 15% increase in GB/s throughput and memory bandwidth and reduces idling.
*/

class VectorManagerAoSoA_AVX2 {
public:
    // Uses your custom allocator to guarantee the entire heap array starts on a 32-byte boundary!
    AlignedVector32<ParticleBlock8> blocks;

    VectorManagerAoSoA_AVX2(size_t particleCount) {
        // Divide by 8, rounding up to get total blocks
        size_t blockCount = (particleCount + 7) / 8;
        blocks.resize(blockCount);

        // Initialization 
        for(auto& b : blocks) {
            b.x = _mm256_set1_ps(1.0f);
            b.y = _mm256_set1_ps(2.0f);
            b.z = _mm256_set1_ps(3.0f);
        }
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        __m256 sX = _mm256_set1_ps(stepX);
        __m256 sY = _mm256_set1_ps(stepY);
        __m256 sZ = _mm256_set1_ps(stepZ);
        __m256 smallVal = _mm256_set1_ps(0.00001f);

        uint32_t blockCount = static_cast<uint32_t>(blocks.size());

        // 1. Thread Chunking (Chunks are now counted in BLOCKS, not single floats)
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t targetChunksPerThread = 16;
        uint32_t CHUNK_SIZE = std::max(1024u, blockCount / (threadCount * targetChunksPerThread));
        // No need to bitwise pad `& ~7` here, because every index is naturally an 8-wide block!

        g_JobSystem.DispatchAndWait(blockCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
            
            for (uint32_t i = start; i < end; ++i) {
                // 1. ALIGNED LOAD: Zero pointer math. Zero straddling.
                // The compiler translates this directly into ultra-fast `vmovaps` instructions.
                SIMDVector8 batch = { blocks[i].x, blocks[i].y, blocks[i].z };

                // 2. MATH
                batch.add(sX, sY, sZ);
                __m256 d = batch.dot_fma(sX, sY, sZ);
                batch.x = _mm256_add_ps(batch.x, _mm256_mul_ps(d, smallVal));
                batch.cross(sX, sY, sZ);

                // 3. ALIGNED STORE: Direct write-back to L1 cache
                blocks[i].x = batch.x;
                blocks[i].y = batch.y;
                blocks[i].z = batch.z;
            }

            // Clean the YMM registers
            _mm256_zeroupper(); 
        });
    }
};
