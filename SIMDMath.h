#pragma once

#include "Math.h"

// --- THE SYSTEM LAYER (8-Wide Alignment) Processes 8 vectors simultaneously ---
class VectorManagerSOA_V2_AVX2 {
public:
    std::vector<float> xs, ys, zs;

    VectorManagerSOA_V2_AVX2(size_t count) {
        size_t paddedCount = (count + 7) & ~7;
        xs.resize(paddedCount, 1.0f);
        ys.resize(paddedCount, 2.0f);
        zs.resize(paddedCount, 3.0f);
    }

    FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
        __m256 sX = _mm256_set1_ps(stepX);
        __m256 sY = _mm256_set1_ps(stepY);
        __m256 sZ = _mm256_set1_ps(stepZ);
        __m256 smallVal = _mm256_set1_ps(0.00001f);

        uint32_t dataCount = xs.size();
    
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
                SIMDVector8 batch = { 
                    // [_mm256_load_ps]: This demands that memory is 32-byte aligned (std::vector only guarantees 16-byte alignment).

                    /* 
                    _mm256_load_ps(&xs[i]), 
                    _mm256_load_ps(&ys[i]), 
                    _mm256_load_ps(&zs[i]) 
                    */
                    // [_mm256_loadu_ps]: The hardware is smart because using it on memory that happens to be aligned incurs zero performance penalty. 
                    // Is the safest way to write SIMD code eithout dealing with custom memory allocators (i.e., prevents any 32-byte memory alignment crashes). 
                    _mm256_loadu_ps(&xs[i]), 
                    _mm256_loadu_ps(&ys[i]), 
                    _mm256_loadu_ps(&zs[i])
                };

                // MATH: Using our clean functions
                batch.add(sX, sY, sZ);
                __m256 d = batch.dot_fma(sX, sY, sZ);
                
                // Single element update (x += d * small)
                batch.x = _mm256_add_ps(batch.x, _mm256_mul_ps(d, smallVal));
                
                batch.cross(sX, sY, sZ);

                // STORE: Write back to main memory
                _mm256_storeu_ps(&xs[i], batch.x);
                _mm256_storeu_ps(&ys[i], batch.y);
                _mm256_storeu_ps(&zs[i], batch.z);
            }

            // Clean the worker CPU registers before handing the thread back to the scheduler
            // _mm256_zeroupper(); // this must be the final instruction executed before returning whenever using bare-metal SIMD using _mm256 registers inside lambdas.
        });

        // Clean the Caller Thread before returning to the SSE benchmark!
        _mm256_zeroupper();
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
    AlignedVector<ParticleBlock8> blocks;

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

        uint32_t blockCount = blocks.size();

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
