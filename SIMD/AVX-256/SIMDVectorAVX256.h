#pragma once

#include "../../Memory.h" // Ensure AlignedVector is included!

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

// ==================================================================================
// AVX2 BARE-METAL SIMD STRUCTURES (8-Wide)
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

// 96 Bytes total: Fits perfectly into two 64-byte L1 Cache lines.
struct alignas(32) ParticleBlock8 {
    __m256 x;
    __m256 y;
    __m256 z;
};

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
