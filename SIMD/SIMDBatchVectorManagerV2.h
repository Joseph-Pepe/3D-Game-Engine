#pragma once

#include "SIMD/SIMDCustomWrapper.h"
#include "../JobSystem.h"
#include <vector>
#include <execution>
#include <algorithm>
#include <cstdlib>

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

    FORCE_INLINE void clear_registers() {
        #if defined(ENGINE_ARCH_AVX2) || defined(ENGINE_ARCH_AVX512)
            _mm256_zeroupper(); // Or _mm512_zeroupper on some older Xeon architectures
        #endif
        // ARM NEON and SSE4.1 do not suffer from state-transition penalties, so they do nothing.
    }

    // ======================================================================
    // CUSTOM ALIGNED ALLOCATOR
    // ======================================================================
    template <typename T, std::size_t Alignment>
    struct DynamicAlignedAllocator {
        using value_type = T;

        template <class U>
        struct rebind {
            using other = DynamicAlignedAllocator<U, Alignment>;
        };

        DynamicAlignedAllocator() noexcept = default;
        template <typename U> DynamicAlignedAllocator(const DynamicAlignedAllocator<U, Alignment>&) noexcept {}

        T* allocate(std::size_t n) {
            if (n == 0) return nullptr;
            void* ptr = nullptr;
            #if defined(_MSC_VER)
                ptr = _aligned_malloc(n * sizeof(T), Alignment);
            #else
                if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
            #endif
            if (!ptr) throw std::bad_alloc();
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, std::size_t) noexcept {
            #if defined(_MSC_VER)
                _aligned_free(p);
            #else
                free(p);
            #endif
        }
    };

    template<typename T>
    using NativeAlignedVector = std::vector<T, DynamicAlignedAllocator<T, alignof(WideBatch<T>)>>;

    template<typename T>
    using FixedAlignedVector = std::vector<T, DynamicAlignedAllocator<T, alignof(FixedBatch4<T>)>>;


    // ======================================================================
    // 1. THE AOSOA BLOCK (The Cache-Friendly Data Chunk)
    // ======================================================================
    // Force the struct itself to align to 32-bytes (3x32 = 96-bytes), not 128-bytes.
    struct alignas(alignof(WideFloat)) PortableParticleBlock {
        WideFloat x; // 32-bytes
        WideFloat y; // 32-bytes
        WideFloat z; // 32-bytes
        // WideFloat w; // By adding w as padding to force a 128-byte alignment, it will tank performance because it needs to move 33% more memory on every single iteration (CPU load and stores).

        FORCE_INLINE void add(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) {
            x += bx;
            y += by;
            z += bz;
        }

        FORCE_INLINE WideFloat dot_fma(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) const {
            WideFloat res = x * bx;
            res = fma(y, by, res);
            res = fma(z, bz, res);
            return res;
        }

        FORCE_INLINE void cross(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) {
            WideFloat rx = fma(y, bz, -(z * by));
            WideFloat ry = fma(z, bx, -(x * bz));
            WideFloat rz = fma(x, by, -(y * bx));
            x = rx; y = ry; z = rz;
        }
    };

    // ======================================================================
    // 2. THE MANAGER (Auto-Vectorizing, Multi-Threaded, Cross-Platform)
    // ======================================================================
    class VectorManagerAoSoA_Portable {
    public:
        std::vector<PortableParticleBlock, DynamicAlignedAllocator<PortableParticleBlock, alignof(WideFloat)>> blocks;

        VectorManagerAoSoA_Portable(size_t particleCount) {
            size_t blockCount = (particleCount + WideFloat::size() - 1) / WideFloat::size();
            blocks.resize(blockCount);

            for (auto& b : blocks) {
                b.x = WideFloat(1.0f);
                b.y = WideFloat(2.0f);
                b.z = WideFloat(3.0f);
                // b.w = WideFloat(0.0f);
            }
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            WideFloat sX(stepX);
            WideFloat sY(stepY);
            WideFloat sZ(stepZ);
            WideFloat smallVal(0.00001f);

            uint32_t blockCount = static_cast<uint32_t>(blocks.size());
            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t targetChunksPerThread = 16;
            
            // Calculate an optimal chunk size to keep L1/L2 caches happy
            uint32_t CHUNK_SIZE = std::max(1024u, blockCount / (threadCount * targetChunksPerThread));

            g_JobSystem.DispatchAndWait(blockCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
                for (uint32_t i = start; i < end; ++i) {
                    
                    // 1. LOAD: Copy from heap memory into CPU registers (No '&' reference!)
                    PortableParticleBlock batch = blocks[i]; 
                    
                    // 2. MATH: Pure register execution (Zero RAM access)
                    batch.add(sX, sY, sZ);
                    WideFloat d = batch.dot_fma(sX, sY, sZ);
                    batch.x += d * smallVal;
                    batch.cross(sX, sY, sZ);

                    // Emits a generic memory copy rather than aligned AVX stores, better to break it up by parts.
                    // blocks[i] = batch;

                    // 3. STORE: Write the final computed result back to memory, break it apart to match AVX2 speed.
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
        NativeAlignedVector<float> xs, ys, zs;

        // AlignedVector32<float> xs, ys, zs;           // AVX-256: 32-byte aligned vectors
        // AlignedVector64<float> xs, ys, zs;           // AVX-512: 64-byte aligned vectors

        VectorManagerSOA_Portable(size_t count) {
            // size_t paddedCount = (count + 15) & ~15; // Pad to nearest multiple of 16 for AVX-512 boundaries
            // size_t paddedCount = (count + 7) & ~7;   // Pad to nearest multiple of  8 for AVX-256 boundaries
             
            constexpr size_t stride = WideFloat::size();
            size_t remainder = count % stride;
            size_t paddedCount = (remainder == 0) ? count : count + (stride - remainder);
            
            xs.resize(paddedCount, 1.0f);
            ys.resize(paddedCount, 2.0f);
            zs.resize(paddedCount, 3.0f);
        }

        struct SOA_Batch {
            WideFloat x, y, z;
            FORCE_INLINE void add(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) { x += bx; y += by; z += bz; }
            FORCE_INLINE WideFloat dot_fma(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) const {
                WideFloat res = x * bx; res = fma(y, by, res); res = fma(z, bz, res); return res;
            }
            FORCE_INLINE void cross(const WideFloat& bx, const WideFloat& by, const WideFloat& bz) {
                WideFloat rx = fma(y, bz, -(z * by)); WideFloat ry = fma(z, bx, -(x * bz)); WideFloat rz = fma(x, by, -(y * bx));
                x = rx; y = ry; z = rz;
            }
        };

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            constexpr size_t stride = WideFloat::size();
        
            // Broadcast the scalars into all [8 slots of the 256-bit registers (AVX-256)] (or) all [16 slots of the 512-bit registers (AVX-512)].
            // AVX-256: [32-bits (float) x  8 slots (vectors) = 256-bits] 
            // AVX-512: [32-bits (float) x 16 slots (vectors) = 512-bits]
            WideFloat sX(stepX);           // AVX-256: __m256 sX       = _mm256_set1_ps(stepX);    AVX-512: __m512 sX       = _mm512_set1_ps(stepX);
            WideFloat sY(stepY);           // AVX-256: __m256 sY       = _mm256_set1_ps(stepY);    AVX-512: __m512 sY       = _mm512_set1_ps(stepY);
            WideFloat sZ(stepZ);           // AVX-256: __m256 sZ       = _mm256_set1_ps(stepZ);    AVX-512: __m512 sZ       = _mm512_set1_ps(stepZ);
            WideFloat smallVal(0.00001f);  // AVX-256: __m256 smallVal = _mm256_set1_ps(0.00001f); AVX-512: __m512 smallVal = _mm512_set1_ps(0.00001f);

            float* ptrX = xs.data();
            float* ptrY = ys.data();
            float* ptrZ = zs.data();

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
                    // We use _loadu_ps (Unaligned) instead of _load_ps (Aligned)
                    SOA_Batch batch = { 
                        WideFloat(ptrX + i),  // AVX-256: _mm256_loadu_ps(&xs[i]);  AVX-512: _mm512_loadu_ps(&xs[i]);
                        WideFloat(ptrY + i),  // AVX-256: _mm256_loadu_ps(&ys[i]);  AVX-512: _mm512_loadu_ps(&ys[i]);
                        WideFloat(ptrZ + i)   // AVX-256: _mm256_loadu_ps(&zs[i]);  AVX-512: _mm512_loadu_ps(&zs[i]);
                    };

                    // 2. Addition (AVX-256: 8 at once, AVX-512: 16 at once)
                    batch.add(sX, sY, sZ);    // AVX-256: x = _mm256_add_ps(x, bx);  AVX-512: x = _mm512_add_ps(x, bx);
                                              // AVX-256: y = _mm256_add_ps(y, by);  AVX-512: y = _mm512_add_ps(y, by);
                                              // AVX-256: z = _mm256_add_ps(z, bz);  AVX-512: z = _mm512_add_ps(z, bz);

                    // 3. Dot Product
                    WideFloat d = batch.dot_fma(sX, sY, sZ);

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

}
