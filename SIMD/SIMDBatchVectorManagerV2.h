#pragma once

#include "SIMD/SIMDCustomWrapper.h"
#include "../JobSystem.h"
#include <vector>
#include <execution>
#include <algorithm>
#include <cstdlib>
#include <new>     // C++17/26 hardware interference sizes
#include <memory>  // C++20/26 std::assume_aligned

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
        std::vector<PortableParticleBlock, DynamicAlignedAllocator<PortableParticleBlock, NATIVE_HARDWARE_ALIGNMENT>> blocks;

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
            //     SIMDVectorP batch = { 
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
                    // We use _loadu_ps (Unaligned) instead of _load_ps (Aligned)
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
        - We group 256 particles into a single block. 
        - 256 floats = 1024 bytes per axis. Total size = 3072 bytes.
        - 3072 is a perfect multiple of 64 (L1 Cache Line). No 'w' padding required.
        - 3072 bytes fits entirely inside a standard 4KB OS Memory Page. Zero TLB Thrashing!
        - Writing sequentially to x[0], x[1]... perfectly saturates the CPU Write-Combine buffer.
    */
    // constexpr size_t MACRO_CHUNK_PARTICLES = 256;
    // constexpr size_t REGISTERS_PER_CHUNK = MACRO_CHUNK_PARTICLES / WideFloat::size();

    // struct alignas(L1_CACHE_CHUNK_SIZE) PortableParticleMacroChunk {
    //     WideFloat x[REGISTERS_PER_CHUNK];
    //     WideFloat y[REGISTERS_PER_CHUNK];
    //     WideFloat z[REGISTERS_PER_CHUNK];

    //     // The math is localized to operate on specific register lanes within the chunk
    //     FORCE_INLINE void process_lane(size_t j, const WideFloat& sX, const WideFloat& sY, const WideFloat& sZ, const WideFloat& smallVal) {
    //         // 1. Addition
    //         x[j] += sX;
    //         y[j] += sY;
    //         z[j] += sZ;

    //         // 2. FMA Dot Product
    //         WideFloat res = x[j] * sX;
    //         res = fma(y[j], sY, res);
    //         res = fma(z[j], sZ, res);

    //         // 3. Scale
    //         x[j] += res * smallVal;

    //         // 4. Cross Product (Reading locally avoids register spilling)
    //         WideFloat rx = fma(y[j], sZ, -(z[j] * sY));
    //         WideFloat ry = fma(z[j], sX, -(x[j] * sZ));
    //         WideFloat rz = fma(x[j], sY, -(y[j] * sX));
            
    //         x[j] = rx; 
    //         y[j] = ry; 
    //         z[j] = rz;
    //     }
    // };

    // ======================================================================
    // THE MACRO-CHUNK (The Perfect AoSoA Memory Layout)
    // ======================================================================    
    // The "Sweet Spot" Macro Chunk (Perfect Cache Line Multiples, No Padding)
    // AVX2 = 32 particles. AVX-512 = 64 particles.
    constexpr size_t REGISTERS_PER_CHUNK = 4; 

    // Particles per chunk now scales natively! 
    // AVX2 = 4 * 8 = 32. NEON = 4 * 4 = 16. AVX-512 = 4 * 16 = 64.
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
        std::vector<OptimalParticleMacroChunk, DynamicAlignedAllocator<OptimalParticleMacroChunk, L1_CACHE_CHUNK_SIZE>> chunks;

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
