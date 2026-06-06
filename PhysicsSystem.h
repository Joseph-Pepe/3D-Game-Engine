#pragma once

#include <vector>
#include <atomic>
#include <immintrin.h> // For AVX2/AVX-512

#include <span>       // REQUIRED for std::span
#include <algorithm>  // REQUIRED for std::fill, std::clamp, std::max, std::min
#include <cmath>      // REQUIRED for std::cos, std::sin

// Engine Dependencies
#include "Memory.h"    // For AlignedVector
#include "JobSystem.h" // For parallel dispatch and thread IDs
#include "Math.h"      // For Morton codes and vector math
#include "Hardware.h"  // For AVX availability checks
#include "EngineSettings.h"

// Cross-platform restrict macro for pointer aliasing guarantees
#if defined(_MSC_VER)
    #define ENGINE_RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
    #define ENGINE_RESTRICT __restrict__
#else
    #define ENGINE_RESTRICT
#endif

class ParticlePhysicsSOA {
public:
    // C++20 Spans are lightweight, zero cost abstractions that act exactly like arrays, but does not try to free() its memory when it goes out of scope.

    // Component 1: Positions (Strictly 32-byte aligned!)
    // AlignedVector32<float> pX, pY, pZ;
    std::span<float> pX, pY, pZ;
    
    // Component 2: Velocities
    // AlignedVector32<float> vX, vY, vZ;
    std::span<float> vX, vY, vZ;

    // =============================================================
    // CONTIGUOUS FLAT 1D ARRAY
    // =============================================================
    /*
        - std::vector<std::vector<uint32_t>> threadLocalCounts is a list of pointers pointing to separate, fragmented allocations.
        - An array of pointers pointing to random locations is bad for the CPU hardware prefetcher b/c it has to fetch random pages from RAM.

        - std::vector<uint32_t> is a single contiguous array.
        - All threads will write into the same massive block of memory.
        - Its mathematically separated by offsets.
    */

    // Histogram per potential thread (maxQueues, Flat 1D Array for contiguous cache locality)
    std::span<uint32_t> threadLocalCounts;

    // Saves the destination indices (i.e., allows the prefetcher to predict where to go next)
    std::span<uint32_t> temp_destIndices;

    // --- SPATIAL GRID SETTINGS ---
    // Assuming our galaxy spans roughly -1000 to +1000 in X and Y
    static constexpr float WORLD_SIZE = 2000.0f; 
    static constexpr float CELL_SIZE = 4.0f; // Tune this: particles can only collide within 4 units (CELL_SIZE = COLLISION_RADIUS), Match exactly to RADIUS, Search Area
    static constexpr float INV_CELL_SIZE = 1.0f / CELL_SIZE; // multiply this inverse (4 clock cycles per floating-point) to save millions of wasted cycles on main thread instead of division (10-15 clock cycles per floating-point)
    static constexpr int GRID_WIDTH = (int)(WORLD_SIZE / CELL_SIZE) + 1; // safety buffer to prevent off-by-one memory crashes (segmentation faults) caused by floating point precision.
    static constexpr int GRID_HEIGHT = GRID_WIDTH;

    // TOTAL_CELLS to a power of 2 (262,144) to unlock bitwise hashing
    static constexpr uint32_t TOTAL_CELLS = 262144; // [(WORLD_SIZE / CELL_SIZE) + 1] = 501, where [501 * 501 = 251,101], so the next power of 2 after 251,101 is [2^18 = 262144].
    static constexpr uint32_t HASH_MASK = TOTAL_CELLS - 1; // 0x3FFFF

    // static constexpr int TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT;

    // --- GRID DATA STRUCTURES ---
    std::span<uint32_t> particleCellIndices;
    std::span<uint32_t> cellStartOffset;
    std::span<uint64_t> cellOccupancyMask; // The 32KB L1-Cache Cull Mask

    // Persistent Working Buffers for the Counting Sort
    std::span<uint32_t> cellCounts; // histogram to count particles per cell
    std::span<uint32_t> currentInsertPos; // We create a working copy of the offsets to increment as we place particles into the temp buffers

    // --- TEMPORARY SOA BUFFERS (For the Counting Sort) ---
    std::span<float> temp_pX, temp_pY, temp_pZ;
    std::span<float> temp_vX, temp_vY, temp_vZ;
    std::span<uint32_t> temp_cellIndices;

    ParticlePhysicsSOA(size_t count) {
        size_t paddedCount = (count + 7uz) & ~7uz; // C++23: 'uz' is the size_t literal, Pad for AVX2, ensures array sizes are multiples of 8.

        // Allocate one massive, contiguous block of memory for all threads
        // threadLocalCounts.resize(g_JobSystem.maxQueues * TOTAL_CELLS, 0);

        // 1. Allocate from the Arena
        uint32_t totalThreadLocalSize = g_JobSystem.maxQueues * TOTAL_CELLS;
        threadLocalCounts = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 64>(totalThreadLocalSize), totalThreadLocalSize);
        
        cellCounts = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 64>(TOTAL_CELLS), TOTAL_CELLS);
        currentInsertPos = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 64>(TOTAL_CELLS), TOTAL_CELLS);

        
        // --- // ALLOCATE DIRECTLY FROM THE ARENA IN O(1) TIME! --- AVX2 32-byte alignment
        pX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        pY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        pZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);

        vX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        vY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        vZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        
        // pX.resize(paddedCount, 0.0f);
        // pY.resize(paddedCount, 0.0f);
        // pZ.resize(paddedCount, 0.0f);
        
        // vX.resize(paddedCount, 1.5f); // Give them some initial speed
        // vY.resize(paddedCount, 0.5f);
        // vZ.resize(paddedCount, -1.0f);

        // Initialize Grid Arrays
        // particleCellIndices.resize(paddedCount, 0);
        // cellStartOffset.resize(TOTAL_CELLS, 0);
        // cellOccupancyMask.resize(TOTAL_CELLS / 64, 0); // 4,096 elements

        particleCellIndices = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 32>(paddedCount), paddedCount);
        
        // Setup Grid
        cellStartOffset = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 64>(TOTAL_CELLS), TOTAL_CELLS);
        cellOccupancyMask = std::span<uint64_t>(t_PhysicsTransientArena.Allocate<uint64_t, 64>(TOTAL_CELLS / 64), TOTAL_CELLS / 64);

        // --- ALLOCATE TEMPORARY BUFFERS FROM THE ARENA ---
        temp_pX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        temp_pY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        temp_pZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);

        temp_vX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        temp_vY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);
        temp_vZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 32>(paddedCount), paddedCount);

        temp_cellIndices = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 32>(paddedCount), paddedCount);
        temp_destIndices = std::span<uint32_t>(t_PhysicsTransientArena.Allocate<uint32_t, 32>(paddedCount), paddedCount);

        // Pre-allocate the working buffers ONCE
        // cellCounts.resize(TOTAL_CELLS, 0);
        // currentInsertPos.resize(TOTAL_CELLS, 0);

        // temp_destIndices.resize(paddedCount, 0);

        // Initialize Temp Arrays (Must be padded for AVX2!)
        // temp_cellIndices.resize(paddedCount, 0);
        // temp_pX.resize(paddedCount, 0.0f);
        // temp_pY.resize(paddedCount, 0.0f);
        // temp_pZ.resize(paddedCount, 0.0f);
        // temp_vX.resize(paddedCount, 0.0f);
        // temp_vY.resize(paddedCount, 0.0f);
        // temp_vZ.resize(paddedCount, 0.0f);

        // ==========================================
        // ZERO-INITIALIZE ARENA MEMORY
        // ==========================================
        // Unlike std::vector, raw Arena memory contains random garbage data from the OS.
        // We MUST explicitly initialize it to prevent NaN explosions in the physics math.

        // Zero-Initialize the new memory
        std::fill(threadLocalCounts.begin(), threadLocalCounts.end(), 0u);
        std::fill(cellCounts.begin(), cellCounts.end(), 0u);
        std::fill(currentInsertPos.begin(), currentInsertPos.end(), 0u);

        // --- 1. Positions (Spawn at origin) ---
        std::fill(pX.begin(), pX.end(), 0.0f);
        std::fill(pY.begin(), pY.end(), 0.0f);
        std::fill(pZ.begin(), pZ.end(), 0.0f);

        // --- 2. Velocities (Initial orbital momentum) ---
        std::fill(vX.begin(), vX.end(), 1.5f);
        std::fill(vY.begin(), vY.end(), 0.5f);
        std::fill(vZ.begin(), vZ.end(), -1.0f);

        // --- 3. Spatial Grid Core ---
        std::fill(particleCellIndices.begin(), particleCellIndices.end(), 0u);
        std::fill(cellStartOffset.begin(), cellStartOffset.end(), 0u);
        
        // Note: cellOccupancyMask is a 64-bit integer mask, so we fill with 0ULL
        std::fill(cellOccupancyMask.begin(), cellOccupancyMask.end(), 0ULL);

        // --- 4. Temporary Sorting Buffers (Zeroed out) ---
        std::fill(temp_pX.begin(), temp_pX.end(), 0.0f);
        std::fill(temp_pY.begin(), temp_pY.end(), 0.0f);
        std::fill(temp_pZ.begin(), temp_pZ.end(), 0.0f);
        
        std::fill(temp_vX.begin(), temp_vX.end(), 0.0f);
        std::fill(temp_vY.begin(), temp_vY.end(), 0.0f);
        std::fill(temp_vZ.begin(), temp_vZ.end(), 0.0f);

        std::fill(temp_cellIndices.begin(), temp_cellIndices.end(), 0u);
        std::fill(temp_destIndices.begin(), temp_destIndices.end(), 0u);
    }

    // --- Reusable Spawner (Thread-Safe & Lock-Free, Trigonometry) ---
    void spawnParticles(size_t startIdx, size_t endIdx, float speed) {

        uint32_t spawnCount = endIdx - startIdx;
        if (spawnCount <= 0) return;

        // 1. Thread Chunking (Offloads the spawn from the UI Thread to the Workers)
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t CHUNK_SIZE = std::max(1024u, static_cast<uint32_t>(spawnCount / (threadCount * 4)));
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7; // Pad for AVX2

        // Dispatch the workload to keep the ImGui slider buttery smooth
        g_JobSystem.DispatchAndWait(spawnCount, CHUNK_SIZE, [&](uint32_t localStart, uint32_t localEnd) {

            float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
            float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());

            float* ENGINE_RESTRICT r_vX = std::assume_aligned<32>(vX.data());
            float* ENGINE_RESTRICT r_vY = std::assume_aligned<32>(vY.data());
            float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());

            // Fast, stateless hash function for lock-free parallel noise
            // Evaluates in ~3 clock cycles per particle in the ALU with zero memory fetches and is much better than C-langauge rand().
            auto fastHash = [](uint32_t index) -> float {
                uint32_t state = index * 747796405u + 2891336453u;
                state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                state = (state >> 22u) ^ state;

                // return (float)state / (float)UINT32_MAX; // Range: [0.0, 1.0]

                // OPTIMIZATION: Multiply by the inverse (1.0 / 4294967295.0)
                // Replaces a 15-cycle division with a 4-cycle multiplication!
                return static_cast<float>(state) * 2.3283064365386963e-10f;
            };

            // OPTIMIZATION: The Rotation Matrix
            float deltaTheta = 0.001f;
            float cos_d = std::cos(deltaTheta);
            float sin_d = std::sin(deltaTheta);

            // Calculate the exact starting angle for this specific thread's chunk ONLY
            float startAngle = (float)(startIdx + localStart) * deltaTheta;
            float current_c = std::cos(startAngle);
            float current_s = std::sin(startAngle);

            // OPTIMIZATION: Traded ~250 clock cycles of trigonometry and division for ~10 clock cycles of pure multiplication.
            for(size_t i = localStart; i < localEnd; ++i) {
                // The true array index in the massive memory pool
                uint32_t globalIdx = startIdx + i;

                // Map the [0.0, 1.0] hash range to your [0.0, 600.0] range
                float randomOffset = fastHash(globalIdx) * 600.0f;
                float r = 200.0f + randomOffset;

                // 1. Apply the current rotation directly
                r_pX[globalIdx] = current_c * r;
                r_pY[globalIdx] = current_s * r;
                r_pZ[globalIdx] = 0.0f; 

                // Tangential velocity is just the perpendicular vector (-y, x)
                r_vX[globalIdx] = -current_s * speed; 
                r_vY[globalIdx] =  current_c * speed;
                r_vZ[globalIdx] = 0.0f;

                // 2. Rotate the direction vector for the NEXT particle
                // (This replaces ~200 cycles of std::cos/sin with just 6 cycles of ALU math)
                float next_c = (current_c * cos_d) - (current_s * sin_d);
                float next_s = (current_s * cos_d) + (current_c * sin_d);
                
                current_c = next_c;
                current_s = next_s;

                // 3. Prevent Floating Point Drift
                // Over thousands of matrix multiplications, float precision degrades.
                // We re-normalize the vector back to a length of 1.0 every 256 particles.
                // Bitwise AND (& 255) is an infinitely fast modulo check.
                if ((i & 255) == 0) {
                    // Fast hardware approximation of 1.0 / sqrt(x)
                    float magSq = current_c * current_c + current_s * current_s;
                    float invMag = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(magSq)));
                    
                    current_c *= invMag;
                    current_s *= invMag;
                }
            }
        });
    }

    FORCE_INLINE void transitionDimensions(int activeCount, bool to2D) {
        // We only need basic threading here, it happens exactly once per toggle
        g_JobSystem.DispatchAndWait(activeCount, 2048, [&](uint32_t start, uint32_t end) {

            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());
            float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());

            // OPTIMIZATION: Branch is hoisted OUTSIDE the hot loop!
            if (to2D) {
                // Fast path: Squash everything flat and kill vertical momentum instantly
                for (uint32_t i = start; i < end; ++i) {
                    // Squash everything flat and kill vertical momentum instantly
                    r_pZ[i] = 0.0f;
                    r_vZ[i] = 0.0f;
                }
            } else {

                // OPTIMIZATION: Only declare the hash function if we actually need it
                // Fast, stateless hash function for lock-free parallel noise
                // This generates a pseudo-random float between 0.0 and 1.0 based purely on an index.
                auto fastHash = [](uint32_t index) -> float {
                    uint32_t state = index * 747796405u + 2891336453u;
                    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                    state = (state >> 22u) ^ state;

                    // OPTIMIZATION: Multiply by inverse (1.0 / 4294967295.0)
                    return static_cast<float>(state) * 2.3283064365386963e-10f;
                };

                // Explode the particles into a 3D disc/cloud safely in parallel
                for (uint32_t i = start; i < end; ++i) {
                    float rand1 = fastHash(i);
                    float rand2 = fastHash(i + activeCount); // Offset the seed for a different result

                    // Map the 0.0 to 1.0 range back to your desired physical bounds
                    r_pZ[i] = (rand1 * 100.0f) - 50.0f;  // -50 to 50 
                    r_vZ[i] = (rand2 * 4.0f) - 2.0f;     // -2 to 2
                }
            }
        });
    }

    // --- Dedicated Physics Benchmark (Zero OS Overhead) ---
    FORCE_INLINE void integrate_benchmark(float deltaTime, int activeCount, float gravityVal, int64_t repeats) {

        [[assume(activeCount > 0)]]; // C++23: Hint to optimizer that arrays aren't empty
        
        // SIMD requires multiples of 8
        int paddedActiveCount = (activeCount + 7) & ~7; 

        // 1. Calculate a fair chunk size for the threads (Max payload, like the SOA test)
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t CHUNK_SIZE = std::max(1u, (uint32_t)(paddedActiveCount / threadCount));
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7; // Ensure AVX2 8-float alignment

        // 2. Dispatch the Job System EXACTLY ONCE
        g_JobSystem.DispatchAndWait(paddedActiveCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Ensure our start and end align to AVX boundaries
            uint32_t alignedStart = start & ~7;
            uint32_t alignedEnd = (end + 7) & ~7;

            // CRITICAL OPTIMIZATION: Extract raw aligned pointers.
            // Guarantees to the compiler that these arrays DO NOT OVERLAP and are strictly 32-byte aligned.
            float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
            float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());
            float* ENGINE_RESTRICT r_vX = std::assume_aligned<32>(vX.data());
            float* ENGINE_RESTRICT r_vY = std::assume_aligned<32>(vY.data());
            float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());

            // Pre-load constants into AVX registers ONCE per thread
            __m256 dt = _mm256_set1_ps(deltaTime);
            __m256 gravityStrength = _mm256_set1_ps(gravityVal); 
            __m256 epsilon = _mm256_set1_ps(0.001f);       
            __m256 damping = _mm256_set1_ps(1.0f);   
            
            __m256 three_halves = _mm256_set1_ps(1.5f);
            __m256 half = _mm256_set1_ps(0.5f);

            // 3. The 20,000 repeats happen INSIDE the thread (Zero OS interruptions!)
            for (int64_t r = 0LL; r < repeats; ++r) {
                
                for (int i = alignedStart; i < alignedEnd; i += 8) {
                    // 1. ALIGNED LOAD (_mm256_load_ps): Positions and Velocities
                    // The silicon no longer has to check for cache line straddling. 
                    // It blindly pulls exactly half of a 64-byte L1 cache line directly into the YMM register.
                    // __m256 px = _mm256_load_ps(&pX[i]);
                    // __m256 py = _mm256_load_ps(&pY[i]);
                    // __m256 pz = _mm256_load_ps(&pZ[i]);

                    // __m256 vx = _mm256_load_ps(&vX[i]);
                    // __m256 vy = _mm256_load_ps(&vY[i]);
                    // __m256 vz = _mm256_load_ps(&vZ[i]);

                    // Loading from the perfectly aliased, restricted pointers
                    __m256 px = _mm256_load_ps(r_pX + i);
                    __m256 py = _mm256_load_ps(r_pY + i);
                    __m256 pz = _mm256_load_ps(r_pZ + i);

                    __m256 vx = _mm256_load_ps(r_vX + i);
                    __m256 vy = _mm256_load_ps(r_vY + i);
                    __m256 vz = _mm256_load_ps(r_vZ + i);

                    // 2. MATH: Distance to Center (0,0,0)
                    __m256 distSq = _mm256_mul_ps(px, px);
                    distSq = _mm256_fmadd_ps(py, py, distSq);
                    distSq = _mm256_fmadd_ps(pz, pz, distSq);

                    // Fast Hardware Approximation (Bypasses Newton-Raphson)
                    // 1. Get the hardware's 12-bit guess
                     __m256 distSq_eps = _mm256_add_ps(distSq, epsilon);
                    __m256 rsqrt_approx = _mm256_rsqrt_ps(distSq_eps);

                    // 2. Newton-Raphson Refinement (Restores 23-bit precision)
                    // Formula: y = y * (1.5 - 0.5 * x * y * y)
                    __m256 half_x = _mm256_mul_ps(distSq_eps, half);
                    __m256 y_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);

                    // 3. _mm256_fnmadd_ps perfectly calculates: -(half_x * y_sq) + 1.5
                    __m256 term = _mm256_fnmadd_ps(half_x, y_sq, three_halves);
                    __m256 invDist = _mm256_mul_ps(rsqrt_approx, term);

                    // 4. GRAVITY CALCULATION
                    __m256 pull = _mm256_mul_ps(invDist, gravityStrength);
                    
                    vx = _mm256_fnmadd_ps(px, pull, vx);
                    vy = _mm256_fnmadd_ps(py, pull, vy);
                    vz = _mm256_fnmadd_ps(pz, pull, vz);

                    // 4. APPLY DAMPING (Friction)
                    vx = _mm256_mul_ps(vx, damping);
                    vy = _mm256_mul_ps(vy, damping);
                    vz = _mm256_mul_ps(vz, damping);

                    // 5. UPDATE POSITION: p = p + v * dt
                    px = _mm256_fmadd_ps(vx, dt, px);
                    py = _mm256_fmadd_ps(vy, dt, py);
                    pz = _mm256_fmadd_ps(vz, dt, pz);

                    // 6. ALIGNED STORE (_mm256_store_ps): Save results
                    // Unaligned stores sometimes force the CPU to do a "Read-Modify-Write" 
                    // if the data crosses a cache line. Aligned stores bypass this entirely,
                    // writing cleanly to the write-combine buffer/L1 cache.
                    // _mm256_store_ps(&pX[i], px);
                    // _mm256_store_ps(&pY[i], py);
                    // _mm256_store_ps(&pZ[i], pz);

                    // _mm256_store_ps(&vX[i], vx);
                    // _mm256_store_ps(&vY[i], vy);
                    // _mm256_store_ps(&vZ[i], vz);

                    // Writing back to restricted pointers
                    _mm256_store_ps(r_pX + i, px);
                    _mm256_store_ps(r_pY + i, py);
                    _mm256_store_ps(r_pZ + i, pz);

                    _mm256_store_ps(r_vX + i, vx);
                    _mm256_store_ps(r_vY + i, vy);
                    _mm256_store_ps(r_vZ + i, vz);
                }
            }
        });
    }

    // --- BUILD SPATIAL GRID (O(N) Counting Sort) RUNS EVERY SINGLE FRAME ---
    FORCE_INLINE void buildSpatialGridParallel(int activeCount) {
        [[assume(activeCount >= 0)]];

        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);

        // Cache to local consts. The compiler locks these into registers!
        const bool localIs2D = g_EngineSettings.is2DMode;
        const bool localIsLegacy = g_EngineSettings.isLegacyCPU;

        // -- SCALAR PREFIX SUM (SLOWER THAN PARALLEL) --
        // Granular chunks! (Targeting ~1024 to 4096 particles per chunk) This ensures idle threads always have tiny, bite-sized tasks to steal.
        // const uint32_t targetChunksPerThread = 8;
        // uint32_t CHUNK_SIZE = std::max(1u, (uint32_t)(activeCount / (threadCount * targetChunksPerThread)));
        // CHUNK_SIZE = std::clamp(CHUNK_SIZE, 1024u, 4096u);

        // =========================================================
        // PHASE 1: PARALLEL HASH & HISTOGRAM (Lock-Free, Dual-Path)
        // =========================================================

        // -- PARALLEL PREFIX SUM (EXTREMELY FAST) --
        // [u]: compiler treats it as an unsigned integer to prevent any signed bitwise overflow.
        uint32_t CHUNK_SIZE = std::max(2048u, (uint32_t)(activeCount / (threadCount * 8)));
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7;  // Pad for AVX2

        // --- AVX2 CODE TAKES ROUGHLY 15-20 CYCLES TO PROCESS 8 PARTICLES ---
        g_JobSystem.DispatchAndWait(activeCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Grab the ID of the thread actually executing this chunk (even if stolen)
            // Safe to use Worker ID here because Histograms are communicative. 
            // It doesn't matter who counts the particle, just that the total is correct.
            uint32_t workerId = tl_workerIndex; 

            // Calculate AVX boundary so we don't read past the array
            uint32_t alignedEnd = start + ((end - start) & ~7u);

            float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
            float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());
            uint32_t* ENGINE_RESTRICT r_particleCellIndices = std::assume_aligned<32>(particleCellIndices.data());

            // Pre-load constants for the AVX2 loop
            __m256 vWorldOffset = _mm256_set1_ps(WORLD_SIZE * 0.5f);
            __m256 vInvCellSize = _mm256_set1_ps(INV_CELL_SIZE);
            __m256i vHashMask   = _mm256_set1_epi32(HASH_MASK);
            __m256i vZero       = _mm256_setzero_si256();

            // The maximum boundary for our 1024-size LUT
            __m256i vMaxGrid = _mm256_set1_epi32(1023);

            // SOFTWARE WRITE-COMBINING BUFFER: 256 integers = 1 Kilobyte. 1KB sits permanently inside the L1 CPU Cache.
            alignas(64) uint32_t hashBuffer[256]; 
            uint32_t bufferIdx = 0;

            // Temporary array to extract hashes back to scalar for the histogram update
            // alignas(32) uint32_t extractedHashes[8];
            
            // ----------------------------------------------------
            // 1. THE AVX2 FAST PATH (8 Particles at a time)
            // ----------------------------------------------------
            for (uint32_t i = start; i < alignedEnd; i += 8) {
                // Load Positions
                // __m256 px = _mm256_load_ps(&pX[i]);
                // __m256 py = _mm256_load_ps(&pY[i]);
                __m256 px = _mm256_load_ps(r_pX + i);
                __m256 py = _mm256_load_ps(r_pY + i);

                // Shift Coordinates to positive space
                px = _mm256_add_ps(px, vWorldOffset);
                py = _mm256_add_ps(py, vWorldOffset);

                // Convert Floats to Grid Integers (Hardware Truncation)
                __m256i gridX = _mm256_cvttps_epi32(_mm256_mul_ps(px, vInvCellSize));
                __m256i gridY = _mm256_cvttps_epi32(_mm256_mul_ps(py, vInvCellSize));
                __m256i gridZ = vZero;

                if (!localIs2D) {
                    // __m256 pz = _mm256_load_ps(&pZ[i]);
                    __m256 pz = _mm256_load_ps(r_pZ + i);
                    pz = _mm256_add_ps(pz, vWorldOffset);
                    gridZ = _mm256_cvttps_epi32(_mm256_mul_ps(pz, vInvCellSize));

                    // Clamp Z axis if we are in 3D mode
                    gridZ = _mm256_max_epi32(vZero, _mm256_min_epi32(gridZ, vMaxGrid));
                }

                // THE AVX2 CLAMP: Restricts integers between 0 and 1023 instantly across all 8 particles to ensure that an exploding particle will never generate an out-of-bounds integer before hashing.
                gridX = _mm256_max_epi32(vZero, _mm256_min_epi32(gridX, vMaxGrid));
                gridY = _mm256_max_epi32(vZero, _mm256_min_epi32(gridY, vMaxGrid));

                // SIMULTANEOUS HASHING (AVX2 Integer Math)
                // If Legacy is off, we still use the AVX2 math here because _pdep_u32 cannot process 8 numbers at once.
                __m256i hashes = getMortonCode_AVX2(gridX, gridY, gridZ);
                hashes = _mm256_and_si256(hashes, vHashMask);

                // Store 8 hashes back to the main array simultaneously
                // _mm256_storeu_si256((__m256i*)&particleCellIndices[i], hashes);

                // Use the restricted pointer for clean, unaliased storing
                _mm256_store_si256((__m256i*)(r_particleCellIndices + i), hashes);

                // Write to our L1 Cache buffer instead of main memory! Ensures AVX2 pipeline never stalls and reduces memory bus contention.
                _mm256_store_si256((__m256i*)&hashBuffer[bufferIdx], hashes);
                bufferIdx += 8;

                // When the L1 buffer is full, flush it to the L3 Cache histogram in one big batch
                if (bufferIdx == 256) {
                    for (uint32_t k = 0; k < 256; ++k) {
                        // Flat array math: (Worker ID * Total Cells) + Hash
                        threadLocalCounts[(workerId * TOTAL_CELLS) + hashBuffer[k]]++;
                    }
                    bufferIdx = 0; // Reset buffer
                }

            }

            // Flush any remaining hashes in the buffer before moving to the scalar remainder!
            for (uint32_t k = 0; k < bufferIdx; ++k) {
                threadLocalCounts[(workerId * TOTAL_CELLS) + hashBuffer[k]]++;
            }

            // ----------------------------------------------------
            // 2. THE SCALAR REMAINDER (LUT Fallback)
            // ----------------------------------------------------

            // Pre-load SSE constants outside the loop to keep registers hot
            __m128 vMinGrid128 = _mm_setzero_ps();
            __m128 vMaxGrid128 = _mm_set_ss(1023.0f);

            // --- C++20 TEMPLATED LAMBDA DISPATCHER ---
            auto processScalarRemainder = [&]<bool Is2D, bool IsLegacy>() {
                // Clean up the remaining 1 to 7 particles using ultra-fast L1 cache table
                for (uint32_t i = alignedEnd; i < end; ++i) {
                    // Shift coordinates from (-1000, 1000) to (0, 2000) so grid math is positive
                    float shiftedX = r_pX[i] + (WORLD_SIZE * 0.5f);
                    float shiftedY = r_pY[i] + (WORLD_SIZE * 0.5f);

                    // std::clamp: Guarantees the index never exceeds our 1024 LUT bounds to prevent segfaults (i.e., particles won't crash the engine if they wander slightly off-grid)
                    // [15-20 clock cycles] to perform a single floating-point division, [4 clock cycles] to perform a single floating-point multiplication
                    // uint32_t gridX = (uint32_t)std::clamp((int)(shiftedX * INV_CELL_SIZE), 0, 1023);
                    // uint32_t gridY = (uint32_t)std::clamp((int)(shiftedY * INV_CELL_SIZE), 0, 1023);
                    // uint32_t gridZ = (uint32_t)std::clamp((int)(shiftedZ * INV_CELL_SIZE), 0, 1023);

                    // --- BRANCHLESS SSE CLAMP ---
                    /*
                        - Executes directly on the Vector ALU ports.
                        - Takes ~3-4 clock cycles to complete, every single time.
                        - Truncates the scalar float in the SSE register directly into 32-bit integer in one step.
                    */
                    // 1. Multiply by inverse cell size directly inside the SSE register
                    __m128 vx = _mm_set_ss(shiftedX * INV_CELL_SIZE);
                    __m128 vy = _mm_set_ss(shiftedY * INV_CELL_SIZE);

                    // 2. Hardware Min/Max (0 branches, pure silicon math)
                    vx = _mm_max_ss(vMinGrid128, _mm_min_ss(vx, vMaxGrid128));
                    vy = _mm_max_ss(vMinGrid128, _mm_min_ss(vy, vMaxGrid128));

                    // 3. Hardware truncation from float to integer
                    uint32_t gridX = static_cast<uint32_t>(_mm_cvttss_si32(vx));
                    uint32_t gridY = static_cast<uint32_t>(_mm_cvttss_si32(vy));

                    // Hash to map 3D space into a fixed-size 1D array.
                    uint32_t hash;

                    // [Hot loop]: using global toggles is safe in hot loops b/c its static ([false] for 1 million cycles or [true] for 1 million cycles).
                    // Compile-time branch: Z-math is entirely deleted from the 2D binary path
                    if constexpr (Is2D) {
                        // Fast 2D Path: Morton Encoding (pass 0 for Z)
                        hash = getMortonCode<IsLegacy>(gridX, gridY, 0) & HASH_MASK;
                    } 
                    else {
                        // True 3D Path: Morton Encoding
                        float shiftedZ = r_pZ[i] + (WORLD_SIZE * 0.5f);

                        // Branchless Z-Clamp
                        __m128 vz = _mm_set_ss(shiftedZ * INV_CELL_SIZE);
                        vz = _mm_max_ss(vMinGrid128, _mm_min_ss(vz, vMaxGrid128));
                        uint32_t gridZ = static_cast<uint32_t>(_mm_cvttss_si32(vz));

                        /* [ALU Bitwise Operator (&)]
                            - The modulo (%) operator compiles to a slow integer division instruction (idiv on x86) that costs 15-20 clock cycles.
                            - hash = getMortonCode(gridX, gridY, gridZ) % TOTAL_CELLS;
                            - It cannot be easily pipelined. 
                            - Doing this 9 times (2D) or 27 times (3D) per particle, for 100,000 particles, means you are burning over 40 million clock cycles per frame.

                            - The bitwise AND (&) operator executes in 1 clock cycle (15-20x faster).
                            - By forcing TOTAL_CELLS to be a strict power of 2 (e.g., 264,144 instead of 250,001) this allow us to replace modulo (%) with the AND (&) bitwise operator to boost performance.
                            - This adjustment alone to use (&) can improve frame rates by 5-7fps compared to modulo (%).
                            - The bitwise XOR (^) prevents diagonal symmetry bugs
                        */
                        hash = getMortonCode<IsLegacy>(gridX, gridY, gridZ) & HASH_MASK;
                    }

                    // Save the hash so we don't calculate it twice, and increment the histogram
                    r_particleCellIndices[i] = hash;

                    // Flat Array Math: (Worker ID * Total Cells) + Hash (i.e., flat mapping)
                    threadLocalCounts[(workerId * TOTAL_CELLS) + hash]++; // Thread-local write! Zero false sharing, zero atomics.
                }
            };

            // --- DISPATCH THE KERNEL ---
            // Evaluates the runtime global booleans exactly once, jumping to the pure assembly path.
            if (localIs2D) {
                if (localIsLegacy) processScalarRemainder.template operator()<true, true>();
                else               processScalarRemainder.template operator()<true, false>();
            } else {
                if (localIsLegacy) processScalarRemainder.template operator()<false, true>();
                else               processScalarRemainder.template operator()<false, false>();
            }
        });

        // ==========================================
        // PHASE 2: PARALLEL GLOBAL PREFIX SUM (SCAN)
        // ==========================================

        // We slice the 262,144 cells into 64 chunks of 4096 cells each.
        constexpr uint32_t PREFIX_CHUNK_SIZE = 4096;
        constexpr uint32_t NUM_CHUNKS = TOTAL_CELLS / PREFIX_CHUNK_SIZE; 
        
        // Stack-allocated array to hold the total particle count of each chunk
        uint32_t chunkTotals[NUM_CHUNKS] = {0};

        /* [Job System]
            1. Calculates the sums in parallel.
            2. Perform a lightning fast 64 iteration stitch on the main thread.
            3. Fan back out.
        */

        // --- STEP 2A: Local Chunk Reduction (Parallel, Extremely Fast) ---
        g_JobSystem.DispatchAndWait(TOTAL_CELLS, PREFIX_CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
            uint32_t chunkIdx = start / PREFIX_CHUNK_SIZE;
            uint32_t localRunningTotal = 0;

            // Extract the thread local counts!
            uint32_t* ENGINE_RESTRICT r_threadLocalCounts = std::assume_aligned<64>(threadLocalCounts.data());
            uint32_t* ENGINE_RESTRICT r_cellStartOffset = std::assume_aligned<64>(cellStartOffset.data());
            uint64_t* ENGINE_RESTRICT r_cellOccupancyMask = std::assume_aligned<64>(cellOccupancyMask.data());

            for (uint32_t cell = start; cell < end; ++cell) {
                // Save the start of this cell so solveCollisions() can find it!
                // cellStartOffset[cell] = localRunningTotal; // Store the local offset temporarily
                r_cellStartOffset[cell] = localRunningTotal;
                uint32_t cellCount = 0;

                // Accumulate all threads' contributions for this specific cell
                for (uint32_t t = 0; t < threadCount; ++t) {
                    uint32_t flatIndex = (t * TOTAL_CELLS) + cell;

                    // 1. Read the data into the ALU
                    cellCount += r_threadLocalCounts[flatIndex];

                    // 2. ZERO-ON-READ: Zero the memory while it is still hot in the L1 Cache! This instantly prepares the buffer for the NEXT frame for free.
                    // Requires no Read-For-Ownership (RFO) penalty from the main RAM.
                    r_threadLocalCounts[flatIndex] = 0;
                }
                localRunningTotal += cellCount;

                // ==========================================
                // POPULATE THE L1 CULLING MASK
                // ==========================================
                // Shift right by 6 (>> 6) is identical to dividing by 64.
                // Bitwise AND 63 (& 63) is identical to modulo 64.
                if (cellCount > 0) {
                    // Set the bit to 1 (Occupied)
                    r_cellOccupancyMask[cell >> 6] |= (1ULL << (cell & 63));
                } else {
                    // Set the bit to 0 (Empty) to clean up from the previous frame
                    r_cellOccupancyMask[cell >> 6] &= ~(1ULL << (cell & 63));
                }
            }
            // Save the total sum of this chunk for the Main Thread to read
            chunkTotals[chunkIdx] = localRunningTotal;
        });

        // --- STEP 2B: Global Scan (Main Thread, Extremely Fast) ---
        // Instead of 2,000,000 iterations, the Main Thread now only does exactly 64 iterations!
        // Evaluates in less than 1 microsecond.
        uint32_t globalOffsets[NUM_CHUNKS] = {0};
        uint32_t globalTotal = 0;
        
        for (uint32_t i = 0; i < NUM_CHUNKS; ++i) {
            globalOffsets[i] = globalTotal;
            globalTotal += chunkTotals[i];
        }

        // --- STEP 2C: Apply Global Offsets & Setup Scatter Buffer (Parallel, Extremely Fast) ---
        // By parallelizing the buffer setup, we completely eliminate the need for 
        // the slow std::memcpy on the main thread!
        g_JobSystem.DispatchAndWait(TOTAL_CELLS, PREFIX_CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
            uint32_t chunkIdx = start / PREFIX_CHUNK_SIZE;
            uint32_t globalBase = globalOffsets[chunkIdx];

            uint32_t* ENGINE_RESTRICT r_cellStartOffset = std::assume_aligned<64>(cellStartOffset.data());
            uint32_t* ENGINE_RESTRICT r_currentInsertPos = std::assume_aligned<64>(currentInsertPos.data());

            for (uint32_t cell = start; cell < end; ++cell) {
                // Add the chunk's global base to the local offsets calculated in Step 2A
                // cellStartOffset[cell] += globalBase;
                
                // Simultaneously initialize the working copy for Phase 3A (Atomic Scatter)
                // currentInsertPos[cell] = cellStartOffset[cell]; 

                r_cellStartOffset[cell] += globalBase;
                r_currentInsertPos[cell] = r_cellStartOffset[cell];
            }
        });

        // ==========================================
        // PHASE 2: SCALAR PREFIX SUM (SLOWER THAN PARALLEL)
        // ==========================================
        // uint32_t currentGlobalOffset = 0;

        // for (uint32_t cell = 0; cell < TOTAL_CELLS; ++cell) {
        //     // Save the global start of this cell so solveCollisions() can find it!
        //     cellStartOffset[cell] = currentGlobalOffset;

        //     for (uint32_t t = 0; t < threadCount; ++t) {
        //         currentGlobalOffset += threadLocalCounts[t][cell];
        //     }
        // }

        // // Quickly copy offsets for Phase 3A
        // std::memcpy copies megabytes of data and hogs the main memory bus.
        // std::memcpy(currentInsertPos.data(), cellStartOffset.data(), TOTAL_CELLS * sizeof(uint32_t));

        // ==========================================
        // PHASE 3A: ATOMIC SCATTER DESTINATIONS
        // ==========================================
        g_JobSystem.DispatchAndWait(activeCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            uint32_t* ENGINE_RESTRICT r_particleCellIndices = std::assume_aligned<32>(particleCellIndices.data());
            uint32_t* ENGINE_RESTRICT r_temp_destIndices = std::assume_aligned<32>(temp_destIndices.data());
            uint32_t* ENGINE_RESTRICT r_temp_cellIndices = std::assume_aligned<32>(temp_cellIndices.data());
            uint32_t* ENGINE_RESTRICT r_currentInsertPos = std::assume_aligned<64>(currentInsertPos.data());

            for (uint32_t i = start; i < end; ++i) {
                // uint32_t hash = particleCellIndices[i];
                uint32_t hash = r_particleCellIndices[i];

                // std::atomic_ref guarantees safety regardless of which worker steals the chunk.
                // C++20: Safely increment the global insert position regardless of which thread stole this job!
                std::atomic_ref<uint32_t> safeOffset(r_currentInsertPos[hash]);

                // Store the exact destination index for this particle.
                // temp_destIndices[i] = safeOffset.fetch_add(1, std::memory_order_relaxed);

                // Keep the hash synced in the new sorted order
                // We can safely move the hash now too
                // temp_cellIndices[temp_destIndices[i]] = hash;

                r_temp_destIndices[i] = safeOffset.fetch_add(1, std::memory_order_relaxed);
                r_temp_cellIndices[r_temp_destIndices[i]] = hash;
            }
            
        });

        // ==========================================
        // PHASE 3B: MULTI-PASS MEMORY STREAMING (Cache Friendly)
        // ==========================================
        // It looks like more code and more loops, but to the silicon, it is significantly faster b/c it prevents the CPU execution units from idling and waiting for RAM.
        // Now we move the data one component at a time. 
        // The hardware prefetcher only has to track: pX (read), destIndices (read), temp_pX (write).
        g_JobSystem.DispatchAndWait(activeCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // By proving these pointers don't overlap, the compiler unrolls these loops flawlessly
            float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
            float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());
            
            float* ENGINE_RESTRICT r_temp_pX = std::assume_aligned<32>(temp_pX.data());
            float* ENGINE_RESTRICT r_temp_pY = std::assume_aligned<32>(temp_pY.data());
            float* ENGINE_RESTRICT r_temp_pZ = std::assume_aligned<32>(temp_pZ.data());

            float* ENGINE_RESTRICT r_vX = std::assume_aligned<32>(vX.data());
            float* ENGINE_RESTRICT r_vY = std::assume_aligned<32>(vY.data());
            float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());

            float* ENGINE_RESTRICT r_temp_vX = std::assume_aligned<32>(temp_vX.data());
            float* ENGINE_RESTRICT r_temp_vY = std::assume_aligned<32>(temp_vY.data());
            float* ENGINE_RESTRICT r_temp_vZ = std::assume_aligned<32>(temp_vZ.data());

            uint32_t* ENGINE_RESTRICT r_temp_destIndices = std::assume_aligned<32>(temp_destIndices.data());

            // By separating these loops we stop thrashing the cache.
            // Streams through pX linearly, looks up the destination, and writes. Then repeats this process for pY, pZ, etc..
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_pX[dest] = pX[i]; // Move the data into the temporary SoA arrays
                r_temp_pX[r_temp_destIndices[i]] = r_pX[i];
            }
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_pY[dest] = pY[i];
                r_temp_pY[r_temp_destIndices[i]] = r_pY[i];
            }
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_pZ[dest] = pZ[i];
                r_temp_pZ[r_temp_destIndices[i]] = r_pZ[i];
            }
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_vX[dest] = vX[i];
                r_temp_vX[r_temp_destIndices[i]] = r_vX[i];
            }
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_vY[dest] = vY[i];
                r_temp_vY[r_temp_destIndices[i]] = r_vY[i];
            }
            for (uint32_t i = start; i < end; ++i) {
                // uint32_t dest = temp_destIndices[i];
                // temp_vZ[dest] = vZ[i];
                r_temp_vZ[r_temp_destIndices[i]] = r_vZ[i];
            }
        });

        // ==========================================
        // PHASE 4: POINTER SWAP
        // ==========================================

        // std::vector::swap does NOT copy data. It merely swaps the internal memory pointers.
        // This is a wildly fast O(1) operation.
        // pX.swap(temp_pX);
        // pY.swap(temp_pY);
        // pZ.swap(temp_pZ);
        // vX.swap(temp_vX);
        // vY.swap(temp_vY);
        // vZ.swap(temp_vZ);
        // particleCellIndices.swap(temp_cellIndices);

        // std::swap flips the memory addresses the spans are looking at in O(1) time.
        std::swap(pX, temp_pX);
        std::swap(pY, temp_pY);
        std::swap(pZ, temp_pZ);
        std::swap(vX, temp_vX);
        std::swap(vY, temp_vY);
        std::swap(vZ, temp_vZ);
        std::swap(particleCellIndices, temp_cellIndices);
    }

    // --- AVX2 SPATIAL HASH COLLISION SOLVER ---
    FORCE_INLINE void solveCollisions(int activeCount) {
        
        // 1. Thread Chunking
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        const uint32_t targetChunksPerThread = 16;
        uint32_t CHUNK_SIZE = std::max(1u, (uint32_t)(activeCount / (threadCount * targetChunksPerThread)));

        // This 512 guarantees the AVX2 payload mathematically drowns out the coroutine indirect branch overhead.
        CHUNK_SIZE = std::clamp(CHUNK_SIZE, 512u, 1024u); // Keep chunks small for L1 Cache

        // Ensure AVX2 alignment
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7;

        // Lock globals into local CPU registers
        const bool localIs2D = g_EngineSettings.is2DMode;
        const bool localIsLegacy = g_EngineSettings.isLegacyCPU;

        // ======================================
        // TEMPLATED KERNEL DISPATCHER
        // ======================================
        /*
            - collisionKernel evaluates g_EngineSettings.is2DMode exactly once per thread chunk.
            - After that it locks this thread onto a perfectly unrolled, branchless track of pure AVX2 execution for the next 512 to 1024 particles.
            - Improves performance.
            - Since congnitive load is shifted over to the compiler, the CPU can focus on the raw, uninterrupted stream of math.
        */

        g_JobSystem.DispatchAndWait(activeCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {
            // --- C++20 TEMPLATED KERNEL DISPATCHER ---
            // By templating this entire block, the compiler generates 4 distinct, highly-optimized branchless versions of your physics engine!
            auto collisionKernel = [&]<bool Is2D, bool IsLegacy>() {

                // Extract aligned, restricted pointers
                float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
                float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
                float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());

                float* ENGINE_RESTRICT r_vX = std::assume_aligned<32>(vX.data());
                float* ENGINE_RESTRICT r_vY = std::assume_aligned<32>(vY.data());
                float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());
                
                uint32_t* ENGINE_RESTRICT r_particleCellIndices = std::assume_aligned<32>(particleCellIndices.data());
                uint32_t* ENGINE_RESTRICT r_cellStartOffset = std::assume_aligned<64>(cellStartOffset.data());
                uint64_t* ENGINE_RESTRICT r_cellOccupancyMask = std::assume_aligned<64>(cellOccupancyMask.data());
            
                // --- COLLISION CONSTANTS ---
                const float RADIUS = CELL_SIZE; // Distance at which they collide
                const float RADIUS_SQ = RADIUS * RADIUS;
                const float REPULSION_STRENGTH = 5.0f;

                __m256 vRadiusSq = _mm256_set1_ps(RADIUS_SQ);
                __m256 vRepulsion = _mm256_set1_ps(REPULSION_STRENGTH);
                __m256 vEpsilon = _mm256_set1_ps(0.0001f); // Prevents divide by zero and self-collision

                __m256 half = _mm256_set1_ps(0.5f);
                __m256 three_halves = _mm256_set1_ps(1.5f);

                // Temporary arrays to extract AVX2 registers back to scalar
                // float tempX[8], tempY[8], tempZ[8];

                // Counts the exact number of neighbor checks a worker performs.
                uint64_t localThreadOps = 0;

                // =========================================================
                // L1 NEIGHBOR CACHE (Eliminates redundant broad-phase lookups) 
                // =========================================================
                uint32_t lastHash = 0xFFFFFFFF; // Set to invalid hash to force initial load
                
                struct NeighborRange {
                    uint32_t startIdx;
                    uint32_t endIdx;
                };
                // 27 max neighbors in 3D. Lives permanently in hot L1 cache!
                NeighborRange neighborCache[27]; 
                int validNeighbors = 0;

                for (uint32_t i = start; i < end; ++i) {
                    // Keep telemetry perfectly accurate!
                    // localThreadOps += 45; // Base Newton-Raphson + Integration cost
                    localThreadOps += Is2D ? 35 : 45;

                    // Broadcast Particle 'i' to all 8 slots in the SIMD register
                    __m256 p_i_x = _mm256_set1_ps(r_pX[i]);
                    __m256 p_i_y = _mm256_set1_ps(r_pY[i]);
                    __m256 p_i_z = _mm256_set1_ps(r_pZ[i]);

                    // Accumulators: We will accumulate forces in these registers
                    __m256 accX = _mm256_setzero_ps();
                    __m256 accY = _mm256_setzero_ps();
                    __m256 accZ = _mm256_setzero_ps();

                    float scalarForceX = 0.0f, scalarForceY = 0.0f, scalarForceZ = 0.0f;

                    // Check if we moved to a new spatial cell
                    uint32_t myHash = r_particleCellIndices[i];

                    // Calculates the neighbor offsets once and stores them in ultra-fast L1 cache, and resuses that list for every subsequent particle 
                    // that shares that cell which will reduce the bounds checking, integer math, and unpredictable branching by 90% in this hot loop.
                    // Dramatically increases performance when increasing the number of particles on screen.  
                    if (myHash != lastHash) {

                        // We crossed a cell boundary! Recompute the neighbor list exactly once.
                        lastHash = myHash;
                        validNeighbors = 0;

                        // =========================================================
                        // RE-CALCULATE GRID COORDS
                        // =========================================================
                        float shiftedX = r_pX[i] + (WORLD_SIZE * 0.5f);
                        float shiftedY = r_pY[i] + (WORLD_SIZE * 0.5f);
                        
                        int gx = (int)(shiftedX * INV_CELL_SIZE);
                        int gy = (int)(shiftedY * INV_CELL_SIZE);
                        int gz = 0; // Default for 2D
                        
                        if constexpr (!Is2D) {
                            float shiftedZ = r_pZ[i] + (WORLD_SIZE * 0.5f);
                            gz = (int)(shiftedZ * INV_CELL_SIZE);
                        }

                        // =========================================================
                        // THE BROAD PHASE: CHECK 9 (2D) OR 27 (3D) NEIGHBORING CELLS
                        // =========================================================
                        // By setting the Z loop bounds dynamically outside the loop, 
                        // we avoid putting 'if' statements inside the hot loop!
                        int zStart = Is2D ? 0 : -1;
                        int zEnd   = Is2D ? 0 : 1;
                    
                        // --- THE BROAD PHASE (Check neighboring cells) ---
                        for (int dz = zStart; dz <= zEnd; ++dz) {
                            for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                    int nx = gx + dx;
                                    int ny = gy + dy;
                                    int nz = gz + dz;

                                    // STRICT BOUNDS CHECK: This perfectly protects our 1024-size LUT!
                                    // If it passes this check, nx, ny, and nz are guaranteed to be between 0 and 501.
                                    // X/Y/Z Bounds check (prevents looking outside our spatial array limits)
                                    if (nx < 0 || nx >= GRID_WIDTH || 
                                        ny < 0 || ny >= GRID_HEIGHT || 
                                        (!Is2D && (nz < 0 || nz >= GRID_HEIGHT))) continue;

                                    uint32_t nHash;

                                    if constexpr (Is2D) {
                                        // 2D Grid Hash
                                        nHash = getMortonCode<IsLegacy>((uint32_t)nx, (uint32_t)ny, 0) & HASH_MASK;
                                    } 
                                    else {
                                        // 3D Grid Hash
                                        nHash = getMortonCode<IsLegacy>((uint32_t)nx, (uint32_t)ny, (uint32_t)nz) & HASH_MASK;
                                    }

                                    // ==========================================
                                    // NEW: THE L1 CACHE BROAD-PHASE CULL
                                    // ==========================================
                                    // Check if the 64-bit integer has a 1 at this exact bit position.
                                    // If it evaluates to 0, the cell is mathematically guaranteed to be empty.
                                    if ((r_cellOccupancyMask[nHash >> 6] & (1ULL << (nHash & 63))) == 0) {
                                        continue; 
                                    }

                                    // Only fetch from the 1MB L3 cache if the cell is actually occupied!
                                    uint32_t startIdx = r_cellStartOffset[nHash];
                                    uint32_t endIdx = (nHash + 1 < TOTAL_CELLS) ? r_cellStartOffset[nHash + 1] : activeCount;

                                    // Only add to our local cache if the cell actually has particles
                                    if (startIdx < endIdx) {
                                        neighborCache[validNeighbors].startIdx = startIdx;
                                        // Cap neighbors to prevent processing dense singularities 
                                        neighborCache[validNeighbors].endIdx = std::min(endIdx, startIdx + 16u);
                                        validNeighbors++;
                                    }
                                }
                            }
                        }
                    } // End of Neighbor Cache Builder

                    // =========================================================
                    // THE NARROW PHASE: Loop over our cached neighbors
                    // HOISTED BRANCH: Predictor hits 100%. L1i Cache perfectly dense.
                    // =========================================================
                    if constexpr (Is2D) {
                        for (int n = 0; n < validNeighbors; ++n) {
                            int j = neighborCache[n].startIdx;
                            int maxNeighborsToCheck = neighborCache[n].endIdx;

                            /* [CPU Execution Ports]
                                - A single CPU core contains multiple execution units.
                                - Instructions route to lanes or ports (Port 0, Port 1, etc..).
                                - These execution ports are pipelined (like an assembly line).
                                - Instruction level parallelism means two instruction in a single cycle put on port 0 and port 1.
                            */

                            // --- 16-WIDE UNROLLED ZERO-SPILL 2D NARROW PHASE ---
                            // Process 16 particles simultaneously to hide FMA latency
                            // for (; j <= maxNeighborsToCheck - 16; j += 16) {
                            //     localThreadOps += 24; // 12 FLOPs * 2 batches

                            //     // 2. INTERLEAVED LOADS (No Z-Cull needed)
                            //     // __m256 p_j_x_A = _mm256_loadu_ps(&pX[j]);
                            //     // __m256 p_j_x_B = _mm256_loadu_ps(&pX[j + 8]);
                                
                            //     // __m256 p_j_y_A = _mm256_loadu_ps(&pY[j]);
                            //     // __m256 p_j_y_B = _mm256_loadu_ps(&pY[j + 8]);

                            //     // 3. INTERLEAVED DISTANCES
                            //     // __m256 diffX_A = _mm256_sub_ps(p_i_x, p_j_x_A);
                            //     // __m256 diffX_B = _mm256_sub_ps(p_i_x, p_j_x_B);
                                
                            //     // __m256 diffY_A = _mm256_sub_ps(p_i_y, p_j_y_A);
                            //     // __m256 diffY_B = _mm256_sub_ps(p_i_y, p_j_y_B);

                            //     // 2. FOLDED MATH
                            //     __m256 diffX_A = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(&pX[j]));
                            //     __m256 diffX_B = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(&pX[j + 8]));
                            //     __m256 diffY_A = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(&pY[j]));
                            //     __m256 diffY_B = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(&pY[j + 8]));

                            //     __m256 distSq_A = _mm256_fmadd_ps(diffY_A, diffY_A, _mm256_mul_ps(diffX_A, diffX_A));
                            //     __m256 distSq_B = _mm256_fmadd_ps(diffY_B, diffY_B, _mm256_mul_ps(diffX_B, diffX_B));

                            //     // 4. CHECK COLLISION MASKS
                            //     __m256 mask_A = _mm256_and_ps(
                            //         _mm256_cmp_ps(distSq_A, vRadiusSq, _CMP_LT_OQ),
                            //         _mm256_cmp_ps(distSq_A, vEpsilon, _CMP_GT_OQ)
                            //     );
                            //     __m256 mask_B = _mm256_and_ps(
                            //         _mm256_cmp_ps(distSq_B, vRadiusSq, _CMP_LT_OQ),
                            //         _mm256_cmp_ps(distSq_B, vEpsilon, _CMP_GT_OQ)
                            //     );

                            //     // 5. COMBINED CULL
                            //     // If NO particles in BOTH 8-wide batches collided, skip the heavy math entirely!
                            //     __m256 combinedMask = _mm256_or_ps(mask_A, mask_B);
                            //     if (_mm256_testz_ps(combinedMask, combinedMask)) continue;

                            //     // 6. INTERLEAVED NEWTON-RAPHSON
                            //     // __m256 distSq_eps_A = _mm256_add_ps(distSq_A, vEpsilon);
                            //     // __m256 distSq_eps_B = _mm256_add_ps(distSq_B, vEpsilon);

                            //     // 3. VARIABLE RECYCLING
                            //     distSq_A = _mm256_add_ps(distSq_A, vEpsilon);
                            //     distSq_B = _mm256_add_ps(distSq_B, vEpsilon);

                            //     // Hardware 12-bit guess
                            //     // __m256 rsqrt_approx_A = _mm256_rsqrt_ps(distSq_eps_A);
                            //     // __m256 rsqrt_approx_B = _mm256_rsqrt_ps(distSq_eps_B);

                            //     __m256 rsqrt_A = _mm256_rsqrt_ps(distSq_A);
                            //     __m256 rsqrt_B = _mm256_rsqrt_ps(distSq_B);

                            //     // __m256 half_x_A = _mm256_mul_ps(distSq_eps_A, half);
                            //     // __m256 half_x_B = _mm256_mul_ps(distSq_eps_B, half);

                            //     // __m256 y_sq_A = _mm256_mul_ps(rsqrt_approx_A, rsqrt_approx_A);
                            //     // __m256 y_sq_B = _mm256_mul_ps(rsqrt_approx_B, rsqrt_approx_B);

                            //     // __m256 term_A = _mm256_fnmadd_ps(half_x_A, y_sq_A, three_halves);
                            //     // __m256 term_B = _mm256_fnmadd_ps(half_x_B, y_sq_B, three_halves);

                            //     // __m256 invDistApprox_A = _mm256_mul_ps(rsqrt_approx_A, term_A);
                            //     // __m256 invDistApprox_B = _mm256_mul_ps(rsqrt_approx_B, term_B);

                            //     // 7. INTERLEAVED FORCE ACCUMULATION
                            //     // __m256 push_A = _mm256_and_ps(_mm256_mul_ps(invDistApprox_A, vRepulsion), mask_A);
                            //     // __m256 push_B = _mm256_and_ps(_mm256_mul_ps(invDistApprox_B, vRepulsion), mask_B);

                            //     // 4. DEEPLY FOLDED NEWTON-RAPHSON & PUSH
                            //     __m256 push_A = _mm256_and_ps(
                            //         _mm256_mul_ps(_mm256_mul_ps(rsqrt_A, _mm256_fnmadd_ps(_mm256_mul_ps(distSq_A, half), _mm256_mul_ps(rsqrt_A, rsqrt_A), three_halves)), vRepulsion), 
                            //         mask_A
                            //     );
                            //     __m256 push_B = _mm256_and_ps(
                            //         _mm256_mul_ps(_mm256_mul_ps(rsqrt_B, _mm256_fnmadd_ps(_mm256_mul_ps(distSq_B, half), _mm256_mul_ps(rsqrt_B, rsqrt_B), three_halves)), vRepulsion), 
                            //         mask_B
                            //     );

                            //     accX = _mm256_fmadd_ps(diffX_A, push_A, accX);
                            //     accX = _mm256_fmadd_ps(diffX_B, push_B, accX);

                            //     accY = _mm256_fmadd_ps(diffY_A, push_A, accY);
                            //     accY = _mm256_fmadd_ps(diffY_B, push_B, accY);
                            // }

                            // --- 2D NARROW PHASE ZERO-SPILL (8-wide, Max Speed, No Z-Math, Pure 2D AVX2 Fast Path) ---
                            for (; j <= maxNeighborsToCheck - 8; j += 8) {
                                localThreadOps += 12; // 2D FLOP count

                                /* [EXPLICIT PREFETCHING]
                                    - CPUs memory controller races ahead of the math.
                                    - Sends the neighbors coordinates from L3 cache to L1 cache.
                                    - Its in L1 cache by the time you need the next neighbor's coordinates.
                                */

                                // 1. LOAD X & Y IMMEDIATELY (No Z-Cull)
                                // __m256 p_j_x = _mm256_loadu_ps(&pX[j]);
                                // __m256 p_j_y = _mm256_loadu_ps(&pY[j]);

                                // 2. CALCULATE 2D DISTANCES
                                // __m256 diffX = _mm256_sub_ps(p_i_x, p_j_x);
                                // __m256 diffY = _mm256_sub_ps(p_i_y, p_j_y);

                                // 2. FOLDED X & Y MATH (Saves 2 Registers)
                                // The load is nested directly inside the subtraction!
                                // The CPU reads straight from the L1 Cache into the ALU execution port.
                                __m256 diffX = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(r_pX + j));
                                __m256 diffY = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(r_pY + j));

                                // 3. DISTANCE SQUARED (distSq = dx*dx + dy*dy)
                                __m256 distSq = _mm256_fmadd_ps(diffY, diffY, _mm256_mul_ps(diffX, diffX));
                                
                                // 4. CHECK COLLISION MASK
                                // (distSq < RadiusSq AND distSq > 0.0001)
                                __m256 mask = _mm256_and_ps(
                                    _mm256_cmp_ps(distSq, vRadiusSq, _CMP_LT_OQ),
                                    _mm256_cmp_ps(distSq, vEpsilon, _CMP_GT_OQ)
                                );

                                // 5. CULL: If no particles in this 8-wide batch collided, skip the heavy math!
                                if (_mm256_testz_ps(mask, mask)) continue;

                                // 6. THE HEAVY MATH (Newton-Raphson approximation)
                                // __m256 invDistApprox = _mm256_rsqrt_ps(distSq);

                                // __m256 distSq_eps = _mm256_add_ps(distSq, vEpsilon);

                                // 6. VARIABLE RECYCLING & MASK PRE-CALCULATION
                                // Overwrite distSq with the epsilon-padded version to save register space
                                distSq = _mm256_add_ps(distSq, vEpsilon);

                                // [LATENCY HIDING]: Calculate masked repulsion early! (1 cycle)
                                // We do this now so the ALU port isn't idle while the rsqrt begins.
                                __m256 masked_repulsion = _mm256_and_ps(vRepulsion, mask);

                                // 7. DEEPLY FOLDED NEWTON-RAPHSON & PUSH
                                // Newton-Raphson Refinement (Restores 23-bit precision)
                                // Formula: y = y * (1.5 - 0.5 * x * y^2)
                                // Get the hardware's 12-bit guess (~5-7 cycles)
                                __m256 rsqrt_approx = _mm256_rsqrt_ps(distSq); 
                                __m256 half_x = _mm256_mul_ps(distSq, half);
                                __m256 y_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);

                                // Calculate Push Force
                                // __m256 push = _mm256_mul_ps(invDistApprox, vRepulsion);
                                // push = _mm256_and_ps(push, mask); // Zero out forces for particles that didn't collide

                                /// [LATENCY HIDING]: Multiply the masked repulsion by the hardware guess!
                                // This executes perfectly in parallel while the 'term' below computes.
                                __m256 rsqrt_repul = _mm256_mul_ps(rsqrt_approx, masked_repulsion);

                                // _mm256_fnmadd_ps perfectly calculates: -(half_x * y_sq) + 1.5 (~4-5 cycles)
                                __m256 term = _mm256_fnmadd_ps(half_x, y_sq, three_halves);

                                // Calculate final push force
                                // We already applied the mask and the repulsion, so this is just one final rapid multiply.
                                __m256 push = _mm256_mul_ps(rsqrt_repul, term);

                                // 7. ACCUMULATE 2D FORCES
                                accX = _mm256_fmadd_ps(diffX, push, accX);
                                accY = _mm256_fmadd_ps(diffY, push, accY);
                            }

                            // --- 2D SCALAR REMAINDER  ---
                            for (; j < maxNeighborsToCheck; ++j) {
                                float diffX = r_pX[i] - r_pX[j];
                                float diffY = r_pY[i] - r_pY[j];

                                float distSq = diffX*diffX + diffY*diffY;
                                if (distSq > 0.0001f && distSq < RADIUS_SQ) {
                                    // 1. Hardware 12-bit approximation (~4 clock cycles)
                                    float approx = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(distSq + 0.0001f)));

                                    // 2. Newton-Raphson Refinement to 23-bit precision (Mapped 1:1 with your AVX2 logic)
                                    float invDist = approx * (1.5f - (0.5f * (distSq + 0.0001f) * approx * approx));
                                    // float invDist = 1.0f / std::sqrt(distSq);

                                    float push = REPULSION_STRENGTH * invDist;
                                    scalarForceX += diffX * push;
                                    scalarForceY += diffY * push;
                                }
                            }
                        }
                    } 
                    else {     
                        for (int n = 0; n < validNeighbors; ++n) {
                            int j = neighborCache[n].startIdx;
                            int maxNeighborsToCheck = neighborCache[n].endIdx;

                            // --- 16-WIDE ZERO-SPILL 3D NARROW PHASE ---
                            // for (; j <= maxNeighborsToCheck - 16; j += 16) {
                            //     localThreadOps += 30; // 15 FLOPs * 2 batches

                            //     // 1. EXPLICIT PREFETCHING (Look 16 floats ahead instead of 8)
                            //     // Tell the memory controller to pull the NEXT 16 floats into the L1 cache as the execution ports crunch the current 16 floats.
                            //     // _mm_prefetch((const char*)&pZ[j + 16], _MM_HINT_T0);
                            //     // _mm_prefetch((const char*)&pX[j + 16], _MM_HINT_T0);
                            //     // _mm_prefetch((const char*)&pY[j + 16], _MM_HINT_T0);

                            //     // 2. INTERLEAVED Z-STACK CULL
                            //     // __m256 p_j_z_A = _mm256_loadu_ps(&pZ[j]);
                            //     // __m256 p_j_z_B = _mm256_loadu_ps(&pZ[j + 8]);

                            //     // __m256 diffZ_A = _mm256_sub_ps(p_i_z, p_j_z_A);
                            //     // __m256 diffZ_B = _mm256_sub_ps(p_i_z, p_j_z_B);

                            //     // 2. FOLDED Z-CULL (Loads directly into subtraction)
                            //     __m256 diffZ_A = _mm256_sub_ps(p_i_z, _mm256_loadu_ps(&pZ[j]));
                            //     __m256 diffZ_B = _mm256_sub_ps(p_i_z, _mm256_loadu_ps(&pZ[j + 8]));

                            //     __m256 zDistSq_A = _mm256_mul_ps(diffZ_A, diffZ_A);
                            //     __m256 zDistSq_B = _mm256_mul_ps(diffZ_B, diffZ_B);

                            //     // __m256 zMask_A = _mm256_cmp_ps(zDistSq_A, vRadiusSq, _CMP_LT_OQ);
                            //     // __m256 zMask_B = _mm256_cmp_ps(zDistSq_B, vRadiusSq, _CMP_LT_OQ);

                            //     // Combine masks: If ALL 16 particles fail the Z-cull, skip the heavy math!
                            //     __m256 combinedZMask = _mm256_or_ps(
                            //         _mm256_cmp_ps(zDistSq_A, vRadiusSq, _CMP_LT_OQ),
                            //         _mm256_cmp_ps(zDistSq_B, vRadiusSq, _CMP_LT_OQ)
                            //     );

                            //     // __m256 combinedZMask = _mm256_or_ps(zMask_A, zMask_B);

                            //     if (_mm256_testz_ps(combinedZMask, combinedZMask)) continue;

                            //     // 3. INTERLEAVED SURVIVOR MATH (X and Y loads)
                            //     // __m256 p_j_x_A = _mm256_loadu_ps(&pX[j]);
                            //     // __m256 p_j_x_B = _mm256_loadu_ps(&pX[j + 8]);
                            //     // __m256 p_j_y_A = _mm256_loadu_ps(&pY[j]);
                            //     // __m256 p_j_y_B = _mm256_loadu_ps(&pY[j + 8]);

                            //     // __m256 diffX_A = _mm256_sub_ps(p_i_x, p_j_x_A);
                            //     // __m256 diffX_B = _mm256_sub_ps(p_i_x, p_j_x_B);
                            //     // __m256 diffY_A = _mm256_sub_ps(p_i_y, p_j_y_A);
                            //     // __m256 diffY_B = _mm256_sub_ps(p_i_y, p_j_y_B);

                            //     // 3. FOLDED SURVIVOR MATH
                            //     __m256 diffX_A = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(&pX[j]));
                            //     __m256 diffX_B = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(&pX[j + 8]));
                            //     __m256 diffY_A = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(&pY[j]));
                            //     __m256 diffY_B = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(&pY[j + 8]));

                            //     // Interleaved Distance Squared
                            //     __m256 distSq_A = _mm256_fmadd_ps(diffY_A, diffY_A, _mm256_fmadd_ps(diffX_A, diffX_A, zDistSq_A));
                            //     __m256 distSq_B = _mm256_fmadd_ps(diffY_B, diffY_B, _mm256_fmadd_ps(diffX_B, diffX_B, zDistSq_B));

                            //     __m256 mask_A = _mm256_and_ps(_mm256_cmp_ps(distSq_A, vRadiusSq, _CMP_LT_OQ), _mm256_cmp_ps(distSq_A, vEpsilon, _CMP_GT_OQ));
                            //     __m256 mask_B = _mm256_and_ps(_mm256_cmp_ps(distSq_B, vRadiusSq, _CMP_LT_OQ), _mm256_cmp_ps(distSq_B, vEpsilon, _CMP_GT_OQ));

                            //     // Combine final 3D masks
                            //     __m256 combinedMask = _mm256_or_ps(mask_A, mask_B);
                            //     if (_mm256_testz_ps(combinedMask, combinedMask)) continue;

                            //     // 4. INTERLEAVED NEWTON-RAPHSON
                            //     // __m256 distSq_eps_A = _mm256_add_ps(distSq_A, vEpsilon);
                            //     // __m256 distSq_eps_B = _mm256_add_ps(distSq_B, vEpsilon);

                            //     // 4. VARIABLE RECYCLING
                            //     distSq_A = _mm256_add_ps(distSq_A, vEpsilon);
                            //     distSq_B = _mm256_add_ps(distSq_B, vEpsilon);

                            //     // __m256 rsqrt_approx_A = _mm256_rsqrt_ps(distSq_eps_A);
                            //     // __m256 rsqrt_approx_B = _mm256_rsqrt_ps(distSq_eps_B);

                            //     __m256 rsqrt_A = _mm256_rsqrt_ps(distSq_A);
                            //     __m256 rsqrt_B = _mm256_rsqrt_ps(distSq_B);

                            //     // __m256 half_x_A = _mm256_mul_ps(distSq_eps_A, half);
                            //     // __m256 half_x_B = _mm256_mul_ps(distSq_eps_B, half);

                            //     // __m256 y_sq_A = _mm256_mul_ps(rsqrt_approx_A, rsqrt_approx_A);
                            //     // __m256 y_sq_B = _mm256_mul_ps(rsqrt_approx_B, rsqrt_approx_B);

                            //     // __m256 term_A = _mm256_fnmadd_ps(half_x_A, y_sq_A, three_halves);
                            //     // __m256 term_B = _mm256_fnmadd_ps(half_x_B, y_sq_B, three_halves);

                            //     // __m256 invDistApprox_A = _mm256_mul_ps(rsqrt_approx_A, term_A);
                            //     // __m256 invDistApprox_B = _mm256_mul_ps(rsqrt_approx_B, term_B);

                            //     // 5. INTERLEAVED FORCE ACCUMULATION
                            //     // __m256 push_A = _mm256_and_ps(_mm256_mul_ps(invDistApprox_A, vRepulsion), mask_A);
                            //     // __m256 push_B = _mm256_and_ps(_mm256_mul_ps(invDistApprox_B, vRepulsion), mask_B);

                            //     // 5. DEEPLY FOLDED NEWTON-RAPHSON & PUSH
                            //     __m256 push_A = _mm256_and_ps(
                            //         _mm256_mul_ps(_mm256_mul_ps(rsqrt_A, _mm256_fnmadd_ps(_mm256_mul_ps(distSq_A, half), _mm256_mul_ps(rsqrt_A, rsqrt_A), three_halves)), vRepulsion), 
                            //         mask_A
                            //     );
                            //     __m256 push_B = _mm256_and_ps(
                            //         _mm256_mul_ps(_mm256_mul_ps(rsqrt_B, _mm256_fnmadd_ps(_mm256_mul_ps(distSq_B, half), _mm256_mul_ps(rsqrt_B, rsqrt_B), three_halves)), vRepulsion), 
                            //         mask_B
                            //     );

                            //     accX = _mm256_fmadd_ps(diffX_A, push_A, accX);
                            //     accX = _mm256_fmadd_ps(diffX_B, push_B, accX);

                            //     accY = _mm256_fmadd_ps(diffY_A, push_A, accY);
                            //     accY = _mm256_fmadd_ps(diffY_B, push_B, accY);

                            //     accZ = _mm256_fmadd_ps(diffZ_A, push_A, accZ);
                            //     accZ = _mm256_fmadd_ps(diffZ_B, push_B, accZ);
                            // }

                            // --- 3D NARROW PHASE (8-Wide AVX2 Full Z-Stack Cull) ---
                            // Process neighbors 8 at a time, bounded to our new capped limit (maxNeighborsToCheck).
                            for (; j <= maxNeighborsToCheck - 8; j += 8) {
                                localThreadOps += 15; // Exact cost of processing 8 neighbors! 3D FLOP Count

                                // EXPLICIT PREFETCHING (Z first, because of Z-cull)
                                // _mm_prefetch((const char*)&pZ[j + 8], _MM_HINT_T0);
                                // _mm_prefetch((const char*)&pX[j + 8], _MM_HINT_T0);
                                // _mm_prefetch((const char*)&pY[j + 8], _MM_HINT_T0);

                                // 1. THE Z-STACK CULL: Load Z-axis data FIRST
                                // __m256 p_j_z = _mm256_loadu_ps(&pZ[j]);

                                // 1. FOLDED Z-CULL (Saves 1 Register)
                                // We inline the load directly into the subtraction. 
                                // The compiler emits: vsubps ymm0, ymm1, YMMWORD PTR [mem]
                                __m256 diffZ = _mm256_sub_ps(p_i_z, _mm256_loadu_ps(r_pZ + j));

                                // Calculate just the Z distance squared
                                __m256 zDistSq = _mm256_mul_ps(diffZ, diffZ);

                                // Did ANY of these 8 particles pass the vertical threshold? (zDistSq < RadiusSq)
                                __m256 zMask = _mm256_cmp_ps(zDistSq, vRadiusSq, _CMP_LT_OQ);

                                // Z-Cull: Only triggers if depth is actually being used
                                // HUGE OPTIMIZATION: If all 8 particles are too far away vertically,
                                // testz returns 1 (true). We skip loading X, Y, and the heavy math!
                                if (_mm256_testz_ps(zMask, zMask)) continue;

                                // =======================================================
                                // 2. SURVIVED CULL: Now we load the rest of the data
                                // =======================================================

                                // Load 8 neighbors simultaneously (Thanks to our SoA Sort, this is L1 Cache perfectly linear!)
                                // __m256 p_j_x = _mm256_loadu_ps(&pX[j]);
                                // __m256 p_j_y = _mm256_loadu_ps(&pY[j]);

                                // Calculate distances: dx, dy, dz
                                // 2. FOLDED X & Y SURVIVOR MATH (Saves 2 Registers)
                                __m256 diffX = _mm256_sub_ps(p_i_x, _mm256_loadu_ps(r_pX + j));
                                __m256 diffY = _mm256_sub_ps(p_i_y, _mm256_loadu_ps(r_pY + j));

                                
                                // 3. FULL DISTANCE SQUARED
                                // Notice we reuse 'zDistSq' here so we don't calculate it twice!
                                __m256 distSq = _mm256_fmadd_ps(diffY, diffY, _mm256_fmadd_ps(diffX, diffX, zDistSq));

                                // Did any of these 8 particles actually collide in 3D space?
                                // (distSq < RadiusSq AND distSq > 0.0001)
                                __m256 mask = _mm256_and_ps(
                                    _mm256_cmp_ps(distSq, vRadiusSq, _CMP_LT_OQ),
                                    _mm256_cmp_ps(distSq, vEpsilon, _CMP_GT_OQ)
                                );

                                // HUGE OPTIMIZATION: If all 8 bits are zero (no collisions), skip the heavy math entirely!
                                // Secondary check: Even if they passed the Z-cull, they might have failed the X/Y cull.
                                if (_mm256_testz_ps(mask, mask)) continue;

                                /* [Newton-Raphson approximation] 
                                    - Refines the hardware's approximation of 1/sqrt(distSq)
                                    - This instruction provides a maximum relative error of 1.5x10^-4.
                                    - If the true distance is 1000, the hardware approximation will give you 1000.15
                                    - That 0.015% margin of error is invisible to the human eye, but the computational cost is massive.
                                    - 50 million ALU operations every second = 4 execution ports per particle * 100,0000 particles * 120fps
                                */ 
                                // THE HEAVY MATH (Newton-Raphson approximation)
                                // __m256 invDistApprox = _mm256_rsqrt_ps(distSq);

                                // __m256 distSq_eps = _mm256_add_ps(distSq, vEpsilon);

                                // 4. VARIABLE RECYCLING & MASK PRE-CALCULATION
                                distSq = _mm256_add_ps(distSq, vEpsilon);

                                // [LATENCY HIDING]: Calculate the masked repulsion early!
                                // This executes in 1 clock cycle on a generic ALU port while the heavier math below is starting up.
                                __m256 masked_repulsion = _mm256_and_ps(vRepulsion, mask);

                                // 5. NEWTON-RAPHSON (The Deep Chain)
                                // Hardware guess takes ~5-7 cycles
                                // Get the hardware's 12-bit guess
                                // Notice how tight this is. The CPU can now hold accX, accY, accZ, 
                                // diffX, diffY, diffZ, and all these temporaries perfectly in 16 registers.
                                __m256 rsqrt_approx = _mm256_rsqrt_ps(distSq); 

                                // Newton-Raphson Refinement (Restores 23-bit precision)
                                // Formula: y = y * (1.5 - 0.5 * x * y^2)
                                __m256 half_x = _mm256_mul_ps(distSq, half);

                                // Wait for rsqrt_approx (~4 cycles)
                                __m256 y_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);

                                // [LATENCY HIDING]: Multiply repulsion by our rsqrt guess right now!
                                // We do this concurrently while waiting for the 'term' below to finish calculating.
                                __m256 rsqrt_repul = _mm256_mul_ps(rsqrt_approx, masked_repulsion);

                                // _mm256_fnmadd_ps perfectly calculates: -(half_x * y_sq) + 1.5
                                __m256 term = _mm256_fnmadd_ps(half_x, y_sq, three_halves);
                                
                                // __m256 invDistApprox = _mm256_mul_ps(rsqrt_approx, term);
                                
                                // Calculate Push Force
                                // We fold the mask ANDing directly into the push multiplier! (Saves 1 Register)
                                // __m256 push = _mm256_mul_ps(invDistApprox, vRepulsion);
                                // push = _mm256_and_ps(push, mask); // Zero out forces for particles that didn't collide

                                // 6. FOLDED PUSH
                                // We already applied the mask and the repulsion, so this is just one final rapid multiply.
                                __m256 push = _mm256_mul_ps(rsqrt_repul, term);

                                // 7. Accumulate Force Vector
                                accX = _mm256_fmadd_ps(diffX, push, accX);
                                accY = _mm256_fmadd_ps(diffY, push, accY);
                                accZ = _mm256_fmadd_ps(diffZ, push, accZ);
                            }

                            // --- 3D SCALAR REMAINDER ---
                            // Handle the remaining 1 to 7 particles that didn't cleanly fit into an 8-wide AVX register
                            for (; j < maxNeighborsToCheck; ++j) {
                                float diffX = r_pX[i] - r_pX[j];
                                float diffY = r_pY[i] - r_pY[j];
                                float diffZ = r_pZ[i] - r_pZ[j]; // (3D only)
                                
                                float distSq = diffX*diffX + diffY*diffY + diffZ*diffZ;
                                
                                // Epsilon check inherently prevents particle 'i' from violently colliding with itself
                                if (distSq > 0.0001f && distSq < RADIUS_SQ) {
                                    // 1. Hardware 12-bit approximation (~4 clock cycles)
                                    float approx = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(distSq + 0.0001f)));

                                    // 2. Newton-Raphson Refinement to 23-bit precision (Mapped 1:1 with your AVX2 logic)
                                    float invDist = approx * (1.5f - (0.5f * (distSq + 0.0001f) * approx * approx));
                                    // float invDist = 1.0f / std::sqrt(distSq);

                                    float push = REPULSION_STRENGTH * invDist;
                                    scalarForceX += diffX * push;
                                    scalarForceY += diffY * push;
                                    scalarForceZ += diffZ * push;
                                }
                            }
                        }
                    }

                    // ==========================================
                    // THE ACCUMULATOR DUMP (Split for 2D/3D)
                    // ==========================================
            
                    // Dump the AVX2 accumulator registers back to standard floats using ALIGNED stores
                    // _mm256_store_ps(tempX, accX);
                    // _mm256_store_ps(tempY, accY);
                    // _mm256_store_ps(tempZ, accZ);

                    // Add up all the vectors
                    // for (int k = 0; k < 8; ++k) {
                    //     scalarForceX += tempX[k];
                    //     scalarForceY += tempY[k];
                    //     scalarForceZ += tempZ[k];
                    // }

                    // X and Y are ALWAYS needed
                    // Instantly collapse the AVX2 accumulator registers using pure silicon math
                    scalarForceX += hsum_avx2(accX);
                    scalarForceY += hsum_avx2(accY);

                    // Apply the final accumulated push force directly to Particle i's velocity
                    r_vX[i] += scalarForceX;
                    r_vY[i] += scalarForceY;
                    // vZ[i] += scalarForceZ;

                    // Only dump Z and modify velocity if we are actually in 3D
                    if constexpr (!Is2D) {
                        // _mm256_storeu_ps(tempZ, accZ); // Unaligned store
                        // _mm256_store_ps(tempZ, accZ);
                        // for (int k = 0; k < 8; ++k) {
                        //     scalarForceZ += tempZ[k];
                        // }
                        scalarForceZ += hsum_avx2(accZ);
                        r_vZ[i] += scalarForceZ;
                    }
                }
                // Safely write to this specific thread's isolated cache line
                g_JobSystem.threadStats[tl_workerIndex]->totalFlops.fetch_add(localThreadOps, std::memory_order_relaxed);
            };

            // ======================================
            // TEMPLATED KERNEL DISPATCHER
            // ======================================
            /*
                - If we use regular booleans for 2D and 3D, the compiled binary will contain assembly instructions for both 2D and 3D math stitched together with jump commands.
                - This bloats the CPU's ultra-fast L1 instruction cache.

                - Instead we hoist the boolean checks outside the job system lambda using a kernel dispatcher.
                - Now the compiler will generate four specialized, highly optimized copies of this physics loop.
                - When the 2D mode is active, the 3D math does not exist in the execution path.
                - Reduces the size to fit in the ultra-fast L1i cache which prevents micro-stalls in the CPU cores.
            */

            // --- DISPATCH THE KERNEL ---
            if (localIs2D) {
                if (localIsLegacy) collisionKernel.template operator()<true, true>();
                else               collisionKernel.template operator()<true, false>();
            } else {
                if (localIsLegacy) collisionKernel.template operator()<false, true>();
                else               collisionKernel.template operator()<false, false>();
            }
        });
    }

    // --- Dynamic Integration Job System---
    FORCE_INLINE void integrate(float deltaTime, int activeCount, float gravityVal, float mouseX = 0.0f, float mouseY = 0.0f, bool isMouseDown = false) {
        [[assume(activeCount > 0)]]; // C++23 hint

        __m256 dt = _mm256_set1_ps(deltaTime);
        __m256 gravityStrength = _mm256_set1_ps(gravityVal); // Tune this for more "violence"
        __m256 epsilon = _mm256_set1_ps(0.001f);       // Prevents infinite force at center
        __m256 damping = _mm256_set1_ps(1.0f);        // Slight friction to keep it stable

        // SIMD requires multiples of 8. If the user slider is at 10,003, we pad it up to 10,008.
        // Because of our memory pool, it is perfectly safe to calculate slightly past the visible limit.
        int paddedActiveCount = (activeCount + 7) & ~7; // Prevents crashing (i.e., no more trying to read unallocated memory).

        // 1. Ask the Job System how many threads are currently awake
        // Ask the actual atomic counter, not the maximum capacity!
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);

        // 2. Over-subscribe the threads (target 16 to 32 chunks per available thread for optimal stealing to ensure it fits into L1 caches of a 4-core, 8-core, 16-core processor)
        const uint32_t targetChunksPerThread = 16;

        // Split the workload into chunks to feed the Job System
        // Profile this chunk size [Too small] = too much overhead, [Too big] = cores idle.
        uint32_t CHUNK_SIZE = paddedActiveCount / (threadCount * targetChunksPerThread);
        
        // 3. Clamp it to safe AVX boundaries and L1 Cache limits
        // Minimum 256 particles to absorb coroutine overhead, maximum 2048 to stay in L1 cache.
        // 512 is usually the sweet spot for AVX2 Coroutine jobs. 
        // Anything smaller and the overhead of 'co_await' starts to cost more than the math.
        CHUNK_SIZE = std::clamp(CHUNK_SIZE, 512u, 2048u);

        // AVX2 requires multiples of 8!
        CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7;

        // Dispatch replaces the OpenMP Pragma
        g_JobSystem.DispatchAndWait(paddedActiveCount, CHUNK_SIZE, [&](uint32_t start, uint32_t end) {

            // Ensure our start and end align to AVX boundaries
            uint32_t alignedStart = start & ~7;
            uint32_t alignedEnd = (end + 7) & ~7;

            // Extract restricted pointers
            float* ENGINE_RESTRICT r_pX = std::assume_aligned<32>(pX.data());
            float* ENGINE_RESTRICT r_pY = std::assume_aligned<32>(pY.data());
            float* ENGINE_RESTRICT r_pZ = std::assume_aligned<32>(pZ.data());
            float* ENGINE_RESTRICT r_vX = std::assume_aligned<32>(vX.data());
            float* ENGINE_RESTRICT r_vY = std::assume_aligned<32>(vY.data());
            float* ENGINE_RESTRICT r_vZ = std::assume_aligned<32>(vZ.data());

            // ==========================================
            // HOISTED CONSTANTS (Shared by both loops)
            // ==========================================

            // HOISTED MOUSE REGISTERS:
            __m256 mx = _mm256_set1_ps(mouseX);
            __m256 my = _mm256_set1_ps(mouseY);
            __m256 mouseRadiusSq = _mm256_set1_ps(150.0f * 150.0f); // Only repel particles within a radius of 150 units from the mouse
            __m256 mouseForce = _mm256_set1_ps(500.0f);

            // Keep particles inside a 1900x1900 box (Slightly smaller than WORLD_SIZE to prevent grid array out-of-bounds)
            __m256 vMaxBound = _mm256_set1_ps(950.0f);
            __m256 vMinBound = _mm256_set1_ps(-950.0f);
            __m256 vBounceDamp = _mm256_set1_ps(-0.8f); // Reverse direction and lose 20% kinetic energy

            __m256 three_halves = _mm256_set1_ps(1.5f);
            __m256 half = _mm256_set1_ps(0.5f);

            // ONLY do Z-math and Z-memory writes if we are actually in 3D mode
            if (g_EngineSettings.is2DMode) {
                // ==========================================
                // PURE 2D LOOP: Maximum Bandwidth, Zero Z-Axis RAM Traffic (2D: 2x performance improvement over 3D)
                // ==========================================
                for (int i = alignedStart; i < alignedEnd; i += 8) {
                    // 1. LOAD: Positions and Velocities (X and Y ONLY)
                    __m256 px = _mm256_load_ps(r_pX + i);
                    __m256 py = _mm256_load_ps(r_pY + i);
                    __m256 vx = _mm256_load_ps(r_vX + i);
                    __m256 vy = _mm256_load_ps(r_vY + i);

                    // 2. MATH: Distance to Center (0,0)
                    __m256 distSq = _mm256_mul_ps(px, px);
                    distSq = _mm256_fmadd_ps(py, py, distSq);

                    // Fast Hardware Approximation (Bypasses Newton-Raphson)
                    __m256 distSq_eps = _mm256_add_ps(distSq, epsilon);

                    // Newton-Raphson Refinement
                    // Formula: y = y * (1.5 - 0.5 * x * y * y)
                    __m256 rsqrt_approx = _mm256_rsqrt_ps(distSq_eps);
                    __m256 half_x = _mm256_mul_ps(distSq_eps, half);
                    __m256 y_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);
                    __m256 term = _mm256_fnmadd_ps(half_x, y_sq, three_halves);
                    __m256 invDist = _mm256_mul_ps(rsqrt_approx, term);

                    // 3. GRAVITY CALCULATION
                    __m256 pull = _mm256_mul_ps(invDist, gravityStrength);

                    // --- MOUSE REPULSION ---
                    if (isMouseDown) {
                        __m256 diffMouseX = _mm256_sub_ps(px, mx);
                        __m256 diffMouseY = _mm256_sub_ps(py, my);

                        __m256 mouseDistSq = _mm256_fmadd_ps(diffMouseY, diffMouseY, _mm256_mul_ps(diffMouseX, diffMouseX));
                        __m256 mouseMask = _mm256_cmp_ps(mouseDistSq, mouseRadiusSq, _CMP_LT_OQ);

                        if (!_mm256_testz_ps(mouseMask, mouseMask)) {
                            __m256 invMouseDist = _mm256_rsqrt_ps(_mm256_add_ps(mouseDistSq, epsilon));
                            __m256 mousePush = _mm256_mul_ps(invMouseDist, mouseForce); 
                            mousePush = _mm256_and_ps(mousePush, mouseMask);

                            vx = _mm256_fmadd_ps(diffMouseX, mousePush, vx);
                            vy = _mm256_fmadd_ps(diffMouseY, mousePush, vy);
                        }
                    }
                    
                    vx = _mm256_fnmadd_ps(px, pull, vx);
                    vy = _mm256_fnmadd_ps(py, pull, vy);

                    // 4. APPLY DAMPING & UPDATE POSITION
                    vx = _mm256_mul_ps(vx, damping);
                    vy = _mm256_mul_ps(vy, damping);

                    px = _mm256_fmadd_ps(vx, dt, px);
                    py = _mm256_fmadd_ps(vy, dt, py);

                    // 5. AVX2 BOUNDING WALLS
                    __m256 maskX = _mm256_or_ps(_mm256_cmp_ps(px, vMaxBound, _CMP_GT_OQ), _mm256_cmp_ps(px, vMinBound, _CMP_LT_OQ));
                    __m256 maskY = _mm256_or_ps(_mm256_cmp_ps(py, vMaxBound, _CMP_GT_OQ), _mm256_cmp_ps(py, vMinBound, _CMP_LT_OQ));

                    px = _mm256_max_ps(vMinBound, _mm256_min_ps(px, vMaxBound));
                    py = _mm256_max_ps(vMinBound, _mm256_min_ps(py, vMaxBound));

                    __m256 bouncedVx = _mm256_mul_ps(vx, vBounceDamp);
                    __m256 bouncedVy = _mm256_mul_ps(vy, vBounceDamp);

                    vx = _mm256_blendv_ps(vx, bouncedVx, maskX);
                    vy = _mm256_blendv_ps(vy, bouncedVy, maskY);

                    // 6. STORE: Save results (X and Y ONLY)
                    _mm256_store_ps(r_pX + i, px);
                    _mm256_store_ps(r_pY + i, py);
                    _mm256_store_ps(r_vX + i, vx);
                    _mm256_store_ps(r_vY + i, vy);
                }
            }
            else {
                // ==========================================
                // PURE 3D LOOP: Full X/Y/Z loads and stores
                // ==========================================
                // #pragma omp parallel for
                for (int i = alignedStart; i < alignedEnd; i += 8) {
                    // ==========================================
                    // OBJECT-ORIENTED SIMD 
                    // ==========================================
                    /*
                        - Wrapping SIMD intrinsics in objects can sometimes force the compiler to spill registers to the stack.
                        - Reframe from using custom structs with intrinsics.

                          // 1. LOAD: Fetch Positions (Unaligned for safety)
                          SIMDVector8 pos = { 
                              _mm256_loadu_ps(&pX[i]), 
                              _mm256_loadu_ps(&pY[i]), 
                              _mm256_loadu_ps(&pZ[i]) 
                          };
                          // 2. LOAD: Fetch Velocities
                          SIMDVector8 vel = { 
                              _mm256_loadu_ps(&vX[i]), 
                              _mm256_loadu_ps(&vY[i]), 
                              _mm256_loadu_ps(&vZ[i]) 
                          };
                          // 3. MATH: Calculate movement for this frame (vel * dt)
                          vel.mul(dt);
                          // 4. MATH: Apply movement to position (pos + movement)
                          pos.add(vel.x, vel.y, vel.z);
                    */

                    // 1. LOAD: Positions and Velocities from RAM into L1 cache.
                    __m256 px = _mm256_load_ps(r_pX + i);
                    __m256 py = _mm256_load_ps(r_pY + i);
                    __m256 pz = _mm256_load_ps(r_pZ + i);

                    __m256 vx = _mm256_load_ps(r_vX + i);
                    __m256 vy = _mm256_load_ps(r_vY + i);
                    __m256 vz = _mm256_load_ps(r_vZ + i);

                    // 2. MATH: Distance to Center (0,0,0)
                    // Vector to center is just -px, -py, -pz
                    __m256 distSq = _mm256_mul_ps(px, px);
                    distSq = _mm256_fmadd_ps(py, py, distSq);
                    distSq = _mm256_fmadd_ps(pz, pz, distSq);

                    // 3. Fast Hardware Approximation (Before Newton-Raphson) Get the hardware's 12-bit guess ("Normalization" factor)
                    __m256 distSq_eps = _mm256_add_ps(distSq, epsilon);
                    __m256 rsqrt_approx = _mm256_rsqrt_ps(distSq_eps); // _mm256_rsqrt_ps is a lightning-fast hardware approximation of 1/sqrt(x)

                    // 4. Newton-Raphson Refinement (Restores 23-bit precision)
                    // Formula: y = y * (1.5 - 0.5 * x * y * y)
                    __m256 half_x = _mm256_mul_ps(distSq_eps, _mm256_set1_ps(0.5f));
                    __m256 y_sq = _mm256_mul_ps(rsqrt_approx, rsqrt_approx);
                    __m256 term = _mm256_fnmadd_ps(half_x, y_sq, three_halves);
                    __m256 invDist = _mm256_mul_ps(rsqrt_approx, term);

                    // 5. GRAVITY CALCULATION: Normalize the vector (multiply by 1/dist) and scale by gravity strength
                    __m256 pull = _mm256_mul_ps(invDist, gravityStrength);

                    // --- MOUSE REPULSION (AVX2) ---
                    if (isMouseDown) {
                        __m256 diffMouseX = _mm256_sub_ps(px, mx);
                        __m256 diffMouseY = _mm256_sub_ps(py, my);

                        __m256 mouseDistSq = _mm256_fmadd_ps(diffMouseY, diffMouseY, _mm256_mul_ps(diffMouseX, diffMouseX));
                        __m256 mouseMask = _mm256_cmp_ps(mouseDistSq, mouseRadiusSq, _CMP_LT_OQ);

                        if (!_mm256_testz_ps(mouseMask, mouseMask)) {
                            __m256 invMouseDist = _mm256_rsqrt_ps(_mm256_add_ps(mouseDistSq, epsilon));
                            // Massive push force
                            __m256 mousePush = _mm256_mul_ps(invMouseDist, mouseForce); 
                            mousePush = _mm256_and_ps(mousePush, mouseMask);

                            // Add the mouse force to the total velocity!
                            vx = _mm256_fmadd_ps(diffMouseX, mousePush, vx);
                            vy = _mm256_fmadd_ps(diffMouseY, mousePush, vy);
                        }
                    }
                    
                    // v = v + (-p * pull) 
                    // We subtract because the vector to the center is (0 - p)
                    vx = _mm256_fnmadd_ps(px, pull, vx);
                    vy = _mm256_fnmadd_ps(py, pull, vy);
                    vz = _mm256_fnmadd_ps(pz, pull, vz);

                    // 6. UPDATE POSITION & APPLY DAMPING (Friction)
                    vx = _mm256_mul_ps(vx, damping);
                    vy = _mm256_mul_ps(vy, damping);
                    vz = _mm256_mul_ps(vz, damping);

                    // (Optional) Hard limit velocity to prevent grid-skipping
                    // vx = _mm256_max_ps(_mm256_set1_ps(-50.0f), _mm256_min_ps(vx, _mm256_set1_ps(50.0f)));

                    // AVX2 TERMINAL VELOCITY (Prevents Wall Tunneling)
                    // __m256 vMaxSpeed = _mm256_set1_ps(200.0f);
                    // __m256 vMinSpeed = _mm256_set1_ps(-200.0f);
                    // vx = _mm256_max_ps(vMinSpeed, _mm256_min_ps(vx, vMaxSpeed));
                    // vy = _mm256_max_ps(vMinSpeed, _mm256_min_ps(vy, vMaxSpeed));

                    // 7. UPDATE POSITION: p = p + v * dt
                    px = _mm256_fmadd_ps(vx, dt, px);
                    py = _mm256_fmadd_ps(vy, dt, py);
                    pz = _mm256_fmadd_ps(vz, dt, pz);

                    // ==========================================
                    // AVX2 BOUNDING WALLS (Branchless)
                    // ==========================================
                    /* [Branchless AVX2 Walls]
                        - Don't use if statements to flip the velocity because some particles in the 8-wide registers hit the wall, and some won't.
                        - The use of if-statements breaks the parallel lines and causes instruction pipeline to stall from branch mispredictions.
                        - Instead we use SIMD bit-masking and blending by telling AVX2 register to calculate the bounced velocity fror all 8 particles.
                        - Then use bitmask to cleanly blend the bounced velocity with the original velocity, keeping the new trajectory only for the particles that actually touched the wall.
                    */

                    // 8. AVX2 BOUNDING WALLS
                    // A. Detect who crossed the lines (Creates a register of 0xFFFFFFFF or 0x00000000 bitmasks)
                    __m256 maskX = _mm256_or_ps(
                        _mm256_cmp_ps(px, vMaxBound, _CMP_GT_OQ),  // Greater Than
                        _mm256_cmp_ps(px, vMinBound, _CMP_LT_OQ)   // Less Than
                    );

                    __m256 maskY = _mm256_or_ps(
                        _mm256_cmp_ps(py, vMaxBound, _CMP_GT_OQ),  // Greater Than
                        _mm256_cmp_ps(py, vMinBound, _CMP_LT_OQ)   // Less Than
                    );

                    __m256 maskZ = _mm256_or_ps(
                        _mm256_cmp_ps(pz, vMaxBound, _CMP_GT_OQ), 
                        _mm256_cmp_ps(pz, vMinBound, _CMP_LT_OQ)
                    );

                    // B. Hard Clamp Positions inside the box (Zero 'if' statements, pure hardware min/max)
                    px = _mm256_max_ps(vMinBound, _mm256_min_ps(px, vMaxBound));
                    py = _mm256_max_ps(vMinBound, _mm256_min_ps(py, vMaxBound));
                    pz = _mm256_max_ps(vMinBound, _mm256_min_ps(pz, vMaxBound));

                    // C. Calculate the bounced velocities for EVERY particle
                    __m256 bouncedVx = _mm256_mul_ps(vx, vBounceDamp);
                    __m256 bouncedVy = _mm256_mul_ps(vy, vBounceDamp);
                    __m256 bouncedVz = _mm256_mul_ps(vz, vBounceDamp);

                    // D. Blend the velocities: If the mask is true, apply 'bouncedVx', otherwise keep original 'vx'
                    vx = _mm256_blendv_ps(vx, bouncedVx, maskX);
                    vy = _mm256_blendv_ps(vy, bouncedVy, maskY);
                    vz = _mm256_blendv_ps(vz, bouncedVz, maskZ);

                    // 8. STORE: Save results of the new positions and velocities back into RAM.
                    _mm256_store_ps(r_pX + i, px);
                    _mm256_store_ps(r_pY + i, py);
                    _mm256_store_ps(r_pZ + i, pz);

                    _mm256_store_ps(r_vX + i, vx);
                    _mm256_store_ps(r_vY + i, vy);
                    _mm256_store_ps(r_vZ + i, vz);
                }
            }

            // Calculate ops for THIS chunk only, not the global activeCount
            uint64_t localOps = (uint64_t)(end - start) * 420ULL; 
            
            // Safely write to this specific thread's isolated cache line
            g_JobSystem.threadStats[tl_workerIndex]->totalFlops.fetch_add(localOps, std::memory_order_relaxed);
        });

        // Clean the Main Thread before it returns to OpenGL/ImGui!
        _mm256_zeroupper();

        // --- TELEMETRY ADJUSTMENT ---
        // Spatial Hash Math Estimate: ~45 FLOPs for the Newton-Raphson/Position Update. Newton-Raphson: ~45 FLOPs (floating-point operations) per particle per frame, excludes collision.
        // uint64_t opsThisFrame = (uint64_t)activeCount * 45ULL;

        // + ~15 FLOPs per neighbor checked in the 9 cells.
        // Assuming average density of ~25 neighbors checked per particle: 45 + (15 * 25) = ~420 FLOPs, includes collision.
        // uint64_t opsThisFrame = (uint64_t)activeCount * 420ULL; 

        // g_TotalMathOperations.fetch_add(opsThisFrame, std::memory_order_relaxed);
    }
};
