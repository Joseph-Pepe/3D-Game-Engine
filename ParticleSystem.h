#pragma once
#include <span>
#include <algorithm> // For std::fill
#include <vector>

#include "EngineSettings.h"
#include "SIMD/SIMDCustomWrapper.h" 
#include "SIMD/SIMDVectorMath.h"
#include "Memory.h"  // For your Arena allocator
#include "MortonCode.h"
#include "Math.h"

// =============================================================
// ARRAY OF POINTERS (DESTROYS PERFORMANCE)
// =============================================================
/*
    - An array of pointers pointing to random locations is bad for the CPU hardware prefetcher b/c it has to fetch random pages from RAM.
    - Its mathematically separated by offsets.
    - All threads will write into the same massive block of memory.

      // Creates a list of pointers pointing to separate, fragmented allocations (i.e., avoid at all costs).
      std::vector<std::vector<float>> pX, pY, pZ;
*/

// =============================================================
// PARTICLE SYSTEM & PHYSICS ENGINE MODULE (AoSoA)
// =============================================================
namespace Engine::Physics {

    // A pure lightweight data container that stores data and manages memory arenas.
    struct ParticleSystem {

        // SOA: Manages 6 separate variables and manually loading/storing them lane by lane.
        // C++20 std::span are lightweight, zero cost abstractions that act exactly like arrays, but does not try to free() its memory when it goes out of scope.
        // std::span<float> pX, pY, pZ;  // CPU needs to fetch memory from 3 different locations to process one physics step.
        // std::span<float> vX, vY, vZ; 

        // AoSoA: Groups X, Y, Z coordinates into L1-cache aligned chunks perfectly aligned to the exact hardware registers (e.g., 32-bytes for AVX2, 64-bytes for AVX-512)! It stores 8 to 16 particles per index.
        Engine::ISAArch::NativeAlignedVector<SIMDVector3D> positions;   // Component 1: Positions  
        Engine::ISAArch::NativeAlignedVector<SIMDVector3D> velocities;  // Component 2: Velocities

        // SPATIAL HASH GRID LOOKUP MAPS
        std::vector<uint32_t> particleCellIndices; // Maps a particle index to its 3D cell Z-order hash
        std::vector<uint32_t> cellStartOffset;     // The lookup table for cell memory boundaries

        // ========================================================
        // RADIX SORT & SPATIAL GRID BUILDER
        // ========================================================

        // Persistent Memory Buffers for Sorting (Zero Per-Frame Allocations)
        struct ParticleHash {
            uint32_t hash;
            uint32_t originalIndex;
        };

        std::vector<ParticleHash> hashes;
        std::vector<ParticleHash> hashesTemp; // For Radix Sort swap
        Engine::ISAArch::NativeAlignedVector<SIMDVector3D> tempPos;
        Engine::ISAArch::NativeAlignedVector<SIMDVector3D> tempVel;

        size_t activeCount = 0;

        void Initialize(size_t maxParticles) {
            // Make alignment padding to restrict the step based on the silicons SIMD float width (AVX2: 8-float widths (8uz), AVX512: 16-float widths (16uz)).
            constexpr size_t stride = Engine::Physics::NativeFloatSIMDBatch::size();

            // Safely pad to the nearest multiple of whatever stride the architecture uses
            size_t paddedCount = (maxParticles + (stride - 1uz)) & ~(stride - 1uz);   // C++23: 'uz' is the size_t literal (e.g., AVX2: (maxParticles + 7uz) & ~7uz)
            size_t batchCount = paddedCount / stride; 

            // Initialize the grid arrays
            particleCellIndices.resize(paddedCount, 0);
            
            // 2^18 = 262,144 cells for a 501x501 grid (18 bits) Or 0x3FFFF + 2 for the maximum hash limit
            cellStartOffset.resize(0x3FFFF + 2, 0);

            // The std::vector handles the strict memory alignment via AlignedAllocator automatically.
            positions.resize(batchCount);
            velocities.resize(batchCount);

            // Pre-allocate the persistent buffers
            hashes.resize(paddedCount);
            hashesTemp.resize(paddedCount);
            tempPos.resize(batchCount);
            tempVel.resize(batchCount);

            // Zero-initialize variables vertically down the lanes to prevent memory bus contention
            for (size_t i = 0; i < batchCount; ++i) {
                positions[i].x = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                positions[i].y = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                positions[i].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);

                velocities[i].x = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                velocities[i].y = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                velocities[i].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);
            }
            
            // --- ALLOCATE MEMORY DIRECTLY FROM THE CUSTOM ARENA ---
            // Allocate with the maximum safe 64-byte alignment boundaries (i.e., AVX-512 requires 16-float padding (64 bytes)).
            // pX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            // pY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            // pZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);

            // vX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            // vY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            // vZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);

            // ==========================================
            // ZERO-INITIALIZE ARENA MEMORY
            // ==========================================
            
            // Zero-initialize memory to prevent NaN explosions in physics math.
            // std::fill(pX.begin(), pX.end(), 0.0f);
            // std::fill(pY.begin(), pY.end(), 0.0f);
            // std::fill(pZ.begin(), pZ.end(), 0.0f);
            // std::fill(vX.begin(), vX.end(), 0.0f);
            // std::fill(vY.begin(), vY.end(), 0.0f);
            // std::fill(vZ.begin(), vZ.end(), 0.0f);
        }

        // --- THE SPAWNER (Game Logic)---
        // This only runs once during initialization or user clicks.
        void spawnParticles(size_t startIdx, size_t endIdx, float speed) {
            uint32_t spawnCount = endIdx - startIdx;
            if (spawnCount <= 0) return;

            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t CHUNK_SIZE = std::max(1024u, static_cast<uint32_t>(spawnCount / (threadCount * 4)));
            CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7; 

            g_JobSystem.DispatchAndWait(spawnCount, CHUNK_SIZE, [&](uint32_t localStart, uint32_t localEnd) {
                constexpr size_t stride = Engine::Physics::NativeFloatSIMDBatch::size();
                float deltaTheta = 0.001f;
                float radiusBase = 200.0f;

                // Fast, stateless hash function for lock-free parallel noise
                auto fastHash = [](uint32_t index) -> float {
                    uint32_t state = index * 747796405u + 2891336453u;
                    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                    state = (state >> 22u) ^ state;
                    return static_cast<float>(state) * 2.3283064365386963e-10f;
                };

                // Correctly loop over batches using the native hardware stride step
                for(size_t i = localStart; i < localEnd; i += stride) {
                    size_t currentBatchIdx = (startIdx + i) / stride;
                    
                    // 1. Generate baseline sequential indices via iota
                    Engine::ISAArch::WideFloat indices = Engine::ISAArch::WideFloat::iota(static_cast<float>(startIdx + i), 1.0f);
                    
                    // 2. Calculate the exact angles for all particles simultaneously
                    Engine::ISAArch::WideFloat angles = indices * deltaTheta;

                    // 3. Vectorized Minimax Trigonometry (Calculates full batch sin/cos concurrently)
                    Engine::ISAArch::WideFloat batchSin = sin(angles); 
                    Engine::ISAArch::WideFloat batchCos = cos(angles);

                    // 4. Generate Random Offsets safely via scalar fallback loops
                    alignas(64) float rArray[stride];
                    for (size_t k = 0; k < stride; ++k) {
                        rArray[k] = radiusBase + (fastHash(static_cast<uint32_t>(startIdx + i + k)) * 600.0f);
                    }
                    Engine::ISAArch::WideFloat r(rArray); 

                    // 5. Commit Positions directly to references without copies
                    positions[currentBatchIdx].x = batchCos * r;
                    positions[currentBatchIdx].y = batchSin * r;
                    positions[currentBatchIdx].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);

                    // 6. Commit Tangential Velocities
                    velocities[currentBatchIdx].x = -batchSin * speed;
                    velocities[currentBatchIdx].y =  batchCos * speed;
                    velocities[currentBatchIdx].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                }
            });
        }

        // --- DIMENSIONAL TRANSITION (Game Logic) ---
        // Flattens the simulation to 2D or explodes it into a 3D disc safely in parallel.
        void transitionDimensions(bool to2D) {
            if (activeCount == 0) return;

            constexpr size_t stride = Engine::Physics::NativeFloatSIMDBatch::size();
            size_t activeBatches = (activeCount + stride - 1uz) / stride;

            g_JobSystem.DispatchAndWait(activeBatches, 256, [&](uint32_t start, uint32_t end) {
                if (to2D) {
                    for (uint32_t i = start; i < end; ++i) {
                        // Fast path: Squash everything flat and kill vertical momentum instantly
                        positions[i].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                        velocities[i].z = Engine::Physics::NativeFloatSIMDBatch(0.0f);
                    }
                } else {
                    auto fastHash = [](uint32_t index) -> float {
                        uint32_t state = index * 747796405u + 2891336453u;
                        state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                        state = (state >> 22u) ^ state;
                        return static_cast<float>(state) * 2.3283064365386963e-10f;
                    };

                    alignas(64) float randZ[stride];
                    alignas(64) float randVZ[stride];

                    // Explode the particles into a 3D disc/cloud safely in parallel
                    for (uint32_t i = start; i < end; ++i) {
                        uint32_t baseGlobalIdx = static_cast<uint32_t>(i * stride);
                        
                        for (size_t k = 0; k < stride; ++k) {
                            // Map the 0.0 to 1.0 range back to your desired physical bounds
                            uint32_t idx = baseGlobalIdx + static_cast<uint32_t>(k);
                            randZ[k]  = (fastHash(idx) * 100.0f) - 50.0f; // -50 to 50 
                            randVZ[k] = (fastHash(idx + static_cast<uint32_t>(activeCount)) * 4.0f) - 2.0f; // Offset the seed for a different result (-2 to 2)
                        }

                        positions[i].z = Engine::Physics::NativeFloatSIMDBatch(randZ);
                        velocities[i].z = Engine::Physics::NativeFloatSIMDBatch(randVZ);
                    }
                }
            });
        }

        // Radix Sort is O(N) instead of O(N log N) like std::sort.
        // It sorts the hashes in 4 passes (8 bits at a time for a 32-bit integer).
        void RadixSort(std::vector<ParticleHash>& array, std::vector<ParticleHash>& tempArray, size_t count) {
            const uint32_t BITS_PER_PASS = 8;
            const uint32_t NUM_PASSES = 32 / BITS_PER_PASS;
            const uint32_t BUCKET_COUNT = 1 << BITS_PER_PASS;
            const uint32_t MASK = BUCKET_COUNT - 1;

            std::vector<ParticleHash>* src = &array;
            std::vector<ParticleHash>* dst = &tempArray;

            for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
                uint32_t shift = pass * BITS_PER_PASS;
                uint32_t counts[BUCKET_COUNT] = {0};

                for (size_t i = 0; i < count; ++i) {
                    uint32_t bucket = ((*src)[i].hash >> shift) & MASK;
                    counts[bucket]++;
                }

                // 2. Prefix sum to find offsets
                uint32_t offsets[BUCKET_COUNT];
                offsets[0] = 0;
                for (uint32_t i = 1; i < BUCKET_COUNT; ++i) {
                    offsets[i] = offsets[i - 1] + counts[i - 1];
                }

                for (size_t i = 0; i < count; ++i) {
                    uint32_t bucket = ((*src)[i].hash >> shift) & MASK;
                    (*dst)[offsets[bucket]++] = (*src)[i];
                }

                // Swap pointers for the next pass
                std::swap(src, dst);
            }

            // If we ended on the temp array, copy it back to the original
            if (src == &tempArray) {
                std::copy(tempArray.begin(), tempArray.begin() + count, array.begin());
            }
        }

        void UpdateSpatialGrid(bool is2D, bool isLegacyCPU) {
            if (activeCount == 0) return;

            constexpr size_t stride = Engine::ISAArch::NativeFloatSIMD::size();
            size_t activeBatches = (activeCount + stride - 1uz) / stride;

            size_t paddedCount = activeBatches * stride;

            // 1. HASH PHASE (Using cached 'hashes' array)
            // Note: We have to unpack the AoSoA slightly to get individual coordinates for the hash.
            for (size_t i = 0; i < activeBatches; ++i) {
                alignas(64) float pX[stride];
                alignas(64) float pY[stride];
                alignas(64) float pZ[stride];
                
                positions[i].x.copy_to(pX);
                positions[i].y.copy_to(pY);
                positions[i].z.copy_to(pZ);

                for (size_t k = 0; k < stride; ++k) {
                    size_t globalIdx = (i * stride) + k;
                    uint32_t hash = 0;

                    // RECTIFIED: Branchless boundary sentinel alignment tracks full vector arrays
                    if (globalIdx >= activeCount) {
                        hash = 0xFFFFFFFF; // Forces unallocated slots into the trailing padding block
                    } else {
                        int gx = static_cast<int>((pX[k] + 1000.0f) * 0.25f);
                        int gy = static_cast<int>((pY[k] + 1000.0f) * 0.25f);
                        int gz = static_cast<int>((pZ[k] + 1000.0f) * 0.25f);

                        if (is2D) {
                            hash = getMortonCode<false>(gx, gy, 0) & 0x3FFFF; 
                        } else {
                            hash = getMortonCode<false>(gx, gy, gz) & 0x3FFFF;
                        }
                    }

                    hashes[globalIdx] = {hash, static_cast<uint32_t>(globalIdx)};
                }
            }

            // 2. SORT PHASE: Radix Sort the hashes
            RadixSort(hashes, hashesTemp, paddedCount);

            // 3. LOOKUP OFFSET TABLE REBUILD PHASE: Physically reorganize the AoSoA memory!
            // Instead of allocating massive new vectors and swapping them, allocate a fast temporary buffer, sort into it, and copy directly back!
            std::fill(cellStartOffset.begin(), cellStartOffset.end(), 0);

            uint32_t previousHash = 0xFFFFFFFF; // Invalid initial hash

            for (size_t i = 0; i < paddedCount; i += stride) {
                uint32_t currentHash = hashes[i].hash;

                // If we hit the padding block (which was sorted to the end), zero the output
                if (currentHash == 0xFFFFFFFF) continue;

                if (currentHash != previousHash) {
                    cellStartOffset[currentHash] = static_cast<uint32_t>(i);
                    
                    if (previousHash != 0xFFFFFFFF) {
                        for (uint32_t h = previousHash + 1; h < currentHash; ++h) {
                            cellStartOffset[h] = static_cast<uint32_t>(i);
                        }
                    }
                    previousHash = currentHash;
                }
            }

            if (previousHash != 0xFFFFFFFF) {
                for (uint32_t h = previousHash + 1; h < cellStartOffset.size(); ++h) {
                    cellStartOffset[h] = static_cast<uint32_t>(activeCount);
                }
            }
        }
    };

    // --- THE ACTUAL HARDWARE COLLISION KERNEL ---
    template <bool Is2D, bool IsLegacy>
    FORCE_INLINE void SolveCollisionsInternal(std::span<SIMDVector3D> pos/*float* xs, float* ys, float* zs*/, 
                                              std::span<SIMDVector3D> vel/*float* vx, float* vy, float* vz*/, 
                                              const ParticleSystem::ParticleHash* sortedHashes,
                                              size_t count, size_t startBatch, size_t endBatch) {
        
        using WideFloat = Engine::ISAArch::NativeFloatSIMD;
        using WideMask = typename WideFloat::mask_type;
        constexpr size_t stride = WideFloat::size();

        const float CELL_SIZE = 4.0f;
        const float RADIUS_SQ = CELL_SIZE * CELL_SIZE;
        const float REPULSION = 5.0f; 

        WideFloat vRadiusSq(RADIUS_SQ);
        WideFloat vRepulsion(REPULSION);
        WideFloat vEpsilon(0.0001f);
        
        // Newton-Raphson Constants
        WideFloat half(0.5f);
        WideFloat three_halves(1.5f);

        // Calculate batches
        size_t batchCount = (count + stride - 1uz) / stride;

        // ===========================================
        // COLLISION DETECTION (SLIDING WINDOWS)
        // ===========================================
        /*
            - Particles that are physically touching eachother in 3D space are guaranteed to be located within a few indices of eachother in the positions array.
            - Compares the current SIMD batch against its immediate adjacent memory blocks in the positions array.
            - Scans 64 indices in front of and behind the current batch to read linearly in memory and reduce computational load.
            - Because the array is Radix Sorted via Morton Z-Order curve, particles near each other physically are near each other in RAM.
            - We scan a window around the current batch.
            - A wider window catches more high-velocity collisions but costs more CPU cycles.
        */

        // How far forward/backward in the ARRAY to look for collisions.
        // Window scaled safely down to prevent O(N^2) load density saturation stalls
        const size_t SEARCH_WINDOW = 32;

        // NOTE THE STRIDE! We load 8/16 particles at once.
        // STRICTLY OBEYS THE THREAD CHUNK
        // Hot Path Loop: Maps localized indices branchlessly across execution pipelines
        for (size_t i = startBatch; i < endBatch; ++i) {

            // MASKING: Prevent gravity/repulsion from applying to trailing padding lanes
            WideFloat myIndices = Engine::ISAArch::WideFloat::iota(static_cast<float>(i * stride), 1.0f);
            WideMask activeMask = myIndices < WideFloat(static_cast<float>(count));

            WideFloat p1_x = pos[i].x;  
            WideFloat p1_y = pos[i].y;  
            WideFloat p1_z = pos[i].z;

            WideFloat accX(0.0f);
            WideFloat accY(0.0f);
            WideFloat accZ(0.0f);

            // UNCLAMPED: Threads can safely read cross-chunk memory without "invisible walls"
            size_t startSearch = (i > SEARCH_WINDOW) ? i - SEARCH_WINDOW : 0;
            size_t endSearch = std::min(batchCount, i + SEARCH_WINDOW + 1);

            // Scan the adjacent memory blocks!
            for (size_t j = startSearch; j < endSearch; ++j) {

                // PERFORMANCE FIX: Direct pointer casting bypasses slow operator[] wrapper spills
                const float* p2_x_ptr = reinterpret_cast<const float*>(&pos[j].x);
                const float* p2_y_ptr = reinterpret_cast<const float*>(&pos[j].y);
                const float* p2_z_ptr = reinterpret_cast<const float*>(&pos[j].z);

                for (size_t lane = 0; lane < stride; ++lane) {

                    // SAFETY: Prevent ghost collisions with 0.0f padded lanes at the end of the array
                    if ((j * stride + lane) >= count) continue; 

                    WideFloat neighborIdx(static_cast<float>(j * stride + lane));
                    WideMask selfMask = (myIndices > neighborIdx) || (myIndices < neighborIdx);

                    // Lightning-fast register broadcast
                    WideFloat p2_x(p2_x_ptr[lane]);  
                    WideFloat p2_y(p2_y_ptr[lane]);  

                    WideFloat diffX = p1_x - p2_x;
                    WideFloat diffY = p1_y - p2_y;
                    WideFloat distSq;

                    if constexpr (Is2D) {
                        distSq = (diffX * diffX) + (diffY * diffY);
                    } else {
                        WideFloat p2_z(p2_z_ptr[lane]);
                        WideFloat diffZ = p1_z - p2_z;
                        distSq = (diffX * diffX) + (diffY * diffY) + (diffZ * diffZ);
                    }

                    // Strict evaluation mask
                    WideMask mask = (distSq < vRadiusSq) && selfMask && activeMask;

                    if (any_of(mask)) {
                        // 2. BREAK PERFECT OVERLAP SYMMETRY
                        // If two particles spawn on the exact same coordinate, distSq is 0. 
                        // We must nudge them slightly so they have a valid direction to repel.
                        auto perfectOverlapMask = mask && (distSq < vEpsilon);
                        Engine::ISAArch::where(perfectOverlapMask, diffX) = 1e-4f;
                        Engine::ISAArch::where(perfectOverlapMask, distSq) = 1e-8f; 

                        // 3. PREVENT NaN INFECTION!
                        // We MUST patch the inactive/empty lanes with a safe non-zero number before the math!
                        // Otherwise rsqrt(0) -> Infinity -> NaN -> entire register is destroyed.
                        Engine::ISAArch::where(!mask, distSq) = vRadiusSq; 

                        distSq = max(distSq, vEpsilon); // Hard clamp prevents Infinity

                        // 4. Execute the math safely across all lanes
                        WideFloat rsqrt_approx = rsqrt(distSq);

                        // Newton-Raphson refinement (optional, but good for precision)
                        WideFloat term = fmadd(-(distSq * half), (rsqrt_approx * rsqrt_approx), three_halves);
                        WideFloat refined_rsqrt = rsqrt_approx * term;

                        // Calculate the actual distance
                        WideFloat distance = distSq * refined_rsqrt; 

                        // Calculate how deeply they are intersecting (e.g., Radius - Distance)
                        WideFloat overlap = CELL_SIZE - distance;
                        
                        // Explicit fallback clamping guarantees zero force for non-colliding lanes
                        WideFloat zero(0.0f);
                        overlap = WideFloat::choose(mask, overlap, zero);
                        overlap = WideFloat::choose(overlap > zero, overlap, zero);

                        // Multiply overlap by the repulsion constant to get the true Force Magnitude
                        WideFloat forceMagnitude = overlap * vRepulsion;

                        // Calculate the Normalized Unit Vector components
                        WideFloat unitX = diffX * refined_rsqrt;
                        WideFloat unitY = diffY * refined_rsqrt;

                        // ACCUMULATE VECTOR FORCES (Normalized Direction * Force Magnitude)
                        accX += (unitX * forceMagnitude);
                        accY += (unitY * forceMagnitude);

                        if constexpr (!Is2D) {
                            WideFloat p2_z(p2_z_ptr[lane]); 
                            WideFloat diffZ = p1_z - p2_z;
                            WideFloat unitZ = diffZ * refined_rsqrt;
                            accZ += (unitZ * forceMagnitude);
                        }
                    }
                }
            }

            // Write back seamlessly
            WideFloat v1_x = vel[i].x;
            WideFloat v1_y = vel[i].y;
            WideFloat v1_z = vel[i].z;

            v1_x += accX;
            v1_y += accY;
            v1_z += accZ;

            vel[i].x = v1_x;
            vel[i].y = v1_y;
            vel[i].z = v1_z;
        }
    }

    // --- THE HARDWARE COLLISION TEMPLATE (SOA PIPELINE) ---
    // NOTE: For batches we don't need to use is2DMode, because all we need to do is not pass in the z coordinates for batch processing.
    // template <typename Abi>
    void SolveCollisions(std::span<SIMDVector3D> pos/*float* xs, float* ys, float* zs*/, 
                                 std::span<SIMDVector3D> vel/*float* vx, float* vy, float* vz*/, 
                                 const ParticleSystem::ParticleHash* sortedHashes,
                                 size_t count, size_t startBatch, size_t endBatch) {

        // Cache the global setting locally so the compiler can aggressively optimize
        const bool isLegacy = g_EngineSettings.isLegacyCPU;

        // Hoist the runtime check to create pure compiled paths
        if (g_EngineSettings.is2DMode) {
            if (isLegacy) SolveCollisionsInternal<true, true>(pos /*xs, ys, zs*/, vel /*vx, vy, vz*/, sortedHashes, count, startBatch, endBatch);
            else          SolveCollisionsInternal<true, false>(pos /*xs, ys, zs*/, vel /*vx, vy, vz*/, sortedHashes, count, startBatch, endBatch);
        } else {
            if (isLegacy) SolveCollisionsInternal<false, true>(pos /*xs, ys, zs*/, vel /*vx, vy, vz*/, sortedHashes, count, startBatch, endBatch);
            else          SolveCollisionsInternal<false, false>(pos /*xs, ys, zs*/, vel /*vx, vy, vz*/, sortedHashes, count, startBatch, endBatch);
        }
    }

    // --- THE HARDWARE INTEGRATION KERNEL ---
    // Fully branchless, parameterized by Abi, compiled uniquely for AVX-512, AVX2, and NEON
    // template <typename Abi>
    void IntegrateParticles(std::span<SIMDVector3D> pos/*float* xs, float* ys, float* zs*/, 
                                    std::span<SIMDVector3D> vel/*float* vx, float* vy, float* vz*/, 
                                    size_t count, float deltaTime, float gravityVal,
                                    float mouseX, float mouseY, bool isMouseDown, size_t startBatch, size_t endBatch) {
        
        // Dynamically instantiate the exact SIMD width for this ABI
        using WideFloat = Engine::ISAArch::NativeFloatSIMD;
        using WideMask = typename WideFloat::mask_type;
        constexpr size_t stride = WideFloat::size();

        // Broadcast scalar to vector natively
        WideFloat dt(deltaTime);
        WideFloat gravity(gravityVal);
        WideFloat damping(1.0f);
        WideFloat epsilon(0.001f);

        // Bounding Wall Constants
        WideFloat maxBound(950.0f);
        WideFloat minBound(-950.0f);
        WideFloat bounceDamp(-0.8f);

        // Mouse Constants
        WideFloat mx(mouseX);
        WideFloat my(mouseY);
        WideFloat mouseRadiusSq(150.0f * 150.0f);
        WideFloat mouseForce(500.0f);

        // Process particles in hardware-specific batches (4, 8, or 16 at a time) and obey the thread chunk.
        for (size_t i = startBatch; i < endBatch; ++i) {

            // Generate the global index mask to prevent processing padded dead-lanes!
            WideFloat myIndices = Engine::ISAArch::WideFloat::iota(static_cast<float>(i * stride), 1.0f);
            WideMask activeMask = myIndices < WideFloat(static_cast<float>(count));

            // 1. LOAD BY VALUE (Pulls data from RAM into ultra-fast CPU Registers)
            WideFloat pX = pos[i].x; 
            WideFloat pY = pos[i].y; 
            WideFloat pZ = pos[i].z; 
            
            WideFloat vX = vel[i].x; 
            WideFloat vY = vel[i].y; 
            WideFloat vZ = vel[i].z;

            // 1. Map modern types over class variables securely via references
            // WideFloat& pX = pos[i].x; // WideFloat pX(xs + i);
            // WideFloat& pY = pos[i].y; // WideFloat pY(ys + i);
            // WideFloat& pZ = pos[i].z; // WideFloat pZ(zs + i); // If 2D, just pass a dummy zero array for Z!
            
            // WideFloat& vX = vel[i].x; // WideFloat vX(vx + i);
            // WideFloat& vY = vel[i].y; // WideFloat vY(vy + i);
            // WideFloat& vZ = vel[i].z; // WideFloat vZ(vz + i);

            // --- Math: Distance to Center (Gravity Pull) ---

            // 1. Calculate Distance Squared via the struct method
            WideFloat distSq = pos[i].length_sq();  // WideFloat distSq = (pX * pX) + (pY * pY) + (pZ * pZ);

            // 2. Hardware Inverse Square Root
            WideFloat invDist = rsqrt(distSq + epsilon);  // Fast hardware approximation (Abstracts _mm256_rsqrt_ps internally)
            WideFloat pull = invDist * gravity;
            Engine::ISAArch::where(!activeMask, pull) = 0.0f; // Silence padding

            // ========================================================
            // 3. MOUSE REPULSION
            // ========================================================
            if (isMouseDown) {
                WideFloat diffMouseX = pX - mx;
                WideFloat diffMouseY = pY - my;

                // Calculate distance squared to the mouse
                WideFloat mouseDistSq = (diffMouseX * diffMouseX) + (diffMouseY * diffMouseY);
                
                // Mask evaluates to true if particle is within the 150-unit radius
                WideMask mouseMask = mouseDistSq < mouseRadiusSq;

                // CULL: If none of the particles in this batch are near the mouse, skip the math! We need to see if any bits in the mask are flipped.
                if (any_of(mouseMask)) {
                    
                    WideFloat invMouseDist = rsqrt(mouseDistSq + epsilon);
                    WideFloat mousePush = invMouseDist * mouseForce;
                    
                    // C++26: Use your 'where' proxy to instantly zero out non-colliding lanes!
                    Engine::ISAArch::where(!mouseMask, mousePush) = 0.0f; 

                    // Old Way (Direct Intrinsic Blend) Zero out the push force for particles in the batch that are NOT near the mouse
                    // WideFloat zero(0.0f);
                    // mousePush = Engine::ISAArch::blend(mouseMask.native_handle(), mousePush.native_handle(), zero.native_handle()); 
                    // mousePush = WideFloat::choose(mouseMask, mousePush, zero);

                    // Add the mouse force to the existing velocity
                    vX += (diffMouseX * mousePush);  // vel[i].x += (diffMouseX * mousePush);   // vX = vX + (diffMouseX * mousePush);
                    vY += (diffMouseY * mousePush);  // vel[i].y += (diffMouseY * mousePush);   // vY = vY + (diffMouseY * mousePush);
                }
            }
            
            // 3. Apply the central gravity pull 
            vX -= (pX * pull);  // vel[i].x -= pos[i].x * pull;  // vX = vX - (pX * pull);
            vY -= (pY * pull);  // vel[i].y -= pos[i].y * pull;  // vY = vY - (pY * pull);
            vZ -= (pZ * pull);  // vel[i].z -= pos[i].z * pull;  // vZ = vZ - (pZ * pull);

            // 4. Update Position & Damping
            vX *= damping;
            vY *= damping;
            vZ *= damping;
            
            // 4. Update Position & Damping via references
            // vel[i].mul(damping); // vel[i].mul(WideFloat(0.99f)); // Damping: vX *= damping; vY *= damping; vZ *= damping;
            
            pX += vX * dt;  // pos[i].x += vel[i].x * dt; // pX += vX * dt;
            pY += vY * dt;  // pos[i].y += vel[i].y * dt; // pY += vY * dt;
            pZ += vZ * dt;  // pos[i].z += vel[i].z * dt; // pZ += vZ * dt;

            // ========================================================
            // 5. BRANCHLESS BOUNDING WALLS
            // ========================================================
            
            // Mask evaluates to true if particle is outside bounds
            WideMask maskX = (pX > maxBound) || (pX < minBound);
            WideMask maskY = (pY > maxBound) || (pY < minBound);
            WideMask maskZ = (pZ > maxBound) || (pZ < minBound);

            // Pure ADL multi-lane horizontal bounds clamp positions back inside the box
            pX = min(maxBound, max(minBound, pX));
            pY = min(maxBound, max(minBound, pY));
            pZ = min(maxBound, max(minBound, pZ));

            // Calculate bounced velocities
            WideFloat bouncedVx = vX * bounceDamp;
            WideFloat bouncedVy = vY * bounceDamp;
            WideFloat bouncedVz = vZ * bounceDamp;

            // Use your 'where' proxy to conditionally assign the bounced velocities! Modify data directly inside an array.
            Engine::ISAArch::where(maskX, vX) = bouncedVx;   // vX = Engine::ISAArch::blend(maskX, bouncedVx, vX); // Blend velocities: If mask is true, use bounced velocity. Else, keep original. Direct register blending (zero-overhead)!
            Engine::ISAArch::where(maskY, vY) = bouncedVy;   // vY = Engine::ISAArch::blend(maskY, bouncedVy, vY);
            Engine::ISAArch::where(maskZ, vZ) = bouncedVz;   // vZ = Engine::ISAArch::blend(maskZ, bouncedVz, vZ);

            // 6. STORE BY VALUE (Flushes the registers back to RAM)
            pos[i].x = pX;
            pos[i].y = pY;
            pos[i].z = pZ;

            vel[i].x = vX;
            vel[i].y = vY;
            vel[i].z = vZ;

            // // 6. Store Data
            // pX.copy_to(xs + i);
            // pY.copy_to(ys + i);
            // pZ.copy_to(zs + i);

            // vX.copy_to(vx + i);
            // vY.copy_to(vy + i);
            // vZ.copy_to(vz + i);
        }
    }

    // --- DEDICATED PHYSICS BENCHMARK KERNEL ---
    // Pure ALU stress test. Executes the gravity math 'repeats' times inside the CPU L1 cache with zero memory barriers or OS/JobSystem interruptions.
    // template <typename Abi>
    void BenchmarkParticles(std::span<SIMDVector3D> pos /*float* xs, float* ys, float* zs*/, 
                                    std::span<SIMDVector3D> vel /*float* vx, float* vy, float* vz*/, 
                                    size_t count, float deltaTime, float gravityVal, int64_t repeats, size_t startBatch, size_t endBatch) {
        
        using WideFloat = Engine::ISAArch::NativeFloatSIMD;
        constexpr size_t stride = WideFloat::size();

        WideFloat dt(deltaTime);
        WideFloat gravity(gravityVal);
        WideFloat damping(1.0f);
        WideFloat epsilon(0.001f);

        // The repeats happen INSIDE the kernel to bypass function call overhead
        for (int64_t r = 0; r < repeats; ++r) {
            for (size_t i = startBatch; i < endBatch; i += 1) {

                // 1. LOAD BY VALUE
                WideFloat pX = pos[i].x; 
                WideFloat pY = pos[i].y; 
                WideFloat pZ = pos[i].z; 
                
                WideFloat vX = vel[i].x; 
                WideFloat vY = vel[i].y; 
                WideFloat vZ = vel[i].z;
                
                // 1. Load Data
                // WideFloat& pX = pos[i].x; // WideFloat pX(xs + i); load from raw scalar arrays.
                // WideFloat& pY = pos[i].y; // WideFloat pY(ys + i);
                // WideFloat& pZ = pos[i].z; // WideFloat pZ(zs + i);
                
                // WideFloat& vX = vel[i].x; // WideFloat vX(vx + i);
                // WideFloat& vY = vel[i].y; // WideFloat vY(vy + i);
                // WideFloat& vZ = vel[i].z; // WideFloat vZ(vz + i);
                
                // 2. Math: Distance to Center
                WideFloat distSq = pos[i].length_sq();
                WideFloat invDist = rsqrt(distSq + epsilon); 
                WideFloat pull = invDist * gravity;

                // 3. Apply Pull
                vX -= pX * pull; // vX = vX - (pX * pull);
                vY -= pY * pull; // vY = vY - (pY * pull);
                vZ -= pZ * pull; // vZ = vZ - (pZ * pull);
                 
                // 4. Update Position & Damping
                vX *= damping;
                vY *= damping;
                vZ *= damping;

                pX += vX * dt;
                pY += vY * dt;
                pZ += vZ * dt;

                // // 5. Store Data
                // No manual copy_to() needed! We are modifying the span memory directly via references.
                // pX.copy_to(xs + i);
                // pY.copy_to(ys + i);
                // pZ.copy_to(zs + i);

                // vX.copy_to(vx + i);
                // vY.copy_to(vy + i);
                // vZ.copy_to(vz + i);

                // 5. STORE BY VALUE
                pos[i].x = pX;
                pos[i].y = pY;
                pos[i].z = pZ;

                vel[i].x = vX;
                vel[i].y = vY;
                vel[i].z = vZ;
            }
        }
    }
}
