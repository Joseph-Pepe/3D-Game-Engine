#pragma once
#include <span>
#include <algorithm> // For std::fill

#include "SIMD/SIMDCustomWrapper.h" 
#include "Memory.h"  // For your Arena allocator
#include "MortonCode.h"
#include "JobSystem.h" 
#include "Math.h"

// =============================================================
// CONTIGUOUS FLAT 1D ARRAY
// =============================================================
/*
    - std::vector<std::vector<uint32_t>> is a list of pointers pointing to separate, fragmented allocations.
    - An array of pointers pointing to random locations is bad for the CPU hardware prefetcher b/c it has to fetch random pages from RAM.

    - std::vector<uint32_t> is a single contiguous array.
    - All threads will write into the same massive block of memory.
    - Its mathematically separated by offsets.
*/

// =============================================================
// PARTICLE SYSTEM & PHYSICS ENGINE MODULE
// =============================================================
namespace Engine::Physics {

    // A pure lightweight data container that stores data and manages memory arenas.
    struct ParticleSystem {
        // C++20 std::span are lightweight, zero cost abstractions that act exactly like arrays, but does not try to free() its memory when it goes out of scope.
        std::span<float> pX, pY, pZ; // Component 1: Positions
        std::span<float> vX, vY, vZ; // Component 2: Velocities

        size_t activeCount = 0;

        void Initialize(size_t maxParticles) {
            // Make alignment padding to restrict the step based on the silicons SIMD float width (AVX2: 8-float widths (8uz), AVX512: 16-float widths (16uz)).
            constexpr size_t SimdAlignmentElements = Engine::ISAArch::WideFloat::size();
            size_t paddedCount = (maxParticles + (SimdAlignmentElements - 1uz)) & ~(SimdAlignmentElements - 1uz); // C++23: 'uz' is the size_t literal (e.g., AVX2: (maxParticles + 7uz) & ~7uz)
            
            // --- ALLOCATE MEMORY DIRECTLY FROM THE CUSTOM ARENA ---
            // Allocate with the maximum safe 64-byte alignment boundaries (i.e., AVX-512 requires 16-float padding (64 bytes)).
            pX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            pY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            pZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);

            vX = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            vY = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);
            vZ = std::span<float>(t_PhysicsTransientArena.Allocate<float, 64>(paddedCount), paddedCount);

            // ==========================================
            // ZERO-INITIALIZE ARENA MEMORY
            // ==========================================
            
            // Zero-initialize memory to prevent NaN explosions in physics math.
            std::fill(pX.begin(), pX.end(), 0.0f);
            std::fill(pY.begin(), pY.end(), 0.0f);
            std::fill(pZ.begin(), pZ.end(), 0.0f);
            std::fill(vX.begin(), vX.end(), 0.0f);
            std::fill(vY.begin(), vY.end(), 0.0f);
            std::fill(vZ.begin(), vZ.end(), 0.0f);
        }

        // --- THE SPAWNER (Game Logic)---
        // Lock-free parallel generation using stateless hashing and trigonometry. This only runs once during initialization or user clicks.
        void spawnParticles(size_t startIdx, size_t endIdx, float speed) {
            uint32_t spawnCount = endIdx - startIdx;
            if (spawnCount <= 0) return;

            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t CHUNK_SIZE = std::max(1024u, static_cast<uint32_t>(spawnCount / (threadCount * 4)));
            CHUNK_SIZE = (CHUNK_SIZE + 7) & ~7; 

            g_JobSystem.DispatchAndWait(spawnCount, CHUNK_SIZE, [&](uint32_t localStart, uint32_t localEnd) {
                
                // Fast, stateless hash function for lock-free parallel noise
                auto fastHash = [](uint32_t index) -> float {
                    uint32_t state = index * 747796405u + 2891336453u;
                    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                    state = (state >> 22u) ^ state;
                    return static_cast<float>(state) * 2.3283064365386963e-10f;
                };

                float deltaTheta = 0.001f;
                auto [sin_d, cos_d] = Engine::Math::Functions::FastSinCos(deltaTheta); 

                float startAngle = (float)(startIdx + localStart) * deltaTheta;
                auto [current_sin, current_cos] = Engine::Math::Functions::FastSinCos(startAngle); 

                for(size_t i = localStart; i < localEnd; ++i) {
                    uint32_t globalIdx = startIdx + i;

                    float randomOffset = fastHash(globalIdx) * 600.0f;
                    float r = 200.0f + randomOffset;

                    pX[globalIdx] = current_cos * r;
                    pY[globalIdx] = current_sin * r;
                    pZ[globalIdx] = 0.0f; 

                    vX[globalIdx] = -current_sin * speed; 
                    vY[globalIdx] =  current_cos * speed;
                    vZ[globalIdx] = 0.0f;

                    // Matrix Rotation (Replaces 200 cycle trig functions with 6 cycle multiplies)
                    float next_cos = (current_cos * cos_d) - (current_sin * sin_d);
                    float next_sin = (current_sin * cos_d) + (current_cos * sin_d);
                    
                    current_cos = next_cos;
                    current_sin = next_sin;

                    // Prevent Float Drift
                    if ((i & 255) == 0) {
                        float magSq = current_cos * current_cos + current_sin * current_sin;
                        float invMag = 1.0f / std::sqrt(magSq); // Standard math is fine for initialization
                        current_cos *= invMag;
                        current_sin *= invMag;
                    }
                }
            });
        }

        // --- DIMENSIONAL TRANSITION (Game Logic) ---
        // Flattens the simulation to 2D or explodes it into a 3D disc safely in parallel.
        void transitionDimensions(bool to2D) {
            if (activeCount == 0) return;

            // Basic threading chunk size
            g_JobSystem.DispatchAndWait(activeCount, 2048, [&](uint32_t start, uint32_t end) {
                // OPTIMIZATION: Branch is hoisted OUTSIDE the hot loop!
                if (to2D) {
                    // Fast path: Squash everything flat and kill vertical momentum instantly
                    for (uint32_t i = start; i < end; ++i) {
                        pZ[i] = 0.0f;
                        vZ[i] = 0.0f;
                    }
                } else {
                    // Fast, stateless hash function for lock-free parallel noise
                    auto fastHash = [](uint32_t index) -> float {
                        uint32_t state = index * 747796405u + 2891336453u;
                        state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                        state = (state >> 22u) ^ state;
                        return static_cast<float>(state) * 2.3283064365386963e-10f;
                    };

                    // Explode the particles into a 3D disc/cloud safely in parallel
                    for (uint32_t i = start; i < end; ++i) {
                        float rand1 = fastHash(i);
                        float rand2 = fastHash(i + activeCount); // Offset the seed for a different result

                        // Map the 0.0 to 1.0 range back to your desired physical bounds
                        pZ[i] = (rand1 * 100.0f) - 50.0f;  // -50 to 50 
                        vZ[i] = (rand2 * 4.0f) - 2.0f;     // -2 to 2
                    }
                }
            });
        }
    };

    // --- THE HARDWARE INTEGRATION KERNEL ---
    // Fully branchless, parameterized by Abi, compiled uniquely for AVX-512, AVX2, and NEON
    template <typename Abi>
    void IntegrateParticlesTemplate(float* xs, float* ys, float* zs, 
                                    float* vx, float* vy, float* vz, 
                                    size_t count, float deltaTime, float gravityVal,
                                    float mouseX, float mouseY, bool isMouseDown) {
        
        // Dynamically instantiate the exact SIMD width for this ABI
        using WideFloat = Engine::ISAArch::simd<float, Abi>;
        using WideMask = typename Engine::ISAArch::simd<float, Abi>::mask_type; // For branchless walls
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

        // Process particles in hardware-specific batches (4, 8, or 16 at a time)
        for (size_t i = 0; i < count; i += stride) {
            
            // 1. Load Data using pointer offsets to preserve unaligned load support
            WideFloat pX(xs + i);
            WideFloat pY(ys + i);
            WideFloat pZ(zs + i); // If 2D, just pass a dummy zero array for Z!
            
            WideFloat vX(vx + i);
            WideFloat vY(vy + i);
            WideFloat vZ(vz + i);

            // 2. Math: Distance to Center (Gravity Pull)
            WideFloat distSq = (pX * pX) + (pY * pY) + (pZ * pZ);
            WideFloat invDist = rsqrt(distSq + epsilon); // Fast hardware approximation (Abstracts _mm256_rsqrt_ps internally)
            WideFloat pull = invDist * gravity;

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
                    vX = vX + (diffMouseX * mousePush);
                    vY = vY + (diffMouseY * mousePush);
                }
            }
            
            // Apply the central gravity pull
            vX = vX - (pX * pull);
            vY = vY - (pY * pull);
            vZ = vZ - (pZ * pull);

            // 4. Update Position & Damping
            vX *= damping;
            vY *= damping;
            vZ *= damping;

            pX += vX * dt;
            pY += vY * dt;
            pZ += vZ * dt;

            // ========================================================
            // 5. BRANCHLESS BOUNDING WALLS
            // ========================================================
            
            // Mask evaluates to true if particle is outside bounds
            WideMask maskX = (pX > maxBound) || (pX < minBound);
            WideMask maskY = (pY > maxBound) || (pY < minBound);
            WideMask maskZ = (pZ > maxBound) || (pZ < minBound);

            // Hard clamp the positions back inside the box
            pX = min(maxBound, max(minBound, pX));
            pY = min(maxBound, max(minBound, pY));
            pZ = min(maxBound, max(minBound, pZ));

            // Calculate bounced velocities
            WideFloat bouncedVx = vX * bounceDamp;
            WideFloat bouncedVy = vY * bounceDamp;
            WideFloat bouncedVz = vZ * bounceDamp;

            // Use your 'where' proxy to conditionally assign the bounced velocities
            Engine::ISAArch::where(maskX, vX) = bouncedVx;
            Engine::ISAArch::where(maskY, vY) = bouncedVy;
            Engine::ISAArch::where(maskZ, vZ) = bouncedVz;

            // Blend velocities: If mask is true, use bounced velocity. Else, keep original.
            // vX = Engine::ISAArch::blend(maskX, bouncedVx, vX);
            // vY = Engine::ISAArch::blend(maskY, bouncedVy, vY);
            // vZ = Engine::ISAArch::blend(maskZ, bouncedVz, vZ);

            // 6. Store Data
            pX.copy_to(xs + i);
            pY.copy_to(ys + i);
            pZ.copy_to(zs + i);

            vX.copy_to(vx + i);
            vY.copy_to(vy + i);
            vZ.copy_to(vz + i);
        }
    }

    // --- THE HARDWARE COLLISION TEMPLATE  ---
    // NOTE: For batches we don't need to use is2DMode, because all we need to do is not pass in the z coordinates for batch processing.
    template <typename Abi>
    void SolveCollisionsTemplate(float* xs, float* ys, float* zs, 
                                 float* vx, float* vy, float* vz, 
                                 const uint32_t* cellStartOffsets, 
                                 const uint32_t* sortedIndices,
                                 size_t count) {

        // Cache the global setting locally so the compiler can aggressively optimize
        const bool isLegacy = g_EngineSettings.isLegacyCPU;

        // Hoist the runtime check to create pure compiled paths
        if (g_EngineSettings.is2DMode) {
            if (isLegacy) SolveCollisionsInternal<Abi, true, true>(xs, ys, zs, vx, vy, vz, cellStartOffsets, sortedIndices, count);
            else          SolveCollisionsInternal<Abi, true, false>(xs, ys, zs, vx, vy, vz, cellStartOffsets, sortedIndices, count);
        } else {
            if (isLegacy) SolveCollisionsInternal<Abi, false, true>(xs, ys, zs, vx, vy, vz, cellStartOffsets, sortedIndices, count);
            else          SolveCollisionsInternal<Abi, false, false>(xs, ys, zs, vx, vy, vz, cellStartOffsets, sortedIndices, count);
        }
    }

    // --- THE ACTUAL HARDWARE COLLISION KERNEL ---
    template <typename Abi, bool Is2D, bool IsLegacy>
    FORCE_INLINE void SolveCollisionsInternal(float* xs, float* ys, float* zs, 
                                              float* vx, float* vy, float* vz, 
                                              const uint32_t* cellStartOffsets, 
                                              const uint32_t* sortedIndices,
                                              size_t count) {
        
        using WideFloat = Engine::ISAArch::simd<float, Abi>;
        using WideMask = typename Engine::ISAArch::simd<float, Abi>::mask_type;
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

        // reduces the amount of slots it checks if its 2D.
        int zStart = Is2D ? 0 : -1;
        int zEnd   = Is2D ? 0 : 1;

        // NOTE THE STRIDE! We load 8 particles at once.
        for (size_t i = 0; i < count; i += stride) {
            // Load a batch of 8 continuous particles
            WideFloat p1_x(xs + i);
            WideFloat p1_y(ys + i);
            WideFloat p1_z(zs + i);

            // Accumulators stay as SIMD vectors! No scalar floats needed.
            WideFloat accX(0.0f);
            WideFloat accY(0.0f);
            WideFloat accZ(0.0f);

            // For the broad-phase, we just use the first particle in the batch to find the cell.
            // Since they are sorted, all 8 particles are highly likely to be in the exact same cell.
            int gx = static_cast<int>((xs[i] + 1000.0f) * 0.25f); 
            int gy = static_cast<int>((ys[i] + 1000.0f) * 0.25f);
            int gz = static_cast<int>((zs[i] + 1000.0f) * 0.25f);

            // 2. Loop over the 27 neighboring cells (3x3x3 grid)
            for (int dz = zStart; dz <= zEnd; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {

                        int nx = gx + dx;
                        int ny = gy + dy;
                        int nz = gz + dz;

                        // Ensure we don't look outside the physical grid limits!
                        // Assuming GRID_WIDTH = 501
                        if (nx < 0 || nx >= 501 || ny < 0 || ny >= 501 || (!Is2D && (nz < 0 || nz >= 501))) {
                            continue; 
                        }
                        
                        // Get Morton Code for neighbor cell (nx, ny, nz)
                        // Now you can safely call the template!
                        uint32_t nHash = 0;
                        if constexpr (Is2D) {
                            nHash = Engine::Math::GetMortonCode<IsLegacy>(gx + dx, gy + dy, 0) & 0x3FFFF;
                        } else {
                            nHash = Engine::Math::GetMortonCode<IsLegacy>(gx + dx, gy + dy, gz + dz) & 0x3FFFF;
                        }

                        // 3. USE THE OFFSETS! Instantly grab the exact memory bounds for this cell
                        uint32_t neighborStart = cellStartOffsets[nHash];
                        uint32_t neighborEnd = cellStartOffsets[nHash + 1];

                        // Skip if empty
                        if (neighborStart >= neighborEnd) continue;

                        // Loop through every neighbor in the cell, one by one.
                        // We test our 8 particles against this 1 neighbor simultaneously!
                        for (size_t j = neighborStart; j < neighborEnd; ++j) {
                            
                            // Prevent particles from colliding with themselves
                            // We construct a mask where the current indices don't match the neighbor index
                            alignas(64) float maskArray[stride];
                            for(int k=0; k<stride; ++k) maskArray[k] = (i+k == j) ? 0.0f : 1.0f;
                            WideMask selfMask = WideFloat(maskArray) > WideFloat(0.5f);

                            // BROADCAST the 1 neighbor to all 8 lanes
                            WideFloat p2_x(xs[j]);
                            WideFloat p2_y(ys[j]);
                            WideFloat p2_z(zs[j]);

                            WideFloat diffX = p1_x - p2_x;
                            WideFloat diffY = p1_y - p2_y;
                            WideFloat diffZ = p1_z - p2_z;
                            WideFloat distSq = (diffX * diffX) + (diffY * diffY) + (diffZ * diffZ);

                            WideMask mask = (distSq < vRadiusSq) && (distSq > vEpsilon) && selfMask;

                            if (any_of(mask)) {
                                distSq += vEpsilon; 

                                WideFloat rsqrt_approx = rsqrt(distSq);
                                
                                WideFloat masked_repulsion = vRepulsion;
                                Engine::ISAArch::where(!mask, masked_repulsion) = 0.0f; 

                                WideFloat term = fmadd(-(distSq * half), (rsqrt_approx * rsqrt_approx), three_halves);
                                WideFloat push = (rsqrt_approx * masked_repulsion) * term;

                                // ACCUMULATE VECTOR FORCES! No reduction!
                                accX += (diffX * push);
                                accY += (diffY * push);
                                accZ += (diffZ * push);
                            }
                        }
                    }
                }
            }

            // At the end, apply the accumulated vector forces to all 8 particles simultaneously!
            WideFloat v1_x(vx + i);
            WideFloat v1_y(vy + i);
            WideFloat v1_z(vz + i);

            v1_x += accX;
            v1_y += accY;
            v1_z += accZ;

            v1_x.copy_to(vx + i);
            v1_y.copy_to(vy + i);
            v1_z.copy_to(vz + i);
        }
    }

    // --- DEDICATED PHYSICS BENCHMARK KERNEL ---
    // Pure ALU stress test. Executes the gravity math 'repeats' times inside the CPU L1 cache with zero memory barriers or OS/JobSystem interruptions.
    template <typename Abi>
    void BenchmarkParticlesTemplate(float* xs, float* ys, float* zs, 
                                    float* vx, float* vy, float* vz, 
                                    size_t count, float deltaTime, float gravityVal, int64_t repeats) {
        
        using WideFloat = Engine::ISAArch::simd<float, Abi>;
        constexpr size_t stride = WideFloat::size();

        WideFloat dt(deltaTime);
        WideFloat gravity(gravityVal);
        WideFloat damping(1.0f);
        WideFloat epsilon(0.001f);

        // The repeats happen INSIDE the kernel to bypass function call overhead
        for (int64_t r = 0; r < repeats; ++r) {
            for (size_t i = 0; i < count; i += stride) {
                
                // 1. Load Data
                WideFloat pX(xs + i);
                WideFloat pY(ys + i);
                WideFloat pZ(zs + i);
                
                WideFloat vX(vx + i);
                WideFloat vY(vy + i);
                WideFloat vZ(vz + i);

                // 2. Math: Distance to Center
                WideFloat distSq = (pX * pX) + (pY * pY) + (pZ * pZ);
                WideFloat invDist = rsqrt(distSq + epsilon); 
                WideFloat pull = invDist * gravity;

                // 3. Apply Pull
                vX = vX - (pX * pull);
                vY = vY - (pY * pull);
                vZ = vZ - (pZ * pull);

                // 4. Update Position & Damping
                vX *= damping;
                vY *= damping;
                vZ *= damping;

                pX += vX * dt;
                pY += vY * dt;
                pZ += vZ * dt;

                // 5. Store Data
                pX.copy_to(xs + i);
                pY.copy_to(ys + i);
                pZ.copy_to(zs + i);

                vX.copy_to(vx + i);
                vY.copy_to(vy + i);
                vZ.copy_to(vz + i);
            }
        }
    }
}
