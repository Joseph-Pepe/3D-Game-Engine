#pragma once

#include "EntityComponentSystem.h"
#include "../PhysicsSystem.h"
#include "../SIMD/SIMDVectorMath.h"
#include "../JobSystem.h"

#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__clang__) || defined(__GNUC__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT // Fallback to nothing if unsupported
#endif


/* [Wire ECS]: Data separated from logic in Data Oriented Design.

int main() {
    // 1. Initialize Engine Architecture
    ECS registry;
    PhysicsSystem physicsSystem;

    // ... (Setup entities and add components using your ECS) ...

    float deltaTime = 0.016f; // Assuming ~60 FPS

    // 2. The Main Engine Loop
    bool isRunning = true;
    while (isRunning) {
        
        // --- 1. Get the raw tightly packed data ---
        // Using the compile-time component trait logic to get the dense vector of AoSoA chunks
        auto& physicsChunks = registry.GetDenseChunkArray<PhysicsComponent>();

        // --- 2. Execute the Systems ---
        // We pass the raw data array directly to the system. 
        // The system has zero virtual function calls, zero cache misses, and zero pointer chasing.
        physicsSystem.Update(physicsChunks, deltaTime);

        // --- 3. Render / End Frame ---
        // graphicsSystem.Render(...);
    }

    return 0;
}
*/

// ==================================================================================
// THE SYSTEMS LAYER
// ==================================================================================
/*
    In Data-Oriented Design, Systems contain zero state. 
    They are pure functions that transform data from State A to State B.
*/

class ParticlePhysicsSystem {
public:
    void Update(ECS& ecs, float dt) {
        // 1. Ask the ECS for the contiguous array of emitter components
        // Correctly fetch the packed dense array from the sparse set manager
        auto& emitters = ecs.GetSparseDenseArray<ParticleEmitterComponent>();
        
        // 2. Iterate serially over the emitters (Outer loop is scalar)
        for (auto& emitter : emitters) {
            if (!emitter.isAwake) continue;

            // 3. Offload the heavy inner-loop math to the hardware-accelerated EngineTick.
            // EngineTick handles its own Job System threading internally.
            // Ensure the pointer is valid, then dereference it (*) to pass by reference
            if (emitter.memoryBlock) {
                Engine::Physics::EngineTick(*(emitter.memoryBlock), dt);
            }
        }
    }
};

class PhysicsSystem {
public:
    void Update(ECS& ecs, float dt) {
        auto query = ecs.Query<TransformComponent, PhysicsComponent>();
        if (query.count == 0) return;

        Engine::Physics::NativeFloatSIMDBatch dtBatch(dt);
        Engine::Physics::NativeFloatSIMDBatch oneBatch(1.0f);

        for (size_t chunkIdx = 0; chunkIdx < query.chunks.size(); ++chunkIdx) {
            uint32_t activeCount = query.chunkActiveCounts[chunkIdx];
            
            auto* RESTRICT transforms = std::get<0>(query.chunks[chunkIdx]);
            auto* RESTRICT physics    = std::get<1>(query.chunks[chunkIdx]);

            uint32_t batchCount = (activeCount + Engine::Physics::NATIVE_BATCH_SIZE - 1) / Engine::Physics::NATIVE_BATCH_SIZE;

            g_JobSystem.DispatchAndWait(batchCount, 16, [&](uint32_t start, uint32_t end) {
                
                for (uint32_t b = start; b < end; ++b) {
                    // 1. Calculate movement vector: velocity * dt
                    Engine::Physics::NativeFloatSIMDBatch moveX = physics[b].velX * dtBatch;
                    Engine::Physics::NativeFloatSIMDBatch moveY = physics[b].velY * dtBatch;
                    Engine::Physics::NativeFloatSIMDBatch moveZ = physics[b].velZ * dtBatch;

                    // 2. Branchless Masking
                    // If isStatic is 1.0f, activeMask becomes 0.0f [Inverse].
                    // If isStatic is 0.0f, activeMask becomes 1.0f [Inverse].
                    Engine::Physics::NativeFloatSIMDBatch activeMask = oneBatch - physics[b].isStaticMask;

                    // Multiply the movement by the mask. Static objects now have a movement of exactly 0.0f.
                    moveX *= activeMask;
                    moveY *= activeMask;
                    moveZ *= activeMask;

                    // 3. Fast Scatter (Still necessary until TransformComponent becomes AoSoA)
                    for (uint32_t lane = 0; lane < Engine::Physics::NATIVE_BATCH_SIZE; ++lane) {
                        uint32_t absoluteIdx = (b * Engine::Physics::NATIVE_BATCH_SIZE) + lane;
                        
                        // We still need the bounds check to prevent memory corruption at the end of the chunk
                        if (absoluteIdx < activeCount) {
                            // No branch needed (if (physics[b].isStaticMask[lane] < 0.5f))! Static objects just add 0.0f.
                            transforms[absoluteIdx].position.x += moveX[lane];
                            transforms[absoluteIdx].position.y += moveY[lane];
                            transforms[absoluteIdx].position.z += moveZ[lane];
                        }
                    }
                }
            });
        }
    }
};

class AISystem {
public:
    void Update(ECS& ecs, float dt) {
        // 1. Query the ECS using Archetypes/Signatures.
        auto query = ecs.Query<PositionComponent, AITargetComponent, AIMovementComponent>();
        if (query.count == 0) return;

        // 2. Iterate through the hardware-aligned chunks
        for (size_t chunkIdx = 0; chunkIdx < query.chunks.size(); ++chunkIdx) {
            
            uint32_t activeCount = query.chunkActiveCounts[chunkIdx];
            
            // Extract the raw pointers for this specific chunk using std::get
            auto* RESTRICT aiPositions = std::get<0>(query.chunks[chunkIdx]);
            auto* RESTRICT aiTargets   = std::get<1>(query.chunks[chunkIdx]);
            auto* RESTRICT aiMovement  = std::get<2>(query.chunks[chunkIdx]);

            uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
            uint32_t chunkSize = std::max(64u, activeCount / threadCount);

            // 3. Dispatch the Job System for this specific chunk
            g_JobSystem.DispatchAndWait(activeCount, chunkSize, [&](uint32_t start, uint32_t end) {
                
                ENGINE_UNROLL_4
                for (uint32_t i = start; i < end; ++i) {
                    
                    float dirX = aiTargets[i].targetX - aiPositions[i].x;
                    float dirY = aiTargets[i].targetY - aiPositions[i].y;
                    float dirZ = aiTargets[i].targetZ - aiPositions[i].z;

                    float lengthSq = (dirX * dirX) + (dirY * dirY) + (dirZ * dirZ);
                    float invLength = 1.0f / std::sqrt(lengthSq + 1e-8f);

                    float moveX = dirX * invLength * aiMovement[i].speed * dt;
                    float moveY = dirY * invLength * aiMovement[i].speed * dt;
                    float moveZ = dirZ * invLength * aiMovement[i].speed * dt;

                    aiPositions[i].x += moveX;
                    aiPositions[i].y += moveY;
                    aiPositions[i].z += moveZ;
                }
            });
        }
    }
};
