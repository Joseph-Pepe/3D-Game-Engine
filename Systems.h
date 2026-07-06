#pragma once

#include "EntityComponentSystem.h"
#include "PhysicsSystem.h"
#include "SIMD/SIMDVectorMath.h"
#include "../JobSystem.h"


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
        // (Assuming GetDenseArray returns your custom small_vector or inplace_vector)
        auto& emitters = ecs.GetDenseArray<ParticleEmitterComponent>();
        
        // 2. Iterate serially over the emitters (Outer loop is scalar)
        for (auto& emitter : emitters) {
            if (!emitter.isAwake) continue;

            // 3. Offload the heavy inner-loop math to the hardware-accelerated EngineTick.
            // EngineTick handles its own Job System threading internally.
            Engine::Physics::EngineTick(emitter.memoryBlock, dt);
        }
    }
};

class AISystem {
public:
    void Update(ECS& ecs, float dt) {
        // 1. Query the ECS using Archetypes/Signatures.
        // We do NOT ask for all AI. We ask ONLY for AI that have a Target and are Moving.
        // This guarantees 100% data density. Every element returned MUST be processed.
        
        // Pseudo-code for ECS Query (Replace with your actual ECS API)
        auto query = ecs.Query<PositionComponent, AITargetComponent, AIMovementComponent>();
        
        uint32_t count = static_cast<uint32_t>(query.count);
        if (count == 0) return;

        // 2. Extract the raw, contiguous pointers
        auto* RESTRICT aiPositions = query.Get<PositionComponent>();
        auto* RESTRICT aiTargets   = query.Get<AITargetComponent>();
        auto* RESTRICT aiMovement  = query.Get<AIMovementComponent>();

        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t chunkSize = std::max(64u, count / threadCount);

        // 3. Multithread the logic
        g_JobSystem.DispatchAndWait(count, chunkSize, [&](uint32_t start, uint32_t end) {
            
            // 4. THE HOT PATH
            // Because there are absolutely ZERO 'if' statements or function calls inside this loop,
            // and we extracted __restrict pointers, the compiler will automatically vectorize this
            // math into AVX/NEON registers.
            
            // ENGINE_UNROLL_4 tells MSVC/Clang to pipeline 4 iterations simultaneously
            ENGINE_UNROLL_4
            for (uint32_t i = start; i < end; ++i) {
                
                // Vector subtraction (Target - Current)
                float dirX = aiTargets[i].targetX - aiPositions[i].x;
                float dirY = aiTargets[i].targetY - aiPositions[i].y;
                float dirZ = aiTargets[i].targetZ - aiPositions[i].z;

                // Length squared (fused multiply-add if compiler is smart)
                float lengthSq = (dirX * dirX) + (dirY * dirY) + (dirZ * dirZ);

                // Fast Inverse Square Root (auto-vectorizes to rsqrt_ps), We add a tiny epsilon to prevent division by zero without using an 'if' statement!
                // Force the hardware to emit the fast rsqrt instruction (e.g., _mm256_rsqrt_ps)
                float invLength = Engine::ISAArch::rsqrt(lengthSq + 1e-8f);

                // Normalize and apply speed
                float moveX = dirX * invLength * aiMovement[i].speed * dt;
                float moveY = dirY * invLength * aiMovement[i].speed * dt;
                float moveZ = dirZ * invLength * aiMovement[i].speed * dt;

                // Write back to memory
                aiPositions[i].x += moveX;
                aiPositions[i].y += moveY;
                aiPositions[i].z += moveZ;
            }
        });
    }
};
