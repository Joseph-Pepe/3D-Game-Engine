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

class ParticlePhysicsSystem {
public:
    void Update(ECS& ecs, float dt) {
        // 1. Query the ECS for all active Particle Emitters
        auto& emitters = ecs.GetDenseArray<ParticleEmitterComponent>();
        
        // 2. Iterate serially over the emitters (there are usually < 100 emitters)
        for (auto& emitter : emitters) {
            if (!emitter.isAwake) continue;

            // 3. The SIMD math handles its own Job System dispatching internally!
            // We just pass the raw data block to the EngineTick.
            Engine::Physics::EngineTick(emitter.memoryBlock, dt);
        }
    }
};

class AISystem {
public:
    void Update(ECS& ecs, float dt) {
        // In DOD, we don't have one big "AIComponent". 
        // We split it into tight, cache-aligned data streams.
        auto& aiPositions = ecs.GetDenseArray<PositionComponent>();
        auto& aiTargets   = ecs.GetDenseArray<AITargetComponent>();
        auto& aiStates    = ecs.GetDenseArray<AIStateComponent>();
        
        uint32_t count = static_cast<uint32_t>(aiPositions.size());
        if (count == 0) return;

        // 1. Ask the Job System for threads
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t chunkSize = std::max(64u, count / threadCount);

        // 2. Multithread the logic, keeping the data completely flat
        g_JobSystem.DispatchAndWait(count, chunkSize, [&](uint32_t start, uint32_t end) {
            
            // The compiler will vectorize this loop because there are no function calls!
            for (uint32_t i = start; i < end; ++i) {
                // If the AI is not moving, skip it (Branchless math is better, but this is a simple example)
                if (aiStates[i].state == AI_STATE_IDLE) continue;

                // Simple scalar math (or use your ISAArch abstractions)
                float dirX = aiTargets[i].targetX - aiPositions[i].x;
                float dirY = aiTargets[i].targetY - aiPositions[i].y;
                float dirZ = aiTargets[i].targetZ - aiPositions[i].z;

                // Normalize and move (Pseudo-code)
                // ...
            }
        });
    }
};
