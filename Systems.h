#pragma once

#include "EntityComponentSystem.h"
#include "PhysicsSystem.h"

#include <cstdint>
#include <immintrin.h>

// The System that bridges the ECS to the Silicon
class ParticleSystem {
public:
    void Update(ECS& ecs, float dt) {
        auto& emitters = ecs.GetArray<ParticleEmitterComponent>();
        
        // Grab the raw contiguous array of components
        for (uint32_t e = 0; e < ecs.GetMaxEntities(); ++e) {
            if (!ecs.HasComponent<ParticleEmitterComponent>(e)) continue;
            
            auto& emitter = emitters[e];
            if (!emitter.isAwake || !emitter.physicsEngine) continue;

            // The ECS triggers the massive AVX2 Job-System integration!
            emitter.physicsEngine->buildSpatialGridParallel(emitter.activeParticles);
            emitter.physicsEngine->solveCollisions(emitter.activeParticles);
            emitter.physicsEngine->integrate(dt, emitter.activeParticles, emitter.gravityPull);
        }
    }
};

class AISystem {
public:
    void Update(ECS& ecs, float dt) {
        auto& aiComponents = ecs.GetArray<AIComponent>();
        uint32_t entityCount = ecs.GetMaxEntities();

        // 1. Ask the Job System for threads
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t chunkSize = std::max(256u, entityCount / threadCount);

        // 2. Multithread the entire AI logic step!
        g_JobSystem.DispatchAndWait(entityCount, chunkSize, [&](uint32_t start, uint32_t end) {
            for (uint32_t e = start; e < end; ++e) {
                // Skip dead or irrelevant entities
                if (!ecs.HasComponent<AIComponent>(e)) continue;

                auto& ai = aiComponents[e];
                // Execute standard scalar C++ AI logic across all CPU cores...
                ai.ProcessState(dt);
            }
        });
    }
};
