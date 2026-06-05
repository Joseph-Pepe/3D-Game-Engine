#pragma once

#include "EntityComponentSystem.h"
#include "PhysicsSystem.h"

#include <cstdint>
#include <immintrin.h>


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

// Only cares about pure, raw arrays of data.
class PhysicsSystem {
public:
    // This is called once per frame from your main engine loop
    void Update(std::vector<PhysicsChunk8>& chunks, float deltaTime) {
        
        // 1. BROADCAST DELTA TIME
        // Load the single float 'deltaTime' into all 8 lanes of a 256-bit register.
        // It looks like: [dt, dt, dt, dt, dt, dt, dt, dt]
        __m256 dt = _mm256_set1_ps(deltaTime); // Broadcast delta time to all 8 lanes of a 256-bit register

        // 2. ITERATE OVER THE CHUNKS
        for (auto& chunk : chunks) {
            
            // Optimization: If the chunk is completely empty, skip it.
            if (chunk.activeCount == 0) continue;

            // 1. Load velocities directly into registers (Zero shuffling!)


            // --- LOAD DATA FROM RAM INTO CPU REGISTERS ---
            // _mm256_load_ps requires the memory to be 32-byte aligned (which we did above!)
            // We are loading the X, Y, and Z velocities for 8 entities simultaneously.
            __m256 vX = _mm256_load_ps(chunk.velX);
            __m256 vY = _mm256_load_ps(chunk.velY);
            __m256 vZ = _mm256_load_ps(chunk.velZ);

            __m256 pX = _mm256_load_ps(chunk.posX);
            __m256 pY = _mm256_load_ps(chunk.posY);
            __m256 pZ = _mm256_load_ps(chunk.posZ);

            // 2. Perform math on 8 entities simultaneously.

            // --- PERFORM SIMD MATH (8 Entities at exactly the same time) ---
            // _mm256_fmadd_ps does: (A * B) + C in a single CPU instruction (Fused Multiply-Add).
            // Here we are doing: new_pos = (velocity * dt) + old_pos
            pX = _mm256_fmadd_ps(vX, dt, pX);
            pY = _mm256_fmadd_ps(vY, dt, pY);
            pZ = _mm256_fmadd_ps(vZ, dt, pZ);

            // --- STORE RESULTS BACK TO RAM ---
            _mm256_store_ps(chunk.posX, pX);
            _mm256_store_ps(chunk.posY, pY);
            _mm256_store_ps(chunk.posZ, pZ);
        }
    }
};

// The System that bridges the ECS to the Silicon
class ParticleSystem {
public:
    void Update(ECS& ecs, float dt) {
        // Grab the raw contiguous array directly
        auto& emitters = ecs.GetDenseArray<ParticleEmitterComponent>();
        
        // Grab the raw contiguous array of components
        // Iterate ONLY over the active components. If there are 50 emitters, this loop runs exactly 50 times. Zero cache misses.
        for (auto& emitter : emitters) {
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
        auto& aiComponents = ecs.GetDenseArray<AIComponent>();
        uint32_t activeAICount = static_cast<uint32_t>(aiComponents.size());

        if (activeAICount == 0) return;

        // 1. Ask the Job System for threads
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);
        uint32_t chunkSize = std::max(64u, activeAICount / threadCount);

        // 2. Multithread the entire AI logic step! Dispatch based on active component count, not max entities
        g_JobSystem.DispatchAndWait(activeAICount, chunkSize, [&](uint32_t start, uint32_t end) {
            for (uint32_t i = start; i < end; ++i) {
                // Execute standard scalar C++ AI logic across all CPU cores...
                aiComponents[i].ProcessState(dt);
            }
        });
    }
};
