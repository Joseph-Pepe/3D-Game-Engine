#pragma once

#include "Math.h"  

// ==================================================================================
// RENDERER TRANSFORM PIPELINE
// ==================================================================================
/*
    - Instanced Based Rendering: Send the GPU exactly one 4x4 Matrix (the Camera) and you send a tightly packed buffer of only raw X, Y, Z positions (100,000 particles, 16 bytes per particle = 1.6MB).
    - GPU APIs do not understand SIMD batches, they expect Array of Structures (AOS).
    - We need a tightly packed 16-byte struct.
*/

// This struct perfectly matches the layout the GPU Vertex Shader expects.
// 16 Bytes total.
struct alignas(16) GPUInstanceData {
    float x, y, z;
    float scale; // We pack scale or radius into the 4th lane to maximize bandwidth!
};

// Calculates the unified View-Projection matrix that the GPU will use for the entire frame.
FORCE_INLINE Matrix4 CalculateCameraMatrix(const Vector3DWorld& cameraWorldPos, 
                                           const Vector3DWorld& cameraTarget, 
                                           float fov, float aspect, float nearZ, float farZ) {
                                               
    // 1. Build the LWC Camera-Relative View Matrix (Translation is forced to 0)
    Vector3D upVec(0.0f, 1.0f, 0.0f);
    Matrix4 view = Matrix4::LookAtLWC(cameraWorldPos, cameraTarget, upVec);

    // 2. Build the Standard Projection Matrix
    Matrix4 proj = Matrix4::Perspective(fov, aspect, nearZ, farZ);

    // 3. Multiply Proj * View (Note: You should implement a standard 4x4 matrix multiplication for your Matrix4 struct)
    Matrix4 viewProj = proj * view; 

    return viewProj;
}

// --- 6. GPU DATA EXTRACTION ---
// Particles: currently stored as [xxxx] [yyyy] [zzzz], the GPU wants them formatted as [xyzs] [xyzs] [xyzs]
// Un-swizzles our hardware-accelerated AoSoA batches into a flat GPU-ready array.
FORCE_INLINE void ExtractRenderData(std::span<const SIMDVector3D> positions, 
                                    std::span<GPUInstanceData> outGPUBuffer, 
                                    size_t activeCount, 
                                    float uniformScale) {
    
    size_t activeBatches = (activeCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;

    // Temporary stack arrays to flush the SIMD registers into scalar memory
    alignas(NATIVE_SIMD_BATCH_ALIGN) float tempX[NATIVE_BATCH_SIZE];
    alignas(NATIVE_SIMD_BATCH_ALIGN) float tempY[NATIVE_BATCH_SIZE];
    alignas(NATIVE_SIMD_BATCH_ALIGN) float tempZ[NATIVE_BATCH_SIZE];

    for (size_t i = 0; i < activeBatches; ++i) {
        // 1. Flush the SIMD registers to aligned stack memory
        positions[i].x.copy_to(tempX, std::element_aligned);
        positions[i].y.copy_to(tempY, std::element_aligned);
        positions[i].z.copy_to(tempZ, std::element_aligned);

        // 2. Interleave the data into the GPU format (AoS)
        for (size_t lane = 0; lane < NATIVE_BATCH_SIZE; ++lane) {
            size_t absoluteIdx = (i * NATIVE_BATCH_SIZE) + lane;
            
            // Stop extracting if we hit the end of the active particles
            if (absoluteIdx >= activeCount) break;

            // 3. Write directly to the mapped GPU buffer
            outGPUBuffer[absoluteIdx].x = tempX[lane];
            outGPUBuffer[absoluteIdx].y = tempY[lane];
            outGPUBuffer[absoluteIdx].z = tempZ[lane];
            outGPUBuffer[absoluteIdx].scale = uniformScale; 
        }
    }
}

// ==========================================
// THE MAIN GAME LOOP
// ==========================================
/*
    // 1. Setup the GPU Buffer (Usually mapped directly from Vulkan/OpenGL)
    std::vector<GPUInstanceData> gpuInstanceBuffer(maxParticles);

    // 2. Define the Camera in 64-bit space
    Vector3DWorld cameraPos(5000000.0, 150.0, 200000.0);
    Vector3DWorld targetPos(5000000.0, 150.0, 199900.0);
*/

void Tick(ParticleMemoryBlock& memory, std::span<GPUInstanceData> mappedGPUBuffer, float deltaTime) {
    
    // --- 1. MEMORY VIEWS ---
    size_t activeBatches = (memory.activeParticleCount + NATIVE_BATCH_SIZE - 1) / NATIVE_BATCH_SIZE;
    
    std::span<SIMDVector3D> posSpan(memory.positions.data(), activeBatches);
    std::span<SIMDVector3D> velSpan(memory.velocities.data(), activeBatches);
    std::span<ParticleSortKey> keySpan(memory.sortKeys.data(), memory.activeParticleCount);
    std::span<ParticleSortKey> bufferSpan(memory.sortKeysBuffer.data(), memory.activeParticleCount);

    // --- 2. PHYSICS & COLLISION PIPELINE ---
    UpdateParticles(posSpan, velSpan, memory.activeParticleCount, deltaTime);
    GenerateMortonKeys(posSpan, keySpan, memory.activeParticleCount);
    RadixSortKeys(keySpan, bufferSpan);

    std::vector<SIMDVector3D> tempPos(activeBatches);
    std::vector<SIMDVector3D> tempVel(activeBatches);
    ReorderParticleData(posSpan, velSpan, tempPos, tempVel, keySpan);

    std::copy(tempPos.begin(), tempPos.end(), memory.positions.begin());
    std::copy(tempVel.begin(), tempVel.end(), memory.velocities.begin());

    ResolveCollisions(posSpan, memory.activeParticleCount, 2.0f);

    // --- 3. RENDER PREPARATION PIPELINE ---
    // Pass the read-only physics span and the write-only GPU span
    ExtractRenderData(posSpan, mappedGPUBuffer, memory.activeParticleCount, 2.0f);

    // --- 4. CAMERA MATH & DRAW (Pseudo-code) ---
    Vector3DWorld cameraPos(5000000.0, 150.0, 200000.0);
    Vector3DWorld targetPos(5000000.0, 150.0, 199900.0);
    Matrix4 cameraViewProj = CalculateCameraMatrix(cameraPos, targetPos, 90.0f, 16.0f/9.0f, 0.1f, 10000.0f);

    // GPU_SetUniformMatrix4("u_ViewProjection", cameraViewProj.m.data());
    // GPU_DrawInstanced(RenderMesh::Quad, memory.activeParticleCount);
}
