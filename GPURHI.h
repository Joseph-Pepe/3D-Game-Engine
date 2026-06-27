#pragma once

#include "Math.h"  

// ===============================================
// RENDERING HARDWARE INTERFACE (RHI) STUBS
// ===============================================
// These are opaque handles. In your actual backend, these will map 
// to ID3D12Resource* (DX12) or VkBuffer/VkImage (Vulkan).
struct RHIBuffer {};
struct RHITexture {};

// Forward declaration so the CommandList knows about it
struct RHIBarrier;

// The abstraction layer for your GPU Command Buffer
class RHICommandList {
public:
    void ClearUAVUint(RHIBuffer* buffer, uint32_t offset, uint32_t value) {}
    
    void SetComputePipelineState(void* pso) {}
    void SetGraphicsPipelineState(void* pso) {}
    
    void SetComputeRootConstantBuffer(uint32_t rootIndex, void* cbv) {}
    
    void SetComputeRootShaderResourceView(uint32_t rootIndex, RHIBuffer* srv) {}
    void SetComputeRootShaderResourceView(uint32_t rootIndex, RHITexture* srv) {}
    void SetGraphicsRootShaderResourceView(uint32_t rootIndex, RHIBuffer* srv) {}
    
    void SetComputeRootUnorderedAccessView(uint32_t rootIndex, RHIBuffer* uav) {}
    void SetComputeRootUnorderedAccessView(uint32_t rootIndex, RHITexture* uav) {}
    
    void Dispatch(uint32_t x, uint32_t y, uint32_t z) {}
    void ExecuteIndirect(void* commandSignature, uint32_t maxCommandCount, RHIBuffer* argumentBuffer, uint64_t argumentBufferOffset) {}
    
    void ResourceBarrier(uint32_t count, const RHIBarrier* barriers) {}
};

// ===============================================
// GPU PIPELINE (Frustum -> Occlusion -> Draw)
// ===============================================
/*
    - Build a Single Pass Downsampling (SPD) and Wave-Compacted Two-Pass Culling.
    - Two-Pass Hierarchical Z-Buffer (HZB) Occlusion Culling System

    1. History Pass: Test all instances against last frame's HZB. If they pass, draw them immediately to generate the current frame's depth buffer. 
    2. HZB Build Pass: The current depth buffer is downsampled into a mipmapped HZB.
    3. Refinement Pass: Any instance that failed the history pass is tested against the new HZB. If it passes (occluded last frame, but the camera moved and it is now visible), it is appended to a second draw list and drawn.
*/


// ===============================================
// RENDERING HARDWARE INTERFACE (RHI)
// ===============================================
/*
    - Is a command list interface that abstracts the modern GPU APIs (DirectX 12, Vulkan, Metal).
*/
enum RHI_RESOURCE_STATE {
    RHI_RESOURCE_STATE_UNORDERED_ACCESS,
    RHI_RESOURCE_STATE_INDIRECT_ARGUMENT,
    RHI_RESOURCE_STATE_SHADER_RESOURCE
};

struct RHIBarrier {
    static RHIBarrier Transition(RHIBuffer* buffer, RHI_RESOURCE_STATE before, RHI_RESOURCE_STATE after) { 
        return {}; 
    }
    // Overload to support transitioning our HZB textures
    static RHIBarrier Transition(RHITexture* texture, RHI_RESOURCE_STATE before, RHI_RESOURCE_STATE after) { 
        return {}; 
    }
};

class RenderSystem {
private:
    // Placeholder for your actual PSO objects
    void* m_FrustumCullPSO = nullptr;
    void* m_OpaqueMeshPSO = nullptr;
    void* m_CommandSignature = nullptr;
    void* m_CameraConstants = nullptr;

public:
    void ExecuteGPUCullingAndDraw(
        RHICommandList* cmdList, 
        RHIBuffer* instanceBuffer, 
        RHIBuffer* indirectArgsBuffer,
        RHIBuffer* visibleIndicesBuffer,
        uint32_t totalInstances) 
    {
        // 1. Reset the Indirect Argument's 'InstanceCount' to 0 before the compute pass
        // Offset 4 bytes = the 'InstanceCount' variable
        cmdList->ClearUAVUint(indirectArgsBuffer, 4, 0); 

        // 2. SET COMPUTE STATE
        cmdList->SetComputePipelineState(m_FrustumCullPSO);
        
        // Bind Buffers
        cmdList->SetComputeRootConstantBuffer(0, m_CameraConstants);
        cmdList->SetComputeRootShaderResourceView(1, instanceBuffer);
        cmdList->SetComputeRootUnorderedAccessView(2, visibleIndicesBuffer);
        cmdList->SetComputeRootUnorderedAccessView(3, indirectArgsBuffer);

        // 3. DISPATCH COMPUTE SHADER
        // Divide total instances by 64 (threads per group) and round up
        uint32_t threadGroupsX = (totalInstances + 63) / 64;
        cmdList->Dispatch(threadGroupsX, 1, 1);

        // ==========================================
        // 4. THE RESOURCE BARRIER (CRITICAL)
        // ==========================================
        RHIBarrier barriers[] = {
            RHIBarrier::Transition(
                indirectArgsBuffer, 
                RHI_RESOURCE_STATE_UNORDERED_ACCESS, 
                RHI_RESOURCE_STATE_INDIRECT_ARGUMENT),
            RHIBarrier::Transition(
                visibleIndicesBuffer, 
                RHI_RESOURCE_STATE_UNORDERED_ACCESS, 
                RHI_RESOURCE_STATE_SHADER_RESOURCE)
        };
        cmdList->ResourceBarrier(2, barriers);

        // 5. SET GRAPHICS STATE
        cmdList->SetGraphicsPipelineState(m_OpaqueMeshPSO);
        
        // The Vertex Shader reads from 'visibleIndicesBuffer' 
        cmdList->SetGraphicsRootShaderResourceView(0, visibleIndicesBuffer); 

        // 6. EXECUTE INDIRECT DRAW
        cmdList->ExecuteIndirect(
            m_CommandSignature, // Defines that this is a DrawIndexedInstanced command
            1,                  // Max command count
            indirectArgsBuffer, 
            0                   // Buffer offset
        );

        // 7. Transition back for the next frame
        RHIBarrier revertBarriers[] = {
            RHIBarrier::Transition(indirectArgsBuffer, RHI_RESOURCE_STATE_INDIRECT_ARGUMENT, RHI_RESOURCE_STATE_UNORDERED_ACCESS),
            RHIBarrier::Transition(visibleIndicesBuffer, RHI_RESOURCE_STATE_SHADER_RESOURCE, RHI_RESOURCE_STATE_UNORDERED_ACCESS)
        };
        cmdList->ResourceBarrier(2, revertBarriers);
    }

    // ==============================================================
    // SINGLE PASS DOWNSAMPLER (SPD)
    // ==============================================================
    /*
        - AMD FidelityFX Single Pass Downsampler.
        - Uses a global atomic counter and Wave intrinsics to generate an entire 12-level mip chain in a single compute dispatch.
        - Cull -> Draw -> Generate Mips -> Cull -> Draw
    */
    void ExecuteTwoPassOcclusion(RHICommandList* cmdList, uint32_t totalInstances) {
        uint32_t threadGroupsX = (totalInstances + 63) / 64;

        // ==========================================
        // PHASE 1: HISTORY PASS
        // ==========================================
        cmdList->ClearUAVUint(m_IndirectArgsBufferPhase1, 4, 0); // Reset InstanceCount to 0
        
        m_CullingConstants.IsPhase2 = 0;
        cmdList->SetComputeRootConstantBuffer(0, &m_CullingConstants);
        
        // Bind LAST FRAME's HZB (Using an alternating frame index)
        cmdList->SetComputeRootShaderResourceView(4, m_HZBTexture[m_PreviousFrameIndex]); 
        
        cmdList->SetComputePipelineState(m_GPUCullingPSO);
        cmdList->Dispatch(threadGroupsX, 1, 1);

        // Barrier: Wait for Phase 1 Args and Bitmask to finish writing
        RHIBarrier p1Barriers[] = {
            RHIBarrier::Transition(m_IndirectArgsBufferPhase1, UAV, INDIRECT_ARG),
            RHIBarrier::Transition(m_OcclusionBitmaskBuffer, UAV, UAV) // UAV barrier ensures bitmask writes complete
        };
        cmdList->ResourceBarrier(2, p1Barriers);


        // ==========================================
        // DRAW 1: RENDER VISIBLE GEOMETRY
        // ==========================================
        // This draws all objects that were visible last frame.
        cmdList->SetGraphicsPipelineState(m_OpaqueMeshPSO);
        cmdList->ExecuteIndirect(m_CommandSignature, 1, m_IndirectArgsBufferPhase1, 0);

        // Barrier: Transition the newly written Depth Buffer so the Compute Shader can read it
        cmdList->ResourceBarrier(Transition(m_MainDepthBuffer, DEPTH_WRITE, SHADER_RESOURCE));


        // ==========================================
        // BUILD NEW HZB (Single Pass Downsample)
        // ==========================================
        cmdList->SetComputePipelineState(m_SpdDownsamplePSO);
        
        // Read from the Main Depth Buffer, output all Mip levels to the Current Frame's HZB
        cmdList->SetComputeRootShaderResourceView(0, m_MainDepthBuffer);
        cmdList->SetComputeRootUnorderedAccessView(1, m_HZBTexture[m_CurrentFrameIndex]); 
        
        // AMD SPD handles all 12 mips in a single dispatch!
        cmdList->Dispatch(SpdCalculateThreadGroups(m_ScreenWidth, m_ScreenHeight), 1, 1);

        // Barrier: Wait for the HZB to finish generating
        cmdList->ResourceBarrier(Transition(m_HZBTexture[m_CurrentFrameIndex], UAV, SHADER_RESOURCE));


        // ==========================================
        // PHASE 2: REFINEMENT PASS
        // ==========================================
        cmdList->ClearUAVUint(m_IndirectArgsBufferPhase2, 4, 0); // Reset Phase 2 Args
        
        m_CullingConstants.IsPhase2 = 1;
        cmdList->SetComputeRootConstantBuffer(0, &m_CullingConstants);
        
        // Bind THIS FRAME's new HZB
        cmdList->SetComputeRootShaderResourceView(4, m_HZBTexture[m_CurrentFrameIndex]); 
        
        cmdList->SetComputePipelineState(m_GPUCullingPSO);
        cmdList->Dispatch(threadGroupsX, 1, 1); // Only tests instances flagged in the Bitmask

        // Barrier: Wait for Phase 2 Args to finish
        cmdList->ResourceBarrier(Transition(m_IndirectArgsBufferPhase2, UAV, INDIRECT_ARG));


        // ==========================================
        // DRAW 2: RENDER NEWLY VISIBLE GEOMETRY
        // ==========================================
        // Render the previously occluded instances that are now visible!
        // (e.g., The camera walked around a corner, exposing a new hallway).
        cmdList->SetGraphicsPipelineState(m_OpaqueMeshPSO);
        cmdList->ExecuteIndirect(m_CommandSignature, 1, m_IndirectArgsBufferPhase2, 0);


        // ==========================================
        // CLEANUP FOR NEXT FRAME
        // ==========================================
        // Swap the frame indices so 'Current' becomes 'Previous' for the next frame.
        std::swap(m_CurrentFrameIndex, m_PreviousFrameIndex);
        
        RHIBarrier revertBarriers[] = {
            RHIBarrier::Transition(m_MainDepthBuffer, SHADER_RESOURCE, DEPTH_WRITE),
            RHIBarrier::Transition(m_IndirectArgsBufferPhase1, INDIRECT_ARG, UAV),
            RHIBarrier::Transition(m_IndirectArgsBufferPhase2, INDIRECT_ARG, UAV)
        };
        cmdList->ResourceBarrier(3, revertBarriers);
    }
};


// ===============================================
// GPU RENDERING PIPELINE (FRUSTUM CULLING)
// ===============================================
/*
    - Shift the workload (loop through scene objects, bounding box checks against the camera frustum, builds a list of visible items to draw) over to the GPU.
    - CPU issues a single compute shader dispatch call, and the GPU dynamically determines visibility and generates its own draw parameters.
*/

// A raw, unaligned 4x4 matrix specifically for uploading to VRAM
// 1. Data-Transfer Matrix
struct Float4x4 {
    float m[16];
};

// 2. Exact 80-byte HLSL Match
// Perfectly matching 80-byte HLSL struct
struct GPUInstanceData {
    Float4x4 ModelMatrix;       // 64 bytes
    float BoundsCenterX;        // 4 bytes
    float BoundsCenterY;        // 4 bytes
    float BoundsCenterZ;        // 4 bytes
    float BoundsRadius;         // 4 bytes
};


// 3. Exact 20-byte Indirect Args
// Standard DirectX 12 / Vulkan Indirect Draw Arguments struct (20 bytes)
struct GPUIndirectDrawArgs {
    uint32_t IndexCountPerInstance;
    uint32_t InstanceCount;      // The Compute Shader writes to this!
    uint32_t StartIndexLocation;
    int32_t  BaseVertexLocation;
    uint32_t StartInstanceLocation;
};

// 4. Exact 112-byte Constant Buffer
// Constant Buffer for the Camera (Matches HLSL 16-byte array packing)
struct alignas(16) GPUCameraConstants {
    // 6 planes: Left, Right, Top, Bottom, Near, Far.
    struct Plane { float x, y, z, distance; } FrustumPlanes[6]; 
    
    // The compute shader needs to know the exact total to prevent reading out of bounds!
    uint32_t TotalInstances;
    
    // Explicit padding to ensure the struct ends on a 16-byte boundary
    uint32_t Padding[3]; 
};

// Runs once on the CPU per frame.
void ExtractFrustumPlanes(const Matrix4& viewProj, GPUCameraConstants& outCameraData) {
    // Left Plane
    outCameraData.FrustumPlanes[0] = {
        viewProj.m[3] + viewProj.m[0],
        viewProj.m[7] + viewProj.m[4],
        viewProj.m[11] + viewProj.m[8],
        viewProj.m[15] + viewProj.m[12]
    };

    // Right Plane
    outCameraData.FrustumPlanes[1] = {
        viewProj.m[3] - viewProj.m[0],
        viewProj.m[7] - viewProj.m[4],
        viewProj.m[11] - viewProj.m[8],
        viewProj.m[15] - viewProj.m[12]
    };

    // Top Plane
    outCameraData.FrustumPlanes[2] = {
        viewProj.m[3] - viewProj.m[1],
        viewProj.m[7] - viewProj.m[5],
        viewProj.m[11] - viewProj.m[9],
        viewProj.m[15] - viewProj.m[13]
    };

    // Bottom Plane
    outCameraData.FrustumPlanes[3] = {
        viewProj.m[3] + viewProj.m[1],
        viewProj.m[7] + viewProj.m[5],
        viewProj.m[11] + viewProj.m[9],
        viewProj.m[15] + viewProj.m[13]
    };

    // Near Plane
    outCameraData.FrustumPlanes[4] = {
        viewProj.m[3] + viewProj.m[2],
        viewProj.m[7] + viewProj.m[6],
        viewProj.m[11] + viewProj.m[10],
        viewProj.m[15] + viewProj.m[14]
    };

    // Far Plane
    outCameraData.FrustumPlanes[5] = {
        viewProj.m[3] - viewProj.m[2],
        viewProj.m[7] - viewProj.m[6],
        viewProj.m[11] - viewProj.m[10],
        viewProj.m[15] - viewProj.m[14]
    };

    // Normalize all 6 planes
    for (int i = 0; i < 6; ++i) {
        float length = std::sqrt(
            outCameraData.FrustumPlanes[i].x * outCameraData.FrustumPlanes[i].x +
            outCameraData.FrustumPlanes[i].y * outCameraData.FrustumPlanes[i].y +
            outCameraData.FrustumPlanes[i].z * outCameraData.FrustumPlanes[i].z
        );
        float invLength = 1.0f / length;

        outCameraData.FrustumPlanes[i].x *= invLength;
        outCameraData.FrustumPlanes[i].y *= invLength;
        outCameraData.FrustumPlanes[i].z *= invLength;
        outCameraData.FrustumPlanes[i].distance *= invLength;
    }
}
