// ====================================================
// HLSL COMPUTE SHADER (GPU)
// ====================================================

// Frustium Cull Pass to filter out instances before they reach the raycast and render pipelines .
struct InstanceData {
    float4x4 ModelMatrix;
    float3 BoundsCenter;
    float BoundsRadius;
};

// Input
StructuredBuffer<InstanceData> InInstances : register(t0);
cbuffer CameraData : register(b0) {
    float4 FrustumPlanes[6]; // xyz = normal, w = distance
    uint TotalInstances;
};

// Output
RWStructuredBuffer<uint> OutVisibleIndices : register(u0);

// Indirect Argument Buffer (Index 1 is 'InstanceCount')
RWByteAddressBuffer OutIndirectDrawArgs : register(u1); 

// 64 threads per threadgroup (Matches AMD Wavefront64, runs as 2x32 on Nvidia)
[numthreads(64, 1, 1)]
void CSMain(uint3 DispatchThreadID : SV_DispatchThreadID)
{
    uint instanceID = DispatchThreadID.x;
    bool isVisible = false;

    // 1. EVALUATE VISIBILITY
    if (instanceID < TotalInstances) {
        InstanceData inst = InInstances[instanceID];
        isVisible = true;

        // Unroll the 6 plane checks. 
        // Dot product of plane normal and sphere center + plane distance.
        [unroll]
        for (int i = 0; i < 6; ++i) {
            float dist = dot(FrustumPlanes[i].xyz, inst.BoundsCenter) + FrustumPlanes[i].w;
            // If the sphere is completely behind the plane, cull it.
            if (dist < -inst.BoundsRadius) {
                isVisible = false;
                break;
            }
        }
    }

    // =====================================================================
    // 2. THE WAVE INTRINSIC OPTIMIZATION (THE ATOMIC KILLER)
    // =====================================================================
    // Instead of every visible thread hitting main VRAM with an InterlockedAdd,
    // we use hardware registers to count how many threads in this 64-lane wave survived.

    // Get a 64-bit bitmask of all threads in this wave that evaluated to true
    uint4 waveBallot = WaveActiveBallot(isVisible); 
    
    // Count how many total bits are 1 (How many instances survived in this wave)
    uint waveVisibleCount = WaveActiveCountBits(isVisible);

    // Count how many surviving instances came BEFORE this specific thread in the wave
    uint wavePrefixSum = WavePrefixCountBits(isVisible);

    // Only the FIRST active thread in the wave talks to global VRAM
    uint baseOutputOffset = 0;
    if (WaveIsFirstLane() && waveVisibleCount > 0) {
        // We do ONE atomic add per 64 instances instead of 64 atomic adds!
        // This adds waveVisibleCount to the IndirectDrawArgs.InstanceCount (Byte offset 4)
        OutIndirectDrawArgs.InterlockedAdd(4, waveVisibleCount, baseOutputOffset);
    }

    // Broadcast the base output offset to all other threads in the wave instantly
    baseOutputOffset = WaveReadLaneFirst(baseOutputOffset);

    // 3. WRITE SURVIVORS TO GLOBAL MEMORY
    if (isVisible) {
        // Calculate the exact unique array index for this specific instance
        uint finalGlobalIndex = baseOutputOffset + wavePrefixSum;
        
        // Store the original InstanceID so the Vertex Shader knows what to draw
        OutVisibleIndices[finalGlobalIndex] = instanceID;
    }
}
