StructuredBuffer<InstanceData> AllInstances : register(t0);
StructuredBuffer<uint> VisibleIndices : register(t1);

struct VSInput {
    float3 Position : POSITION;
    uint InstanceID : SV_InstanceID;
};

struct VSOutput {
    float4 PositionCS : SV_POSITION;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;

    // INDIRECTION: Look up the real instance ID from our culled array
    uint realInstanceID = VisibleIndices[input.InstanceID];
    
    // Fetch the matrix
    float4x4 modelMatrix = AllInstances[realInstanceID].ModelMatrix;

    // Standard transformation
    float4 worldPos = mul(modelMatrix, float4(input.Position, 1.0));
    output.PositionCS = mul(ViewProjectionMatrix, worldPos);

    return output;
}
