#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).

// ================================================================================
// VECTOR3D STRUCTS (INTRINSICS)
// ================================================================================

struct alignas(16) Vector3DStackAligned {
    float x, y, z, w; // Total 16 bytes
};

// This represents 4 vectors at once
struct Vector3D_SOA_Batch {
    __m128 x; // [v1.x, v2.x, v3.x, v4.x]
    __m128 y; // [v1.y, v2.y, v3.y, v4.y]
    __m128 z; // [v1.z, v2.z, v3.z, v4.z]
};
