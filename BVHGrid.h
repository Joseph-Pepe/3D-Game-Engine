#pragma once
#include "Math.h"
#include <span>
#include <vector>

// ==================================================================================
// LINEAR BOUNDING VOLUME HIERARCHY (RAYTRACING & STATIC LEVEL GEOMETRY)
// ==================================================================================
/*
    - When loading a massive city into the level, we calculate the BVHNode bounds relative to the center of the city block using 32-bit floats.
    - LinearBVH is pre-baked and saved to the hard drive.

    1. Player fires a bullet.
    2. Take player's LWC 64-bit Vector3DWorld position and bullet position.
    3. Subtract the city block's LWC 64-bit position from the player's position.
    4. Cast the resulting difference to 32-bit Vector3D.
    5. Run the bvh.Raycast(localRay) using pure SIMD.
*/

// alignas(128): forces the struct to sit perfectly inside exactly two 64-byte L1 CPU cache lines without straddling boundaries.
struct alignas(128) BVH4Node {
    // We use anonymous unions. This costs zero extra memory but allows us to access
    // the registers as arrays when building the BVH, and as SIMD during the raycast.
    union { __m128 minX; float minX_f[4]; };
    union { __m128 minY; float minY_f[4]; };
    union { __m128 minZ; float minZ_f[4]; };
    
    union { __m128 maxX; float maxX_f[4]; };
    union { __m128 maxY; float maxY_f[4]; };
    union { __m128 maxZ; float maxZ_f[4]; };

    // 4x 4-byte integers = 16 Bytes
    // If MSB (Most Significant Bit) is 1, it's a leaf. Lower 31 bits are the index.
    uint32_t children[4]; 

    // 16 Bytes of padding to hit exactly 128 Bytes
    uint32_t padding[4]; 
    
    FORCE_INLINE bool IsLeaf(int childIndex) const { 
        return (children[childIndex] & 0x80000000) != 0; 
    }
    
    FORCE_INLINE uint32_t GetIndex(int childIndex) const { 
        return children[childIndex] & 0x7FFFFFFF; 
    }

    // Gotcha 1 FIXED: Call this inside your BVH Builder for any unused lanes
    void SetDegenerateLane(int laneIndex) {
        constexpr float INF = std::numeric_limits<float>::infinity();
        
        // Setting Min to +INF and Max to -INF guarantees tNear <= tFar will fail
        minX_f[laneIndex] = INF;
        minY_f[laneIndex] = INF;
        minZ_f[laneIndex] = INF;

        maxX_f[laneIndex] = -INF;
        maxY_f[laneIndex] = -INF;
        maxZ_f[laneIndex] = -INF;

        // Nullify the child pointer just to be safe
        children[laneIndex] = 0;
    }
};

// Represents a mathematical Ray for intersection testing
struct Ray4 {
    // Each register holds four identical copies of the X, Y, or Z component
    __m128 origX, origY, origZ;
    __m128 invDirX, invDirY, invDirZ;

    Ray4(const Vector3D& o, const Vector3D& d) {
        // Broadcast a single float to all 4 lanes
        origX = _mm_set1_ps(o.x);
        origY = _mm_set1_ps(o.y);
        origZ = _mm_set1_ps(o.z);

        // Pre-calculate inverse direction
        const float epsilon = 1e-8f;
        Vector3D invD(
            1.0f / (std::abs(d.x) < epsilon ? epsilon : d.x),
            1.0f / (std::abs(d.y) < epsilon ? epsilon : d.y),
            1.0f / (std::abs(d.z) < epsilon ? epsilon : d.z)
        );
        invDirX = _mm_set1_ps(invD.x);
        invDirY = _mm_set1_ps(invD.y);
        invDirZ = _mm_set1_ps(invD.z);
    }
};

class LinearBVH {
private:
    std::vector<BVH4Node> m_nodes;
    std::vector<uint32_t> m_primitiveIndices; // Sorted indices of triangles/static meshes
    uint32_t m_rootNodeIndex = 0;

public:
    // Builds the BVH (Typically done once on level load, or offline during asset baking)
    // You would implement a Surface Area Heuristic (SAH) or Median Split builder here.
    // void Build(std::span<Triangle> geometry) { ... }

    // --- SIMD RAY-AABB INTERSECTION (DISTANCE RETURNING) ---
    // Returns a 4-bit mask (0 to 15). 
    // Bit 0 = Child 0 hit. Bit 1 = Child 1 hit, etc.
    FORCE_INLINE int IntersectNode4(const BVH4Node& node, const Ray4& ray, float maxDist) const {
        
        // --- X Axis ---
        __m128 t1x = _mm_mul_ps(_mm_sub_ps(node.minX, ray.origX), ray.invDirX);
        __m128 t2x = _mm_mul_ps(_mm_sub_ps(node.maxX, ray.origX), ray.invDirX);
        __m128 tNear = _mm_min_ps(t1x, t2x);
        __m128 tFar  = _mm_max_ps(t1x, t2x);

        // --- Y Axis ---
        __m128 t1y = _mm_mul_ps(_mm_sub_ps(node.minY, ray.origY), ray.invDirY);
        __m128 t2y = _mm_mul_ps(_mm_sub_ps(node.maxY, ray.origY), ray.invDirY);
        tNear = _mm_max_ps(tNear, _mm_min_ps(t1y, t2y));
        tFar  = _mm_min_ps(tFar,  _mm_max_ps(t1y, t2y));

        // --- Z Axis ---
        __m128 t1z = _mm_mul_ps(_mm_sub_ps(node.minZ, ray.origZ), ray.invDirZ);
        __m128 t2z = _mm_mul_ps(_mm_sub_ps(node.maxZ, ray.origZ), ray.invDirZ);
        tNear = _mm_max_ps(tNear, _mm_min_ps(t1z, t2z));
        tFar  = _mm_min_ps(tFar,  _mm_max_ps(t1z, t2z));

        // --- Logical Checks ---
        // tNear <= tFar AND tFar > 0.0f AND tNear < maxDist
        __m128 hitMask = _mm_cmple_ps(tNear, tFar);
        hitMask = _mm_and_ps(hitMask, _mm_cmpgt_ps(tFar, _mm_setzero_ps()));
        hitMask = _mm_and_ps(hitMask, _mm_cmplt_ps(tNear, _mm_set1_ps(maxDist)));

        // Extract the highest bit from each lane to create a 4-bit integer
        return _mm_movemask_ps(hitMask);
    }

    // --- BVH TRAVERSAL (STACK-BASED) ---
    // We do NOT use recursion. Recursion destroys the call stack and instruction cache.
    // We use a fixed-size local array to simulate a stack.
    bool Raycast(const Ray4& ray, float maxDistance) const {
        if (m_nodes.empty()) return false;

        // Stack 128 to handle BVH4 worst-case growth
        uint32_t stack[128];
        uint32_t stackPtr = 0;
        
        // Push the root node to begin
        stack[stackPtr++] = m_rootNodeIndex;
        bool hitAnything = false;

        while (stackPtr > 0) {
            // Safety rail to prevent silent memory corruption during development
            assert(stackPtr < 128 && "BVH Traversal Stack Overflow! Tree is dangerously unbalanced.");
            
            // Pop the top node
            uint32_t nodeIndex = stack[--stackPtr];
            const BVH4Node& node = m_nodes[nodeIndex];

            // Get the 4-bit mask of which of the 4 children were hit
            int hitMask = IntersectNode4(node, ray, maxDistance);

            // Loop as long as there is at least one bit set to 1 in the mask
            while (hitMask != 0) {
                
                // Hardware instruction to find the index of the lowest set bit (0, 1, 2, or 3)
                #ifdef _MSC_VER
                    unsigned long childIdx;
                    _BitScanForward(&childIdx, hitMask);
                #else
                    int childIdx = __builtin_ctz(hitMask);
                #endif

                // Clear that bit so we don't process it again on the next loop
                hitMask &= ~(1 << childIdx);

                if (node.IsLeaf(childIdx)) {
                    // --- NARROW PHASE ---
                    // You hit a leaf! Test the triangles using: node.GetIndex(childIdx)
                    // If you find a closer triangle:
                    // 1. Record hit data
                    // 2. maxDistance = new_triangle_t;
                    // 3. hitAnything = true;
                } else {
                    // --- INTERNAL NODE ---
                    // Push the valid child node to the stack
                    stack[stackPtr++] = node.GetIndex(childIdx);
                }
            }
        }
        
        return hitAnything;
    }
};
