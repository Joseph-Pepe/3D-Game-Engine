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

// Exactly 32 Bytes. Fits perfectly into half a CPU Cache Line.
struct BVHNode {
    float minX, minY, minZ;
    uint32_t leftFirst;     // If internal: index of Left Child. If leaf: index of first primitive.
    
    float maxX, maxY, maxZ;
    uint32_t primitiveCount;// If 0, this is an internal node. If > 0, this is a leaf node.

    FORCE_INLINE bool IsLeaf() const { return primitiveCount > 0; }
};

// Represents a mathematical Ray for intersection testing
struct Ray {
    Vector3D origin;
    Vector3D direction;
    Vector3D invDirection; // 1.0f / direction (Pre-calculated to prevent slow divisions)
    
    Ray(const Vector3D& o, const Vector3D& d) {
        origin = o.asPoint();         // w = 1.0
        direction = d.asDirection();  // w = 0.0
        
        // Prevent division by zero using a tiny epsilon
        __m128 epsilon = _mm_set1_ps(1e-8f);
        __m128 dirReg = direction.reg;
        
        // Replace 0.0 with epsilon before division
        __m128 mask = _mm_cmp_ps(dirReg, _mm_setzero_ps(), _CMP_EQ_OQ);
        dirReg = _mm_blendv_ps(dirReg, epsilon, mask);
        
        invDirection = Vector3D(_mm_div_ps(_mm_set1_ps(1.0f), dirReg)).asDirection();
    }
};

class LinearBVH {
private:
    std::vector<BVHNode> m_nodes;
    std::vector<uint32_t> m_primitiveIndices; // Sorted indices of triangles/static meshes
    uint32_t m_rootNodeIndex = 0;

public:
    // Builds the BVH (Typically done once on level load, or offline during asset baking)
    // You would implement a Surface Area Heuristic (SAH) or Median Split builder here.
    // void Build(std::span<Triangle> geometry) { ... }

    // --- SIMD RAY-AABB INTERSECTION (THE SLAB METHOD) ---
    // Tests a ray against a bounding box completely branchlessly.
    FORCE_INLINE bool IntersectNode(const BVHNode& node, const Ray& ray, float hitDistanceLimit) const {
        
        // Load the Box Min and Max into SIMD registers
        __m128 boxMin = _mm_set_ps(0.0f, node.minZ, node.minY, node.minX);
        __m128 boxMax = _mm_set_ps(0.0f, node.maxZ, node.maxY, node.maxX);

        // Math: t = (Box - RayOrigin) * InvRayDirection
        __m128 t1 = _mm_mul_ps(_mm_sub_ps(boxMin, ray.origin.reg), ray.invDirection.reg);
        __m128 t2 = _mm_mul_ps(_mm_sub_ps(boxMax, ray.origin.reg), ray.invDirection.reg);

        // We want the minimum and maximum distances along the ray
        __m128 tMin = _mm_min_ps(t1, t2);
        __m128 tMax = _mm_max_ps(t1, t2);

        // Find the absolute highest tMin and lowest tMax across X, Y, Z
        // We use our trusty manual SIMD horizontal shuffle reduction!

        // --- Calculate tNear (Horizontal Max of X, Y, Z) ---
        
        // 1. Move Z into the X lane and compare
        __m128 max1 = _mm_max_ps(tMin, _mm_shuffle_ps(tMin, tMin, _MM_SHUFFLE(0, 0, 3, 2))); 
        // 2. Move Y into the X lane and compare against the previous result
        __m128 tNearSIMD = _mm_max_ps(max1, _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(0, 0, 0, 1))); 
        // 3. Extract the final scalar float from the X lane (0th index)
        float tNear = _mm_cvtss_f32(tNearSIMD);

        // --- Calculate tFar (Horizontal Min of X, Y, Z) ---

        // 1. Move Z into the X lane and compare
        __m128 min1 = _mm_min_ps(tMax, _mm_shuffle_ps(tMax, tMax, _MM_SHUFFLE(0, 0, 3, 2))); 
        // 2. Move Y into the X lane and compare against the previous result
        __m128 tFarSIMD  = _mm_min_ps(min1, _mm_shuffle_ps(min1, min1, _MM_SHUFFLE(0, 0, 0, 1))); 
        // 3. Extract the final scalar float from the X lane (0th index)
        float tFar = _mm_cvtss_f32(tFarSIMD);

        // If tNear <= tFar, the ray hit the box. 
        // We also check if the hit is physically in front of us (tFar > 0) 
        // and closer than our maximum cast limit.
        return tNear <= tFar && tFar > 0.0f && tNear < hitDistanceLimit;
    }

    // --- BVH TRAVERSAL (STACK-BASED) ---
    // We do NOT use recursion. Recursion destroys the call stack and instruction cache.
    // We use a fixed-size local array to simulate a stack.
    bool Raycast(const Ray& ray, float maxDistance) const {
        if (m_nodes.empty()) return false;

        uint32_t stack[64]; // Max depth of 64 is enough to hold 18 quintillion nodes
        uint32_t stackPtr = 0;
        
        stack[stackPtr++] = m_rootNodeIndex;

        while (stackPtr > 0) {
            // Pop the top node off the stack
            uint32_t nodeIndex = stack[--stackPtr];
            const BVHNode& node = m_nodes[nodeIndex];

            // SIMD Intersection Test
            if (IntersectNode(node, ray, maxDistance)) {
                
                if (node.IsLeaf()) {
                    // --- NARROW PHASE ---
                    // We hit a leaf! Now you test the ray against the actual triangles 
                    // stored in m_primitiveIndices[node.leftFirst]...
                    // If a hit occurs, return true or record the impact data.
                    return true; 
                } else {
                    // --- INTERNAL NODE ---
                    // Push children to the stack. 
                    // OPTIMIZATION: Push the right child first so the left child is popped first 
                    // (Assuming left children are usually closer).
                    stack[stackPtr++] = node.leftFirst + 1; // Right Child
                    stack[stackPtr++] = node.leftFirst;     // Left Child
                }
            }
        }
        return false;
    }
};
