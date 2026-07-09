#pragma once
#include "Math.h" 
#include "Math/SSE-128/MathSSE128.h"
#include <span>
#include <vector>
#include <cstdint>

#include <algorithm> // Required for std::partition and std::min
#include <limits>    // Required for std::numeric_limits<float>::infinity()
#include <cmath>     // Required for std::abs()
#include <immintrin.h> // Required for AVX/SSE intrinsics
#include <cassert>   // Required for the assert() macro


// ==================================================================================
// LINEAR BOUNDING VOLUME HIERARCHY (RAYTRACING & STATIC LEVEL GEOMETRY)
// ==================================================================================
/*
    - When loading a massive city into the level, we calculate the BVHNode bounds relative to the center of the city block using 32-bit floats.
    - LinearBVH is pre-baked and saved to the hard drive.
    - 3D raytracing and physics collision engine.

    1. Player fires a bullet.
    2. Take player's LWC 64-bit Vector3DWorld position and bullet position.
    3. Subtract the city block's LWC 64-bit position from the player's position.
    4. Cast the resulting difference to 32-bit SIMDVector3D.
    5. Run the bvh.Raycast(localRay) using pure SIMD.
*/

constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;

// alignas(128): forces the struct to sit perfectly inside exactly two 64-byte L1 CPU cache lines without straddling boundaries.
struct alignas(128) BVH4Node {
    // We use anonymous unions. This costs zero extra memory but allows us to access
    // the registers as arrays when building the BVH, and as SIMD during the raycast.
    alignas(16) float minX[4];
    alignas(16) float minY[4];
    alignas(16) float minZ[4];
    
    alignas(16) float maxX[4];
    alignas(16) float maxY[4];
    alignas(16) float maxZ[4];

    // 4x 4-byte integers = 16 Bytes
    // If MSB (Most Significant Bit) is 1, it's a leaf. Lower 31 bits are the index.
    uint32_t children[4]; 

    // 16 Bytes of padding to hit exactly 128 Bytes
    uint32_t primCounts[4]; 
    
    FORCE_INLINE bool IsLeaf(int childIndex) const { 
        return (children[childIndex] & 0x80000000) != 0; 
    }
    
    FORCE_INLINE uint32_t GetIndex(int childIndex) const { 
        return children[childIndex] & 0x7FFFFFFF; 
    }

    // Call this inside your BVH Builder for any unused lanes
    void SetDegenerateLane(int laneIndex) {
        constexpr float INF = std::numeric_limits<float>::infinity();
        
        minX[laneIndex] = INF; minY[laneIndex] = INF; minZ[laneIndex] = INF;
        maxX[laneIndex] = -INF; maxY[laneIndex] = -INF; maxZ[laneIndex] = -INF;

        // Nullify the child pointer just to be safe
        children[laneIndex] = 0;
    }
};

// Exactly 32 bytes (half a cache line). Built for extreme traversal speed.
struct alignas(32) TLASNode {
    float minX, minY, minZ;
    uint32_t leftFirst; // If MSB is 1, this is a leaf and holds the instanceIndex. Otherwise, Left Child.
    
    float maxX, maxY, maxZ;
    uint32_t padding;

    FORCE_INLINE bool IsLeaf() const { return (leftFirst & 0x80000000) != 0; }
    FORCE_INLINE uint32_t GetIndex() const { return leftFirst & 0x7FFFFFFF; }
};

struct Ray {
    SIMDVector3D origin;
    SIMDVector3D direction;
    SIMDVector3D invDirection; 
    
    Ray(const SIMDVector3D& o, const SIMDVector3D& d) : origin(o), direction(d) {
        // Prevents division by zero.
        const float epsilon = 1e-8f;
        invDirection = SIMDVector3D(
            1.0f / (std::abs(d.x()) < epsilon ? epsilon : d.x()),
            1.0f / (std::abs(d.y()) < epsilon ? epsilon : d.y()),
            1.0f / (std::abs(d.z()) < epsilon ? epsilon : d.z())
        );
    }
};

// Represents a mathematical Ray for intersection testing
struct Ray4 {
    // Each register holds four identical copies of the X, Y, or Z component
    __m128 origX, origY, origZ;
    __m128 invDirX, invDirY, invDirZ;
    __m128 dirX, dirY, dirZ;

    Ray4(const SIMDVector3D& o, const SIMDVector3D& d) {
        // Broadcast a single float to all 4 lanes
        origX = _mm_set1_ps(o.x());
        origY = _mm_set1_ps(o.y());
        origZ = _mm_set1_ps(o.z());

        dirX = _mm_set1_ps(d.x());
        dirY = _mm_set1_ps(d.y());
        dirZ = _mm_set1_ps(d.z());

        // Pre-calculate inverse direction
        const float epsilon = 1e-8f;
        SIMDVector3D invD(
            1.0f / (std::abs(d.x()) < epsilon ? epsilon : d.x()),
            1.0f / (std::abs(d.y()) < epsilon ? epsilon : d.y()),
            1.0f / (std::abs(d.z()) < epsilon ? epsilon : d.z())
        );
        invDirX = _mm_set1_ps(invD.x());
        invDirY = _mm_set1_ps(invD.y());
        invDirZ = _mm_set1_ps(invD.z());
    }
};

// The ultimate payload that gets passed back to gameplay/rendering
struct RayHit {
    float t = std::numeric_limits<float>::infinity();
    float u = 0.0f;
    float v = 0.0f;
    uint32_t instanceIndex = INVALID_INDEX; // Which car did we hit?
    uint32_t triangleIndex = INVALID_INDEX; // Which triangle on the car?

    FORCE_INLINE bool HasHit() const { return instanceIndex != INVALID_INDEX; }
};

// ==================================================================================
// SURFACE AREA HEURITSIC (SAH)
// ==================================================================================
/*  
    - Is an algorithm used to feed the 3D static geometry for ray casting.
    - Is the probability that a ray will hit (or collide) with a box. 
    - When a player fires a bullet into empty space, the bullet hits a box, traverse the tree to see what was hit.
    - Calculates the surface area of the bounding box to estimate the statistical likelihood that the ray hit it.
    - i.e., collision detection (physics), and raytracing (rendering / line of sight).
*/

// A standard 3D Bounding Box for the builder
struct AABB {
    SIMDVector3D bmin = SIMDVector3D( 1e30f,  1e30f,  1e30f);
    SIMDVector3D bmax = SIMDVector3D(-1e30f, -1e30f, -1e30f);

    void Grow(const SIMDVector3D& p) {
        // Direct SIMD min/max evaluation (1 clock cycle)
        bmin = SIMDVector3D(_mm_min_ps(bmin.reg, p.reg));
        bmax = SIMDVector3D(_mm_max_ps(bmax.reg, p.reg));
    }

    void Grow(const AABB& b) {
        bmin = SIMDVector3D(_mm_min_ps(bmin.reg, b.bmin.reg));
        bmax = SIMDVector3D(_mm_max_ps(bmax.reg, b.bmax.reg));
    }

    float Area() const {
        SIMDVector3D e = bmax - bmin; // Extents
        return e.x() * e.y() + e.y() * e.z() + e.z() * e.x();
    }
};

// Represents your raw level geometry
struct Triangle {
    SIMDVector3D v0, v1, v2;
    SIMDVector3D centroid; // Pre-calculated (v0+v1+v2)/3 for fast binning
    AABB bounds;       // Pre-calculated AABB of this specific triangle
};

// 192-bytes (three 64-byte chunks) that stores 4 triangles ready for pure parallel execution.
// alignas(64) ensures it perfectly snaps into 3 CPU cache lines
struct alignas(64) Tri4 {
    // Vertex 0 (X, Y, Z for 4 triangles)
    alignas(16) float v0x[4]; 
    alignas(16) float v0y[4]; 
    alignas(16) float v0z[4];
    
    // Pre-calculated Edge 1 (v1 - v0)
    alignas(16) float e1x[4]; 
    alignas(16) float e1y[4]; 
    alignas(16) float e1z[4];
    
    // Pre-calculated Edge 2 (v2 - v0)
    alignas(16) float e2x[4]; 
    alignas(16) float e2y[4]; 
    alignas(16) float e2z[4];

    alignas(16) uint32_t indices[4];
    alignas(16) uint32_t materialIDs[4]; 
    alignas(16) uint32_t padding[4]; // 16 bytes of padding hits exactly 192!

    // Phase 1/Phase 2 builder to populate the lanes
    void SetTriangle(int lane, const Triangle& tri, uint32_t index) {
        v0x[lane] = tri.v0.x();
        v0y[lane] = tri.v0.y();
        v0z[lane] = tri.v0.z();

        e1x[lane] = tri.v1.x() - tri.v0.x();
        e1y[lane] = tri.v1.y() - tri.v0.y();
        e1z[lane] = tri.v1.z() - tri.v0.z();

        e2x[lane] = tri.v2.x() - tri.v0.x();
        e2y[lane] = tri.v2.y() - tri.v0.y();
        e2z[lane] = tri.v2.z() - tri.v0.z();

        indices[lane] = index;
    }
};

// A struct to return the best hit out of the 4
struct Tri4Hit {
    float t;
    float u, v;
    uint32_t hitIndex = INVALID_INDEX; // Will be -1 (or max uint32) if no hit
};

// -- SIMD MOLLER TRUMBORE -- 4 complex 3D triangle intersections simultaneously.
FORCE_INLINE Tri4Hit IntersectTri4_MT(const Tri4& tri4, const Ray4& ray, float maxDist) {
    const __m128 epsilon = _mm_set1_ps(1e-8f);
    const __m128 zero = _mm_setzero_ps();

    // --- LOAD SIMD REGISTERS FROM STRUCT ---
    __m128 v0x = _mm_load_ps(tri4.v0x);
    __m128 v0y = _mm_load_ps(tri4.v0y);
    __m128 v0z = _mm_load_ps(tri4.v0z);

    __m128 e1x = _mm_load_ps(tri4.e1x);
    __m128 e1y = _mm_load_ps(tri4.e1y);
    __m128 e1z = _mm_load_ps(tri4.e1z);

    __m128 e2x = _mm_load_ps(tri4.e2x);
    __m128 e2y = _mm_load_ps(tri4.e2y);
    __m128 e2z = _mm_load_ps(tri4.e2z);

    // 1. pvec = cross(ray.dir, tri4.e2)
    __m128 pvecX = _mm_sub_ps(_mm_mul_ps(ray.dirY, e2z), _mm_mul_ps(ray.dirZ, e2y));
    __m128 pvecY = _mm_sub_ps(_mm_mul_ps(ray.dirZ, e2x), _mm_mul_ps(ray.dirX, e2z));
    __m128 pvecZ = _mm_sub_ps(_mm_mul_ps(ray.dirX, e2y), _mm_mul_ps(ray.dirY, e2x));

    // 2. det = dot(tri4.e1, pvec)
    __m128 det = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(e1x, pvecX), _mm_mul_ps(e1y, pvecY)),
        _mm_mul_ps(e1z, pvecZ)
    );

    // Backface Culling Mask (det > epsilon)
    __m128 validMask = _mm_cmpgt_ps(det, epsilon);
    
    // EARLY EXIT: If all 4 triangles are backfacing or parallel, abort immediately!
    if (_mm_movemask_ps(validMask) == 0) return { maxDist, 0, 0, INVALID_INDEX };

    // 3. tvec = ray.orig - tri4.v0
    __m128 tvecX = _mm_sub_ps(ray.origX, v0x);
    __m128 tvecY = _mm_sub_ps(ray.origY, v0y);
    __m128 tvecZ = _mm_sub_ps(ray.origZ, v0z);

    // 4. U = dot(tvec, pvec)
    __m128 U = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(tvecX, pvecX), _mm_mul_ps(tvecY, pvecY)),
        _mm_mul_ps(tvecZ, pvecZ)
    );

    // Cull U: (U >= 0) AND (U <= det)
    validMask = _mm_and_ps(validMask, _mm_cmpge_ps(U, zero));
    validMask = _mm_and_ps(validMask, _mm_cmple_ps(U, det));
    if (_mm_movemask_ps(validMask) == 0) return { maxDist, 0, 0, (uint32_t)-1 };

    // 5. qvec = cross(tvec, tri4.e1)
    __m128 qvecX = _mm_sub_ps(_mm_mul_ps(tvecY, e1z), _mm_mul_ps(tvecZ, e1y));
    __m128 qvecY = _mm_sub_ps(_mm_mul_ps(tvecZ, e1x), _mm_mul_ps(tvecX, e1z));
    __m128 qvecZ = _mm_sub_ps(_mm_mul_ps(tvecX, e1y), _mm_mul_ps(tvecY, e1x));

    // 6. V = dot(ray.dir, qvec)
    __m128 V = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(ray.dirX, qvecX), _mm_mul_ps(ray.dirY, qvecY)),
        _mm_mul_ps(ray.dirZ, qvecZ)
    );

    // Cull V: (V >= 0) AND (U + V <= det)
    validMask = _mm_and_ps(validMask, _mm_cmpge_ps(V, zero));
    validMask = _mm_and_ps(validMask, _mm_cmple_ps(_mm_add_ps(U, V), det));
    if (_mm_movemask_ps(validMask) == 0) return { maxDist, 0, 0, (uint32_t)-1 };

    // --- NEWTON-RAPHSON RECIPROCAL (The Division Killer) ---
    // Instead of _mm_div_ps (20 cycles), we use a hardware approximation (1 cycle) 
    // and refine it with one step of Newton-Raphson math.
    __m128 invDet = _mm_rcp_ps(det);
    invDet = _mm_mul_ps(invDet, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(det, invDet)));

    // 7. T = dot(tri4.e2, qvec) * invDet
    __m128 unscaledT = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(e2x, qvecX), _mm_mul_ps(e2y, qvecY)),
        _mm_mul_ps(e2z, qvecZ)
    );
    __m128 T = _mm_mul_ps(unscaledT, invDet);

    // Cull T: (T > epsilon) AND (T < maxDist)
    __m128 maxDistVec = _mm_set1_ps(maxDist);
    validMask = _mm_and_ps(validMask, _mm_cmpgt_ps(T, epsilon));
    validMask = _mm_and_ps(validMask, _mm_cmplt_ps(T, maxDistVec));

    int hitMask = _mm_movemask_ps(validMask);
    if (hitMask == 0) return { maxDist, 0, 0, (uint32_t)-1 };

    // --- EXTRACT THE CLOSEST HIT ---
    // If multiple triangles hit, we must find the closest one.
    Tri4Hit bestHit = { maxDist, 0, 0, (uint32_t)-1 };

    // Dump the SIMD registers into aligned cache memory once
    alignas(16) float finalT[4];
    alignas(16) float finalU[4];
    alignas(16) float finalV[4];
    alignas(16) float finalInvDet[4];

    _mm_store_ps(finalT, T);
    _mm_store_ps(finalU, U);
    _mm_store_ps(finalV, V);
    _mm_store_ps(finalInvDet, invDet);
    
    // We only loop over the bits that are actually set to 1 in the mask
    while (hitMask != 0) {
        uint32_t lane;
        #ifdef _MSC_VER
            unsigned long msLane;
            _BitScanForward(&msLane, hitMask);
            lane = msLane;
        #else
            lane = __builtin_ctz(hitMask);
        #endif
        
        hitMask &= ~(1 << lane); // Clear the bit

        // Read directly from the aligned arrays
        if (finalT[lane] < bestHit.t) {
            bestHit.t = finalT[lane];
            bestHit.u = finalU[lane] * finalInvDet[lane];
            bestHit.v = finalV[lane] * finalInvDet[lane];
            bestHit.hitIndex = tri4.indices[lane];
        }
    }

    return bestHit;
}

struct HitResult {
    float t;     // Distance along the ray
    float u, v;  // Barycentric coordinates for texture mapping
    bool hit;    // Did we hit?
};

FORCE_INLINE HitResult IntersectTriangle_MT(const SIMDVector3D& rayOrigin, const SIMDVector3D& rayDir, const SIMDVector3D& v0, const SIMDVector3D& v1, const SIMDVector3D& v2, bool cullBackfaces = true) {
    HitResult result = { 0.0f, 0.0f, 0.0f, false };

    // Find vectors for two edges sharing v0
    SIMDVector3D e1 = v1 - v0;
    SIMDVector3D e2 = v2 - v0;

    // Begin calculating determinant - also used to calculate U parameter
    SIMDVector3D pvec = rayDir.cross(e2);
    
    // If determinant is near zero, ray lies in plane of triangle
    float det = pvec.dot(e1);
    constexpr float EPSILON = 1e-8f;

    // --- OPTIMIZATION 1: BACKFACE CULLING ---
    if (cullBackfaces) {
        // If det is negative, the ray hit the back of the polygon.
        // If det is < EPSILON, it's parallel or a backface. Instantly reject.
        if (det < EPSILON) return result;
        
        // Calculate distance from v0 to ray origin
        SIMDVector3D tvec = rayOrigin - v0;
        
        // Calculate U parameter and test bounds (Notice we compare against 'det', not 1.0f!)
        result.u = tvec.dot(pvec);
        if (result.u < 0.0f || result.u > det) return result;
        
        // Prepare to test V parameter
        SIMDVector3D qvec = tvec.cross(e1);
        
        // Calculate V parameter and test bounds
        result.v = qvec.dot(rayDir);
        if (result.v < 0.0f || result.u + result.v > det) return result;
        
        // Calculate T (distance)
        result.t = qvec.dot(e2);

    } else {
        // --- TWO-SIDED GEOMETRY (e.g., foliage, glass) ---
        if (det > -EPSILON && det < EPSILON) return result;
        
        float invDet_early = 1.0f / det; // Have to divide early for two-sided math
        
        SIMDVector3D tvec = rayOrigin - v0;
        result.u = tvec.dot(pvec) * invDet_early;
        if (result.u < 0.0f || result.u > 1.0f) return result;
        
        SIMDVector3D qvec = tvec.cross(e1);
        result.v = qvec.dot(rayDir) * invDet_early;
        if (result.v < 0.0f || result.u + result.v > 1.0f) return result;
        
        result.t = qvec.dot(e2) * invDet_early;
        if (result.t < 0.0f) return result;
        
        result.hit = true;
        return result;
    }

    // --- OPTIMIZATION 2: DEFERRED DIVISION ---
    // If we made it here (Backface culling pathway), the ray definitely hit!
    // Now, and ONLY now, do we pay the 20-cycle cost for the division.
    float invDet = 1.0f / det;
    
    result.t *= invDet;
    result.u *= invDet;
    result.v *= invDet;
    
    // Ensure the hit is strictly in front of the camera
    result.hit = result.t > EPSILON;
    
    return result;
}

// =========================================================================
// PHASE 1: TEMPORARY BINARY NODE (SORT MEMORY)
// =========================================================================
struct BVHNode_Binary {
    AABB bounds;
    uint32_t leftFirst; 
    uint32_t primCount;

    bool IsLeaf() const { return primCount > 0; }
};

class BVHBuilder {
private:
    std::vector<BVHNode_Binary> m_binaryNodes;
    std::vector<uint32_t> m_indices;
    std::span<const Triangle> m_geometry;
    uint32_t m_nodesUsed = 0;

    // Constants for tuning the builder
    static constexpr int BINS = 16; 
    static constexpr float SAH_INTERSECT_COST = 1.0f; // Cost of ray-triangle test
    static constexpr float SAH_TRAVERSAL_COST = 1.2f; // Cost of ray-AABB test

public:
    // Entry point
    void Build(std::span<const Triangle> geometry) {
        m_geometry = geometry;
        uint32_t N = geometry.size();

        // Optimization: Pre-allocate maximum possible nodes (2N - 1) 
        // to prevent dynamic array resizing during the recursive build.
        m_binaryNodes.resize(N * 2);
        
        // Populate initial index array (0, 1, 2, 3...)
        m_indices.resize(N);
        for (uint32_t i = 0; i < N; i++) m_indices[i] = i;

        // Initialize the Root Node
        BVHNode_Binary& root = m_binaryNodes[0];
        root.leftFirst = 0;
        root.primCount = N;
        m_nodesUsed = 1;

        UpdateNodeBounds(0);
        Subdivide(0);
    }

    const std::vector<BVHNode_Binary>& GetBinaryNodes() const { return m_binaryNodes; }
    const std::vector<uint32_t>& GetIndices() const { return m_indices; }

private:
    // Calculates the tightest possible box around the triangles in this node
    void UpdateNodeBounds(uint32_t nodeIdx) {
        BVHNode_Binary& node = m_binaryNodes[nodeIdx];
        node.bounds = AABB();

        for (uint32_t i = 0; i < node.primCount; i++) {
            uint32_t leafPrimIdx = m_indices[node.leftFirst + i];
            node.bounds.Grow(m_geometry[leafPrimIdx].bounds);
        }
    }

    // The core Binned SAH algorithm
    void Subdivide(uint32_t nodeIdx) {
        BVHNode_Binary& node = m_binaryNodes[nodeIdx];

        // Stop splitting if we have 2 or fewer triangles (Leaf Node)
        if (node.primCount <= 2) return;

        // Find the best split plane across all 3 axes using 16 Bins
        float bestCost = 1e30f;
        int bestAxis = -1;
        int bestSplitBin = -1;

        // Struct to hold bin data
        struct Bin { AABB bounds; int primCount = 0; };

        // Test X, Y, and Z axes
        for (int axis = 0; axis < 3; axis++) {
            Bin bins[BINS];
            float boundsMin = node.bounds.bmin[axis];
            float boundsMax = node.bounds.bmax[axis];
            
            // If the box is completely flat on this axis, skip it
            if (boundsMax - boundsMin < 1e-5f) continue;

            // 1. Populate the Bins
            float scale = BINS / (boundsMax - boundsMin);
            for (uint32_t i = 0; i < node.primCount; i++) {
                const Triangle& tri = m_geometry[m_indices[node.leftFirst + i]];
                float centroidPos = tri.centroid[axis];
                
                int binIdx = std::min(BINS - 1, static_cast<int>((centroidPos - boundsMin) * scale));
                bins[binIdx].primCount++;
                bins[binIdx].bounds.Grow(tri.bounds);
            }

            // 2. Evaluate Split Planes (Sweep Left to Right)
            float leftArea[BINS - 1];
            int leftCount[BINS - 1];
            AABB leftBox;
            int leftSum = 0;

            for (int i = 0; i < BINS - 1; i++) {
                leftSum += bins[i].primCount;
                leftCount[i] = leftSum;
                leftBox.Grow(bins[i].bounds);
                leftArea[i] = leftBox.Area();
            }

            // 3. Evaluate Split Planes (Sweep Right to Left) & Calculate Cost
            AABB rightBox;
            int rightSum = 0;
            
            for (int i = BINS - 1; i > 0; i--) {
                rightSum += bins[i].primCount;
                rightBox.Grow(bins[i].bounds);
                float rightArea = rightBox.Area();

                // SAH Cost Formula: 
                // Cost = TraversalCost + (AreaLeft * CountLeft + AreaRight * CountRight) / AreaParent
                float planeCost = SAH_TRAVERSAL_COST + 
                    (leftArea[i - 1] * leftCount[i - 1] + rightArea * rightSum) / node.bounds.Area();

                if (planeCost < bestCost) {
                    bestCost = planeCost;
                    bestAxis = axis;
                    bestSplitBin = i;
                }
            }
        }

        // Cost of doing nothing (making this node a leaf)
        float leafCost = node.primCount * SAH_INTERSECT_COST;

        // If splitting is more expensive than just testing the triangles, stop!
        if (bestCost >= leafCost) return;

        // --- THE IN-PLACE PARTITION ---
        // Swap integers around so all left-bin indices are on the left, right-bin on the right.
        float boundsMin = node.bounds.bmin[bestAxis];
        float boundsMax = node.bounds.bmax[bestAxis];
        float scale = BINS / (boundsMax - boundsMin);

        auto firstIt = m_indices.begin() + node.leftFirst;
        auto lastIt = firstIt + node.primCount;

        // std::partition is insanely fast for this
        auto splitIt = std::partition(firstIt, lastIt, [&](uint32_t idx) {
            const Triangle& tri = m_geometry[idx];
            float centroidPos = tri.centroid[bestAxis];
            int binIdx = std::min(BINS - 1, static_cast<int>((centroidPos - boundsMin) * scale));
            return binIdx < bestSplitBin;
        });

        int leftCount = std::distance(firstIt, splitIt);
        
        // Edge case: If the partition failed to split anything, force a leaf
        if (leftCount == 0 || leftCount == node.primCount) return;

        // Create the children
        uint32_t leftChildIdx = m_nodesUsed++;
        uint32_t rightChildIdx = m_nodesUsed++;

        m_binaryNodes[leftChildIdx].leftFirst = node.leftFirst;
        m_binaryNodes[leftChildIdx].primCount = leftCount;

        m_binaryNodes[rightChildIdx].leftFirst = node.leftFirst + leftCount;
        m_binaryNodes[rightChildIdx].primCount = node.primCount - leftCount;

        // Convert the parent node into an internal node
        node.leftFirst = leftChildIdx; 
        node.primCount = 0; 

        UpdateNodeBounds(leftChildIdx);
        UpdateNodeBounds(rightChildIdx);

        Subdivide(leftChildIdx);
        Subdivide(rightChildIdx);
    }
};

// =========================================================================
// PHASE 2: COLLAPSE TO BVH4NODE (SIMD)
// =========================================================================
/*
    - Dump a raw vector of triangles in, calculates the statistical probability to build a tight binary tree.
    - Squashes it to utilize 100% of the CPU's hardware lanes.
*/

class BVHCollapser {
public:
    // Takes the binary tree from Phase 1, outputs the SIMD-ready tree for Phase 3
    void Collapse(const std::vector<BVHNode_Binary>& binaryNodes, std::vector<BVH4Node>& outBVH4) {
        if (binaryNodes.empty()) return;

        // BVH4 tree mathematically contains significantly fewer nodes 
        // than a binary tree. Pre-allocating the max size guarantees zero dynamic memory 
        // reallocations while the recursion runs.
        outBVH4.reserve(binaryNodes.size());
        
        // Root BVH4 node
        outBVH4.emplace_back(); 
        
        // Edge case: If the entire level geometry fits into a single leaf node
        if (binaryNodes[0].IsLeaf()) {
            ProcessLeafMap(binaryNodes[0], outBVH4, 0, 0);
            outBVH4[0].SetDegenerateLane(1);
            outBVH4[0].SetDegenerateLane(2);
            outBVH4[0].SetDegenerateLane(3);
        } else {
            // Kick off the greedy collapse
            ProcessNode(binaryNodes, outBVH4, 0, 0);
        }
    }

private:
    void ProcessNode(const std::vector<BVHNode_Binary>& binaryNodes, std::vector<BVH4Node>& outBVH4, uint32_t binIdx, uint32_t bvh4Idx) {
        
        // The pool holds the binary node indices we are considering collapsing
        uint32_t pool[4];
        int poolSize = 2;
        
        // Seed the pool with the two direct children of the current binary node
        // (Phase 1 guarantees that left and right children are allocated contiguously)
        pool[0] = binaryNodes[binIdx].leftFirst;
        pool[1] = binaryNodes[binIdx].leftFirst + 1;

        // --- GREEDY SURFACE AREA PULL-UP ---
        while (poolSize < 4) {
            int largestIdx = -1;
            float maxArea = -1.0f;

            // Find the internal node in the pool with the largest surface area
            for (int i = 0; i < poolSize; i++) {
                const BVHNode_Binary& n = binaryNodes[pool[i]];
                if (!n.IsLeaf()) {
                    float area = n.bounds.Area();
                    if (area > maxArea) {
                        maxArea = area;
                        largestIdx = i;
                    }
                }
            }

            // If we only have leaf nodes left in the pool, we cannot collapse any further
            if (largestIdx == -1) break;

            // Replace the largest node in the pool with its two children
            uint32_t largestBinIdx = pool[largestIdx];
            pool[largestIdx] = binaryNodes[largestBinIdx].leftFirst;     // Left child takes original slot
            pool[poolSize]   = binaryNodes[largestBinIdx].leftFirst + 1; // Right child goes to the end
            poolSize++;
        }

        // --- SIMD LANE POPULATION ---
        // Maps our pool (which now has 2, 3, or 4 optimized nodes) into the SoA structure
        for (int i = 0; i < 4; i++) {
            if (i < poolSize) {
                uint32_t childBinIdx = pool[i];
                const BVHNode_Binary& childBin = binaryNodes[childBinIdx];

                // 1. Splat the bounds into the Structure of Arrays (SoA)
                outBVH4[bvh4Idx].minX[i] = childBin.bounds.bmin.x();
                outBVH4[bvh4Idx].minY[i] = childBin.bounds.bmin.y();
                outBVH4[bvh4Idx].minZ[i] = childBin.bounds.bmin.z();
                
                outBVH4[bvh4Idx].maxX[i] = childBin.bounds.bmax.x();
                outBVH4[bvh4Idx].maxY[i] = childBin.bounds.bmax.y();
                outBVH4[bvh4Idx].maxZ[i] = childBin.bounds.bmax.z();

                // 2. Handle routing (Leaf vs Internal)
                if (childBin.IsLeaf()) {
                    ProcessLeafMap(childBin, outBVH4, bvh4Idx, i);
                } else {
                    // Allocate a new internal BVH4 node
                    uint32_t newBvh4Idx = static_cast<uint32_t>(outBVH4.size());
                    outBVH4.emplace_back();
                    
                    // Link it (MSB is 0, meaning internal)
                    outBVH4[bvh4Idx].children[i] = newBvh4Idx;
                    outBVH4[bvh4Idx].primCounts[i] = 0;

                    // Recurse down the tree
                    ProcessNode(binaryNodes, outBVH4, childBinIdx, newBvh4Idx);
                }
            } else {
                // 3. Mask off empty lanes with Infinities so SIMD safely ignores them
                outBVH4[bvh4Idx].SetDegenerateLane(i);
            }
        }
    }

    // Maps a binary leaf into a specific SIMD lane
    void ProcessLeafMap(const BVHNode_Binary& leafNode, std::vector<BVH4Node>& outBVH4, uint32_t bvh4Idx, int lane) {
        // Set MSB to 1 to flag as leaf, store index in lower 31 bits
        outBVH4[bvh4Idx].children[lane] = leafNode.leftFirst | 0x80000000;
        
        // Store how many triangles are in this leaf using the padding bytes
        outBVH4[bvh4Idx].primCounts[lane] = leafNode.primCount; 
    }
};

// =========================================================================
// PHASE 3: ZERO-BRANCH TRAVERSAL 
// =========================================================================

class LinearBVH {
private:
    std::vector<BVH4Node> m_nodes;
    std::vector<uint32_t> m_primitiveIndices; // Sorted indices of triangles/static meshes
    uint32_t m_rootNodeIndex = 0;
    std::vector<Tri4> m_tri4Geometry;

    AABB m_rootBounds;

public:
    const AABB& GetRootBounds() const { return m_rootBounds; }

    // Builds the BVH (Typically done once on level load, or offline during asset baking)
    // Pipeline that connects Phase 1 and Phase 2
    void Build(std::span<const Triangle> levelGeometry) {
        // Phase 1: Build the perfect SAH Binary Tree
        BVHBuilder builder;
        builder.Build(levelGeometry);

        // Steal the root bounds from Phase 1 before it collapses!
        m_rootBounds = builder.GetBinaryNodes()[0].bounds;

        // Phase 2: Collapse it into the Wide BVH4 SIMD format
        BVHCollapser collapser;
        collapser.Collapse(builder.GetBinaryNodes(), m_nodes);
        m_primitiveIndices = builder.GetIndices();

        // Phase 2.5: Pack the sorted triangles into Tri4 SIMD blocks PER LEAF
        m_tri4Geometry.clear();
        
        // Pre-allocate to prevent dynamic memory reallocations during level load
        // Add a 10% buffer to account for degenerate padding in the leaves
        size_t estimatedBlocks = (m_primitiveIndices.size() / 4) + (m_nodes.size() / 2);
        m_tri4Geometry.reserve(estimatedBlocks);

        // Iterate over all nodes in the BVH4 tree we just collapsed
        for (auto& node : m_nodes) {
            for (int i = 0; i < 4; i++) {
                
                // Only process valid leaf lanes
                if (node.IsLeaf(i) && node.primCounts[i] > 0) {
                    
                    uint32_t firstTri = node.GetIndex(i);
                    uint32_t triCount = node.primCounts[i];
                    
                    // Record where in the Tri4 array this leaf's geometry will start
                    uint32_t startTri4Block = static_cast<uint32_t>(m_tri4Geometry.size());
                    uint32_t blocksNeeded = (triCount + 3) / 4; // Round up
                    
                    // Build the blocks for this specific leaf
                    for (uint32_t b = 0; b < blocksNeeded; b++) {
                        Tri4 tri4Block;
                        
                        for (int lane = 0; lane < 4; lane++) {
                            uint32_t localIdx = b * 4 + lane;
                            
                            if (localIdx < triCount) {
                                // Valid geometry
                                uint32_t realTriIndex = m_primitiveIndices[firstTri + localIdx];
                                tri4Block.SetTriangle(lane, levelGeometry[realTriIndex], realTriIndex);
                            } else {
                                // Degenerate geometry padding!
                                // Setting all vertices to 0,0,0 means edges are 0,0,0. 
                                // Cross products evaluate to 0, Determinant evaluates to 0. 
                                // The SIMD Backface Culling instantly drops this lane safely!
                                Triangle degenerate;
                                degenerate.v0 = SIMDVector3D(0, 0, 0);
                                degenerate.v1 = SIMDVector3D(0, 0, 0);
                                degenerate.v2 = SIMDVector3D(0, 0, 0);
                                tri4Block.SetTriangle(lane, degenerate, (uint32_t)-1);
                            }
                        }
                        m_tri4Geometry.push_back(tri4Block);
                    }
                    
                    // --- THE UPGRADE ---
                    // Re-link the leaf! It no longer points to the scalar m_primitiveIndices.
                    // It now points directly into our SIMD-ready Tri4 array!
                    node.children[i] = startTri4Block | 0x80000000; // Keep MSB as 1
                    node.primCounts[i] = blocksNeeded;              // Overwrite count to mean Block Count
                }
            }
        }
    }

    // --- SIMD RAY-AABB INTERSECTION (DISTANCE RETURNING) ---
    // Returns a 4-bit mask (0 to 15). 
    // Bit 0 = Child 0 hit. Bit 1 = Child 1 hit, etc.
    FORCE_INLINE int IntersectNode4(const BVH4Node& node, const Ray4& ray, float maxDist) const {

        // --- LOAD SIMD REGISTERS FROM ALIGNED ARRAYS ---
        __m128 minX = _mm_load_ps(node.minX);
        __m128 maxX = _mm_load_ps(node.maxX);
        __m128 minY = _mm_load_ps(node.minY);
        __m128 maxY = _mm_load_ps(node.maxY);
        __m128 minZ = _mm_load_ps(node.minZ);
        __m128 maxZ = _mm_load_ps(node.maxZ);
        
        // --- X Axis ---
        __m128 t1x = _mm_mul_ps(_mm_sub_ps(minX, ray.origX), ray.invDirX);
        __m128 t2x = _mm_mul_ps(_mm_sub_ps(maxX, ray.origX), ray.invDirX);
        __m128 tNear = _mm_min_ps(t1x, t2x);
        __m128 tFar  = _mm_max_ps(t1x, t2x);

        // --- Y Axis ---
        __m128 t1y = _mm_mul_ps(_mm_sub_ps(minY, ray.origY), ray.invDirY);
        __m128 t2y = _mm_mul_ps(_mm_sub_ps(maxY, ray.origY), ray.invDirY);
        tNear = _mm_max_ps(tNear, _mm_min_ps(t1y, t2y));
        tFar  = _mm_min_ps(tFar,  _mm_max_ps(t1y, t2y));

        // --- Z Axis ---
        __m128 t1z = _mm_mul_ps(_mm_sub_ps(minZ, ray.origZ), ray.invDirZ);
        __m128 t2z = _mm_mul_ps(_mm_sub_ps(maxZ, ray.origZ), ray.invDirZ);
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
    bool Raycast(const Ray4& ray, float& maxDistance, Tri4Hit& outBestHit) const {
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

            if (stackPtr >= 128) break;

            // Pop the top node
            uint32_t nodeIndex = stack[--stackPtr];
            const BVH4Node& node = m_nodes[nodeIndex];

            // Get the 4-bit mask of which of the 4 children were hit
            int hitMask = IntersectNode4(node, ray, maxDistance);

            // Loop as long as there is at least one bit set to 1 in the mask
            while (hitMask != 0) {
                
                uint32_t childIdx;
                // Hardware instruction to find the index of the lowest set bit (0, 1, 2, or 3)
                #ifdef _MSC_VER
                    unsigned long msChildIdx;
                    _BitScanForward(&msChildIdx, hitMask);
                    childIdx = msChildIdx;
                #else
                    childIdx = __builtin_ctz(hitMask);
                #endif

                // Clear that bit so we don't process it again on the next loop
                hitMask &= ~(1 << childIdx);

                if (node.IsLeaf(childIdx)) {
                    // --- NARROW PHASE (SIMD TRI4) ---
                    // You hit a leaf! Test the triangles using: node.GetIndex(childIdx)
                    // If you find a closer triangle:
                    // 1. Record hit data
                    // 2. maxDistance = new_triangle_t;
                    // 3. hitAnything = true;
                    // Thanks to Phase 2.5, these variables now point directly 
                    // to the pre-packaged, perfectly padded Tri4 blocks!
                    uint32_t startTri4Block = node.GetIndex(childIdx);
                    uint32_t tri4BlockCount = node.primCounts[childIdx]; 

                    for (uint32_t i = 0; i < tri4BlockCount; i++) {
                        Tri4Hit hit = IntersectTri4_MT(m_tri4Geometry[startTri4Block + i], ray, maxDistance);
                        
                        // Note: hitIndex is -1 if it missed or hit a degenerate padding lane
                        if (hit.hitIndex != INVALID_INDEX && hit.t < maxDistance) {
                            maxDistance = hit.t; 
                            hitAnything = true;

                            // Record the best hit data to pass up the chain!
                            outBestHit = hit;
                            
                            // Record hit data here
                            // bestHitData.t = hit.t;
                            // bestHitData.u = hit.u;
                            // bestHitData.v = hit.v;
                            // bestHitData.triangleIndex = hit.hitIndex;
                        }
                    }
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

// ==================================================================================
// TWO-LEVEL ACCELERATION STRUCTURE (TLAS, BLAS)
// ==================================================================================
/*
    - Handles dynamic moving geometry (like a car driving through a city), entities, characters, doors, etc..
    - BLAS: Contains raw triangles of a single, specific 3D model (e.g., car, player). Is built once and stored in Local Object Space centered at (0, 0, 0).
    - TLAS: Only contains instances which is a 3D Transfrom Matrix (Position, Rotation, Scale) and a pointer to a BLAS.
    - We move the ray, not the car's triangles.
    - This ensures millions of polygons are moved for the computational cost of a single 4x4 matrix inversion.
*/

struct alignas(64) Matrix4x4 {
    float m[16];

    Matrix4x4() {
        std::fill_n(m, 16, 0.0f);
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    SIMDVector3D MultiplyPoint(const SIMDVector3D& v) const {
        return SIMDVector3D(
            v.x() * m[0] + v.y() * m[4] + v.z() * m[8]  + m[12],
            v.x() * m[1] + v.y() * m[5] + v.z() * m[9]  + m[13],
            v.x() * m[2] + v.y() * m[6] + v.z() * m[10] + m[14]
        );
    }

    SIMDVector3D MultiplyDirection(const SIMDVector3D& v) const {
        return SIMDVector3D(
            v.x() * m[0] + v.y() * m[4] + v.z() * m[8],
            v.x() * m[1] + v.y() * m[5] + v.z() * m[9],
            v.x() * m[2] + v.y() * m[6] + v.z() * m[10]
        );
    }

    Matrix4x4 Transpose() const {
        Matrix4x4 res;
        res.m[0] = m[0]; res.m[1] = m[4]; res.m[2] = m[8]; res.m[3] = m[12];
        res.m[4] = m[1]; res.m[5] = m[5]; res.m[6] = m[9]; res.m[7] = m[13];
        res.m[8] = m[2]; res.m[9] = m[6]; res.m[10] = m[10]; res.m[11] = m[14];
        res.m[12] = m[3]; res.m[13] = m[7]; res.m[14] = m[11]; res.m[15] = m[15];
        return res;
    }

    Matrix4x4 Invert() const {
        // Fast affine inverse
        Matrix4x4 res;
        
        // Transpose top-left 3x3
        res.m[0] = m[0]; res.m[1] = m[4]; res.m[2] = m[8]; 
        res.m[4] = m[1]; res.m[5] = m[5]; res.m[6] = m[9]; 
        res.m[8] = m[2]; res.m[9] = m[6]; res.m[10]= m[10];
        
        res.m[3] = res.m[7] = res.m[11] = 0.0f;
        res.m[15] = 1.0f;
        
        // Translation
        res.m[12] = -(m[12] * res.m[0] + m[13] * res.m[4] + m[14] * res.m[8]);
        res.m[13] = -(m[12] * res.m[1] + m[13] * res.m[5] + m[14] * res.m[9]);
        res.m[14] = -(m[12] * res.m[2] + m[13] * res.m[6] + m[14] * res.m[10]);
        
        return res;
    }
};

FORCE_INLINE AABB CalculateWorldBounds(const AABB& localBounds, const Matrix4x4& transform) {
    AABB worldBounds;
    
    // Extract the 8 corners of the local AABB
    SIMDVector3D corners[8] = {
        SIMDVector3D(localBounds.bmin.x(), localBounds.bmin.y(), localBounds.bmin.z()),
        SIMDVector3D(localBounds.bmax.x(), localBounds.bmin.y(), localBounds.bmin.z()),
        SIMDVector3D(localBounds.bmin.x(), localBounds.bmax.y(), localBounds.bmin.z()),
        SIMDVector3D(localBounds.bmax.x(), localBounds.bmax.y(), localBounds.bmin.z()),
        SIMDVector3D(localBounds.bmin.x(), localBounds.bmin.y(), localBounds.bmax.z()),
        SIMDVector3D(localBounds.bmax.x(), localBounds.bmin.y(), localBounds.bmax.z()),
        SIMDVector3D(localBounds.bmin.x(), localBounds.bmax.y(), localBounds.bmax.z()),
        SIMDVector3D(localBounds.bmax.x(), localBounds.bmax.y(), localBounds.bmax.z())
    };

    // Transform each corner into world space and grow the new box
    for (int i = 0; i < 8; i++) {
        worldBounds.Grow(transform.MultiplyPoint(corners[i]));
    }

    return worldBounds;
}

FORCE_INLINE bool IntersectAABB(const AABB& bounds, const Ray& ray, float hitDistanceLimit) {
    float t1 = (bounds.bmin.x() - ray.origin.x()) * ray.invDirection.x();
    float t2 = (bounds.bmax.x() - ray.origin.x()) * ray.invDirection.x();
    float tNear = std::min(t1, t2);
    float tFar = std::max(t1, t2);

    t1 = (bounds.bmin.y() - ray.origin.y()) * ray.invDirection.y();
    t2 = (bounds.bmax.y() - ray.origin.y()) * ray.invDirection.y();
    tNear = std::max(tNear, std::min(t1, t2));
    tFar = std::min(tFar, std::max(t1, t2));

    t1 = (bounds.bmin.z() - ray.origin.z()) * ray.invDirection.z();
    t2 = (bounds.bmax.z() - ray.origin.z()) * ray.invDirection.z();
    tNear = std::max(tNear, std::min(t1, t2));
    tFar = std::min(tFar, std::max(t1, t2));

    return tNear <= tFar && tFar > 0.0f && tNear < hitDistanceLimit;
}

// Represents the object (e.g., car) existing in the world and stores the pre-calculated inverse matrix.
struct alignas(64) BVHInstance {
    // The World-Space bounds of this object (Calculated by transforming the BLAS root bounds)
    AABB worldBounds;
    
    // Pointer/Index to the static, local-space BLAS (The Car Model)
    const LinearBVH* blas;
    
    // The 4x4 Transformation Matrices
    Matrix4x4 transform;
    Matrix4x4 inverseTransform;

    bool hasMovedThisFrame = true; 

    // Call this every frame the car moves
    // Called externally by the Game Loop/Bridge
    void UpdateTransform(const Matrix4x4& newTransform) {
        transform = newTransform;
        inverseTransform = newTransform.Invert();
        
        // Transform the 8 corners of the BLAS root AABB by the new matrix
        // and create a new worldBounds AABB.
        worldBounds = CalculateWorldBounds(blas->GetRootBounds(), transform);

        // Flag it so the TLAS knows to process it this frame
        hasMovedThisFrame = true;
    }
};

// ==================================================================================
// LINEAR BOUNDING VOLUME HIERARCHY (LBVH) & MORTON CODES (Z-CURVE)
// ==================================================================================
/*
    - Use Morton Codes (Z-Curve) to take 3D coordinates of the car's center point (x, y, z) and interleave the binary bits of those numbers into a single 32-bit integer.
    - This maps a 3D world onto a 1D line called Z-Order Curve.
    - If two cars are physically close to eachother in the 3D world, their 1D morton code integers will be numerically close to eachother.
    - Dynamic real-time geometry.
*/

// Stores the 1D spatial integer and a pointer to the original instance
struct MortonData {
    uint32_t mortonCode;
    uint32_t instanceIndex;
};

class TLASBuilder {
private:
    std::vector<MortonData> m_mortonArray;
    std::vector<MortonData> m_radixBuffer; // Persistent scratch space to avoid allocations per frame
    uint32_t m_nodesUsed = 0;

    // Hardware-level bit manipulation: Expands a 10-bit int to 30 bits by inserting 2 zeros after each bit.
    FORCE_INLINE uint32_t ExpandBits(uint32_t v) {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    // Calculates a 30-bit Morton code for a 3D point strictly inside the unit cube [0,1].
    FORCE_INLINE uint32_t Morton3D(float x, float y, float z) {
        x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
        y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
        z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
        uint32_t xx = ExpandBits(static_cast<uint32_t>(x));
        uint32_t yy = ExpandBits(static_cast<uint32_t>(y));
        uint32_t zz = ExpandBits(static_cast<uint32_t>(z));
        return xx * 4 + yy * 2 + zz; // Interleave the bits
    }

    // --- AAA ENGINE RADIX SORT CORE ---
    void RadixSortMortonCodes() {
        const size_t N = m_mortonArray.size();
        if (N < 2) return;

        // Ensure our scratch buffer is appropriately sized without losing pre-allocated capacity
        if (m_radixBuffer.size() < N) {
            m_radixBuffer.resize(N);
        }

        // 4 passes (8-bits per pass for 32-bit integers), 256 buckets per pass
        uint32_t histograms[4][256] = { {0} };

        // OPTIMIZATION 1: Fused Histogram Generation
        // We scan the source data exactly once, pulling all 4 bytes out simultaneously.
        // This keeps the source data warm in the L1/L2 cache lines.
        for (size_t i = 0; i < N; ++i) {
            const uint32_t code = m_mortonArray[i].mortonCode;
            histograms[0][code & 0xFF]++;
            histograms[1][(code >> 8) & 0xFF]++;
            histograms[2][(code >> 16) & 0xFF]++;
            histograms[3][(code >> 24) & 0xFF]++;
        }

        // OPTIMIZATION 2: Prefix Sum (Exclusive Scan)
        // Turn counts into absolute starting array index offsets
        uint32_t offsets[4][256];
        for (int pass = 0; pass < 4; ++pass) {
            uint32_t currentOffset = 0;
            for (int bucket = 0; bucket < 256; ++bucket) {
                offsets[pass][bucket] = currentOffset;
                currentOffset += histograms[pass][bucket];
            }
        }

        // OPTIMIZATION 3: Zero-Branch Ping-Pong Scatters
        // Pass 0: m_mortonArray -> m_radixBuffer (Byte 0)
        for (size_t i = 0; i < N; ++i) {
            const uint32_t bucket = m_mortonArray[i].mortonCode & 0xFF;
            const uint32_t destIdx = offsets[0][bucket]++;
            m_radixBuffer[destIdx] = m_mortonArray[i];
        }

        // Pass 1: m_radixBuffer -> m_mortonArray (Byte 1)
        for (size_t i = 0; i < N; ++i) {
            const uint32_t bucket = (m_radixBuffer[i].mortonCode >> 8) & 0xFF;
            const uint32_t destIdx = offsets[1][bucket]++;
            m_mortonArray[destIdx] = m_radixBuffer[i];
        }

        // Pass 2: m_mortonArray -> m_radixBuffer (Byte 2)
        for (size_t i = 0; i < N; ++i) {
            const uint32_t bucket = (m_mortonArray[i].mortonCode >> 16) & 0xFF;
            const uint32_t destIdx = offsets[2][bucket]++;
            m_radixBuffer[destIdx] = m_mortonArray[i];
        }

        // Pass 3: m_radixBuffer -> m_mortonArray (Byte 3)
        // Final pass naturally lands the perfectly sorted data back into m_mortonArray
        for (size_t i = 0; i < N; ++i) {
            const uint32_t bucket = (m_radixBuffer[i].mortonCode >> 24) & 0xFF;
            const uint32_t destIdx = offsets[3][bucket]++;
            m_mortonArray[destIdx] = m_radixBuffer[i];
        }
    }

    // Recursively splits the 1D array to build the 3D tree
    AABB GenerateHierarchy(const std::vector<BVHInstance>& instances, std::vector<TLASNode>& outNodes, uint32_t nodeIdx, uint32_t start, uint32_t end) {
        TLASNode& node = outNodes[nodeIdx];

        if (start == end) {
            uint32_t instIdx = m_mortonArray[start].instanceIndex;

            // --- Assign scalar floats directly ---
            node.minX = instances[instIdx].worldBounds.bmin.x();
            node.minY = instances[instIdx].worldBounds.bmin.y();
            node.minZ = instances[instIdx].worldBounds.bmin.z();

            node.maxX = instances[instIdx].worldBounds.bmax.x();
            node.maxY = instances[instIdx].worldBounds.bmax.y();
            node.maxZ = instances[instIdx].worldBounds.bmax.z();

            // Set MSB to 1 to flag as leaf, lower 31 bits hold instance index
            node.leftFirst = instIdx | 0x80000000; 
            return instances[instIdx].worldBounds;
        }

        // --- INTERNAL NODE (Z-CURVE MEDIAN SPLIT) ---
        // Because the array is sorted by 3D spatial Morton Codes, a simple array bisection
        // statistically guarantees the two halves are spatially separated in the world.
        uint32_t split = start + (end - start) / 2;
        uint32_t leftChildIdx = m_nodesUsed++;
        uint32_t rightChildIdx = m_nodesUsed++;

        node.leftFirst = leftChildIdx;

        // Recurse down both sides
        AABB leftBounds = GenerateHierarchy(instances, outNodes, leftChildIdx, start, split);
        AABB rightBounds = GenerateHierarchy(instances, outNodes, rightChildIdx, split + 1, end);

        AABB combinedBounds;
        combinedBounds.Grow(leftBounds);
        combinedBounds.Grow(rightBounds);

        // --- Assign scalar floats directly ---
        node.minX = combinedBounds.bmin.x();
        node.minY = combinedBounds.bmin.y();
        node.minZ = combinedBounds.bmin.z();
        
        node.maxX = combinedBounds.bmax.x();
        node.maxY = combinedBounds.bmax.y();
        node.maxZ = combinedBounds.bmax.z();

        return combinedBounds;
    }

public:
    void Build(const std::vector<BVHInstance>& instances, std::vector<TLASNode>& outNodes) {
        uint32_t N = static_cast<uint32_t>(instances.size());
        if (N == 0) return;

        // 1. Pre-allocate exact node count (A perfectly balanced binary tree has 2N - 1 nodes)
        outNodes.resize(N * 2);
        m_mortonArray.resize(N);
        m_nodesUsed = 1;  // Root is at index 0

        // 2. Find the global bounding box of all instances to normalize the centroids
        AABB globalBounds;
        for (const auto& inst : instances) {
            globalBounds.Grow(inst.worldBounds);
        }

         // 3. Calculate Morton Codes for all instances
        float extentX = globalBounds.bmax.x() - globalBounds.bmin.x();
        float extentY = globalBounds.bmax.y() - globalBounds.bmin.y();
        float extentZ = globalBounds.bmax.z() - globalBounds.bmin.z();
        
        // Prevent division by zero if the entire scene is perfectly flat
        if (extentX == 0.0f) extentX = 1e-5f;
        if (extentY == 0.0f) extentY = 1e-5f;
        if (extentZ == 0.0f) extentZ = 1e-5f;

        float invExtentX = 1.0f / extentX;
        float invExtentY = 1.0f / extentY;
        float invExtentZ = 1.0f / extentZ;

        for (uint32_t i = 0; i < N; i++) {
            // Find the center of the instance
            float cx = (instances[i].worldBounds.bmin.x() + instances[i].worldBounds.bmax.x()) * 0.5f;
            float cy = (instances[i].worldBounds.bmin.y() + instances[i].worldBounds.bmax.y()) * 0.5f;
            float cz = (instances[i].worldBounds.bmin.z() + instances[i].worldBounds.bmax.z()) * 0.5f;

            // Normalize the center to a [0.0, 1.0] scale relative to the global scene bounds
            float nx = (cx - globalBounds.bmin.x()) * invExtentX;
            float ny = (cy - globalBounds.bmin.y()) * invExtentY;
            float nz = (cz - globalBounds.bmin.z()) * invExtentZ;
            
            m_mortonArray[i].instanceIndex = i;
            m_mortonArray[i].mortonCode = Morton3D(nx, ny, nz);
        }

        // 4. Sort the instances along the Space-Filling Z-Curve
        // For standard games (< 20,000 moving dynamic objects), std::sort runs in < 1 millisecond.
        // If you ever push beyond 100,000 dynamic objects, swap this for a custom linear-time Radix Sort to maintain high speeds.
        RadixSortMortonCodes();

        // 5. Recursively build the tree using the sorted array
        GenerateHierarchy(instances, outNodes, 0, 0, N - 1);
    }
};

class SceneTLAS {
private:
    std::vector<BVHInstance> m_instances;
    std::vector<TLASNode> m_tlasNodes; // Rebuilt every frame!

    // The builder and its internal vectors now PERSIST in memory
    TLASBuilder builder;

    // --- DEGRADATION TRACKING ---
    float m_optimalRootArea = 0.0f;
    int m_framesSinceRebuild = 0;

    bool m_treeNeedsRefit = true; // Force build on frame 1

    // Post-order recursive traversal to snap AABBs to new geometry positions
    AABB RefitNode(uint32_t nodeIdx) {
        TLASNode& node = m_tlasNodes[nodeIdx];

        // BASE CASE: We hit a leaf (A single Car/Entity)
        if (node.IsLeaf()) {
            uint32_t instIdx = node.GetIndex();
            const BVHInstance& inst = m_instances[instIdx];

            // Snap the leaf node bounds to the newly calculated world bounds of the car
            // --- Assign scalar floats directly ---
            node.minX = inst.worldBounds.bmin.x();
            node.minY = inst.worldBounds.bmin.y();
            node.minZ = inst.worldBounds.bmin.z();
            
            node.maxX = inst.worldBounds.bmax.x();
            node.maxY = inst.worldBounds.bmax.y();
            node.maxZ = inst.worldBounds.bmax.z();
            
            return inst.worldBounds; // Return the car's bounding box directly
        }

        // INTERNAL NODE: Phase 1's builder guarantees that the right child is exactly leftFirst + 1.
        uint32_t leftChildIdx = node.leftFirst;
        uint32_t rightChildIdx = node.leftFirst + 1;

        // Traverse down first (Post-Order)
        AABB leftBounds = RefitNode(leftChildIdx);
        AABB rightBounds = RefitNode(rightChildIdx);

        // Combine the updated children to form the new parent bounds
        AABB combinedBounds;
        combinedBounds.Grow(leftBounds);
        combinedBounds.Grow(rightBounds);

        // --- Assign scalar floats directly ---
        node.minX = combinedBounds.bmin.x();
        node.minY = combinedBounds.bmin.y();
        node.minZ = combinedBounds.bmin.z();
        
        node.maxX = combinedBounds.bmax.x();
        node.maxY = combinedBounds.bmax.y();
        node.maxZ = combinedBounds.bmax.z();

        return combinedBounds;
    }

public:
    // Call this ONLY when loading a level or when the tree is irreversibly bloated. After physics/animations update your car matrices
    void RebuildTLAS() {
        if (m_instances.empty()) return;
    
        // You would run a high-speed SAH or Morton Code builder here over m_instances.
        // It populates m_tlasNodes, nesting the instance AABBs perfectly.
        builder.Build(m_instances, m_tlasNodes);

        // Record the "perfect" surface area of the newly optimized tree
        if (!m_tlasNodes.empty()) {
            // Reconstruct the SIMDVector3D from the new 32-byte optimized scalar floats
            AABB rootBounds = { 
                SIMDVector3D(m_tlasNodes[0].minX, m_tlasNodes[0].minY, m_tlasNodes[0].minZ), 
                SIMDVector3D(m_tlasNodes[0].maxX, m_tlasNodes[0].maxY, m_tlasNodes[0].maxZ) 
            };
            m_optimalRootArea = rootBounds.Area();
        }
        
        m_framesSinceRebuild = 0;
    }

    // The Game Loop calls this to push new matrices
    FORCE_INLINE void UpdateInstanceTransform(uint32_t instanceIndex, const Matrix4x4& newTransform) {
        m_instances[instanceIndex].UpdateTransform(newTransform);
        m_treeNeedsRefit = true; // Tell the tree something moved!
    }

    // --- GAMEPLAY LOOP ENTRY POINT ---
    // Call this every single frame
    void UpdateTLAS() {
        if (m_tlasNodes.empty() || !m_treeNeedsRefit) return; // Instantly exit if scene is static!

        // Step 1: Snap the tree boxes to the new object positions
        AABB newRootBounds = RefitNode(0);
        float currentRootArea = newRootBounds.Area();
        m_framesSinceRebuild++;

        // Step 2: Clear the dirty flags for the next frame 
        for (auto& inst : m_instances) {
            inst.hasMovedThisFrame = false;
        }

        // Step 3: Degradation Heuristic
        // If the tree's surface area has grown by more than 30% (bloated with empty space),
        // or it has been 120 frames (~2 seconds), force a full Radix Sort bisection rebuild.
        constexpr float DEGRADATION_TOLERANCE = 1.30f; 
        constexpr int MAX_FRAMES_BEFORE_REBUILD = 120;

        if (currentRootArea > (m_optimalRootArea * DEGRADATION_TOLERANCE) || 
            m_framesSinceRebuild > MAX_FRAMES_BEFORE_REBUILD) {
            RebuildTLAS();  // Trash the refit and calculate a perfect tree next frame
        }

        // Reset the flag at the end to let it know that nothing is moving (e.g., game paused or player is still)
        m_treeNeedsRefit = false;
    }

    // Returns the number of overlapping instances hit
    uint32_t RaycastMulti(const Ray& worldRay, float maxDistance, std::span<uint32_t> outInstanceIndices) const {
        if (m_tlasNodes.empty() || outInstanceIndices.empty()) return 0;

        uint32_t hitCount = 0;
        uint32_t maxCapacity = static_cast<uint32_t>(outInstanceIndices.size());

        // 128 to match primary Raycast and prevent overflows
        uint32_t stack[128];
        uint32_t stackPtr = 0;
        stack[stackPtr++] = 0; // Push TLAS Root

        // --- TREE TRAVERSAL ---
        while (stackPtr > 0) {
            // Safety rail
            assert(stackPtr < 128 && "TLAS Traversal Stack Overflow!");

            if(stackPtr >= 128) break;

            uint32_t nodeIdx = stack[--stackPtr];
            const TLASNode& node = m_tlasNodes[nodeIdx];

            AABB nodeBounds;
            nodeBounds.bmin = SIMDVector3D(node.minX, node.minY, node.minZ);
            nodeBounds.bmax = SIMDVector3D(node.maxX, node.maxY, node.maxZ);

            // Standard AABB test. Notice we pass maxDistance, which NEVER shrinks in a Multi-Cast!
            float unusedHitDistance = 0.0f;
            if (IntersectAABB(nodeBounds, worldRay, maxDistance)) {
                
                if (node.IsLeaf()) {
                    uint32_t instIdx = node.GetIndex();
                    const BVHInstance& instance = m_instances[instIdx];

                    // --- THE MATRIX JUMP (INVERSE TRANSFORM) ---
                    SIMDVector3D localOrig = instance.inverseTransform.MultiplyPoint(worldRay.origin);
                    SIMDVector3D localDir = instance.inverseTransform.MultiplyDirection(worldRay.direction);

                    // Construct the heavy SIMD Ray4 for the BLAS
                    Ray4 localRay(localOrig, localDir);

                    // --- THE BLAS DIVE ---
                    // We use a local hit variable because we don't want the BLAS to shrink 
                    // the global maxDistance! We need the ray to keep piercing through.
                    Tri4Hit localHit;
                    float localMaxDistance = maxDistance; 
                    
                    if (instance.blas->Raycast(localRay, localMaxDistance, localHit)) {
                        // We hit a triangle inside this instance!
                        // Record the instance index into the span.
                        outInstanceIndices[hitCount++] = instIdx;

                        // If we have filled the requested array, bail out immediately 
                        // to prevent a buffer overrun.
                        if (hitCount >= maxCapacity) {
                            return hitCount; 
                        }
                    }
                } else {
                    // Internal Node: Push children
                    stack[stackPtr++] = node.leftFirst + 1;
                    stack[stackPtr++] = node.leftFirst;
                }
            }
        }

        return hitCount;
    }

    // The Master Raycast Entry Point
    bool Raycast(const Ray& worldRay, RayHit& outHit) const {
        if (m_tlasNodes.empty()) return false;

        uint32_t stack[64];
        uint32_t stackPtr = 0;
        stack[stackPtr++] = 0; // Push TLAS Root

        // --- PHASE 1: TREE TRAVERSAL ---
        while (stackPtr > 0) {
            uint32_t nodeIdx = stack[--stackPtr];
            const TLASNode& node = m_tlasNodes[nodeIdx];

            // Standard scalar AABB intersection
            AABB nodeBounds;
            nodeBounds.bmin = SIMDVector3D(node.minX, node.minY, node.minZ);
            nodeBounds.bmax = SIMDVector3D(node.maxX, node.maxY, node.maxZ);

            // Standard AABB test against the instance's world bounds
            if (IntersectAABB(nodeBounds, worldRay, outHit.t)) {
                
                if (node.IsLeaf()) {
                    uint32_t instIdx = node.GetIndex();
                    const BVHInstance& instance = m_instances[instIdx];

                    // --- THE MATRIX JUMP (INVERSE TRANSFORM) ---
                    // We hit the car's bounding box in the world! 
                    // Now, teleport the ray into the car's local 0,0,0 space.
                    SIMDVector3D localOrig = instance.inverseTransform.MultiplyPoint(worldRay.origin);
                    SIMDVector3D localDir = instance.inverseTransform.MultiplyDirection(worldRay.direction);

                    // If a car is scaled up to be 3x larger than normal, this inverse matrix will shrink localDir to be 3x smaller than a unit vector (no need to normalize it).

                    // Construct the heavy SIMD Ray4 for the BLAS
                    Ray4 localRay(localOrig, localDir);

                    // --- THE BLAS DIVE ---
                    // Because BLAS::Raycast now takes maxDistance by reference, 
                    // outHitDistance will automatically shrink if a triangle is hit!
                    Tri4Hit localHit;
                    if (instance.blas->Raycast(localRay, outHit.t, localHit)) {
                        
                        // We found a closer hit! Record the payload, but DO NOT do normal math yet.
                        outHit.u = localHit.u;
                        outHit.v = localHit.v;
                        outHit.triangleIndex = localHit.hitIndex;
                        outHit.instanceIndex = instIdx;
                    }
                } else {
                    // Push children. (For max speed, you could sort these front-to-back here)
                    stack[stackPtr++] = node.leftFirst + 1;
                    stack[stackPtr++] = node.leftFirst;
                }
            }
        }

        return outHit.HasHit();
    }

    // --- PHASE 2: DEFERRED ATTRIBUTE EVALUATION ---
    // Call this only IF Raycast() returned true, and only for the final pixel/bullet calculation.
    SIMDVector3D GetWorldNormal(const RayHit& hit, const std::vector<Triangle>& rawLevelGeometry) const {
        const BVHInstance& inst = m_instances[hit.instanceIndex];
        
        // 1. Fetch the exact triangle from RAM (Only happens ONCE per ray!)
        const Triangle& tri = rawLevelGeometry[hit.triangleIndex];

        // 2. Calculate the local flat normal via cross product 
        // (Or interpolate vertex normals using hit.u and hit.v for smooth shading)
        SIMDVector3D e1 = tri.v1 - tri.v0;
        SIMDVector3D e2 = tri.v2 - tri.v0;
        SIMDVector3D localNormal = e1.cross(e2);

        // Inline Normalization
        float lenSq1 = localNormal.dot(localNormal);
        if (lenSq1 > 1e-8f) {
            localNormal = localNormal * (1.0f / std::sqrt(lenSq1));
        }

        // 3. THE AAA MATRIX TRAP: 
        // To rotate a normal into world space, you CANNOT use the standard transform matrix. 
        // If the car was scaled (e.g., squashed on the Z axis), the normal will warp.
        // Multiply the normal by the Transpose of the Inverse Matrix to prevent scaling distortion.
        Matrix4x4 transposeInverse = inst.inverseTransform.Transpose(); // Guarentees that lighting and physics reflections will always behave the same, even if the designer squashes, stretches, or warps the 3D models in the editor.
        SIMDVector3D worldNormal = transposeInverse.MultiplyDirection(localNormal);

        // Inline Normalization
        float lenSq2 = worldNormal.dot(worldNormal);
        if (lenSq2 > 1e-8f) {
            worldNormal = worldNormal * (1.0f / std::sqrt(lenSq2));
        }

        // Return the perfectly scaled, world-space normal.
        return worldNormal;
    }
};

// =========================================================================
// GAME LOOP ARCHITECTURE
// =========================================================================
/*
    // 1. To connect the physics system to the Graphics/TLS system, we need an entity registry. It stores IDs linking the different systems together.

    // A simple Entity representation
    struct Entity {
        uint32_t physicsID; // Index in the Physics Engine
        uint32_t tlasID;    // Index in the SceneTLAS
    };

    // 2. Query the physics engine for what moved, and pushes that data directly to the TLAS.

    class GameEngine {
    private:
        PhysicsEngine physics;
        SceneTLAS tlas;
        std::vector<Entity> activeEntities;

    public:
        void Tick(float deltaTime) {
            
            // ---------------------------------------------------------
            // 1. UPDATE PHYSICS
            // ---------------------------------------------------------
            // The physics engine solves collisions, gravity, and velocity.
            physics.StepSimulation(deltaTime);

            // ---------------------------------------------------------
            // 2. THE SYNCHRONIZATION BRIDGE (The Push)
            // ---------------------------------------------------------
            // Iterate through game entities and sync their matrices
            for (const auto& entity : activeEntities) {
                
                // Optimization: Only query awake/moving physics bodies
                if (physics.IsBodyAwake(entity.physicsID)) {
                    
                    // Pull matrix from Physics
                    Matrix4x4 newMatrix = physics.GetTransformMatrix(entity.physicsID);
                    
                    // Push matrix to TLAS
                    tlas.UpdateInstanceTransform(entity.tlasID, newMatrix);
                }
            }

            // ---------------------------------------------------------
            // 3. UPDATE SPATIAL HIERARCHY
            // ---------------------------------------------------------
            // The TLAS refits itself based ONLY on the instances that 
            // were flagged as 'hasMovedThisFrame' in Step 2.
            tlas.UpdateTLAS();

            // ---------------------------------------------------------
            // 4. RENDER / LOGIC
            // ---------------------------------------------------------
            // Now you can safely cast rays for AI line-of-sight, rendering, etc.
        }
    };
*/


