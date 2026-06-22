#pragma once
#include "Math.h"
#include <span>
#include <vector>

// ==================================================================================
// DYNAMIC SPATIAL HASH GRID (BROAD-PHASE ENTITY PHYSICS)
// ==================================================================================
/*
    - Rebuilt every frame. Zero heap allocations during the game loop.
    - Used for physical game objects (Crates, Enemies, Vehicles).
    - Maps unbounded 3D coordinates into a fixed-size 1D array of "Buckets".
    - We rebuild the grid from scratch every tick, takes less than a millisecond for 50,000 entities.
    - Also known as Dynamic Octrees.
    - As the player drives a car through the city, the car's absolute double position changes. 
    - Before the physics step, you snap the camera to the player, converting all cars and creates within a few kilometers into 32-bit camera relative floats.
    - It instantly finds the crates that it needs to test for collisions against, ignoring the other 5,000 crates in the world.
*/

struct SpatialHashCell {
    uint32_t startIndex; // Where in the sorted entity array does this cell begin?
    uint32_t count;      // How many entities are in this cell?
};

class SpatialHashGrid {
private:
    float m_cellSize;
    float m_invCellSize;
    uint32_t m_tableSize;

    // The Grid Buckets
    std::vector<SpatialHashCell> m_cells;
    
    // The Entity Indices, sorted so that entities in the same cell sit next to each other
    std::vector<uint32_t> m_sortedEntityIndices;
    
    // Internal buffers for the counting sort
    std::vector<uint32_t> m_entityCellHashes;
    std::vector<uint32_t> m_cellCounts;

    // --- SPATIAL HASH FUNCTION ---
    // Uses prime numbers to scramble the grid coordinates into a semi-random bucket
    FORCE_INLINE uint32_t HashCoords(int x, int y, int z) const {
        const uint32_t p1 = 73856093;
        const uint32_t p2 = 19349663;
        const uint32_t p3 = 83492791;
        
        uint32_t hash = (static_cast<uint32_t>(x) * p1) ^ 
                        (static_cast<uint32_t>(y) * p2) ^ 
                        (static_cast<uint32_t>(z) * p3);
                        
        return hash % m_tableSize;
    }

public:
    void Initialize(float cellSize, uint32_t maxEntities, uint32_t tableSize = 100003) {
        m_cellSize = cellSize;
        m_invCellSize = 1.0f / cellSize;
        m_tableSize = tableSize;

        m_cells.resize(tableSize);
        m_cellCounts.resize(tableSize);
        m_sortedEntityIndices.resize(maxEntities);
        m_entityCellHashes.resize(maxEntities);
    }

    // --- BUILD THE GRID (O(n) Time) ---
    // Pass in your Local (32-bit float) positions. Do NOT pass Large World Coordinates here.
    // LWC coordinates should be converted to camera-relative floats before physics ticks.
    void BuildGrid(std::span<const Vector3DStack> entityPositions, size_t activeEntities) {
        
        // 1. Reset cell counts to zero
        std::fill(m_cellCounts.begin(), m_cellCounts.end(), 0);

        // 2. Compute Hashes & Count elements per cell
        for (size_t i = 0; i < activeEntities; ++i) {
            int gridX = static_cast<int>(std::floor(entityPositions[i].data[0] * m_invCellSize));
            int gridY = static_cast<int>(std::floor(entityPositions[i].data[1] * m_invCellSize));
            int gridZ = static_cast<int>(std::floor(entityPositions[i].data[2] * m_invCellSize));

            uint32_t hash = HashCoords(gridX, gridY, gridZ);
            m_entityCellHashes[i] = hash;
            m_cellCounts[hash]++;
        }

        // 3. Prefix Sum: Calculate the starting index for each cell
        uint32_t currentOffset = 0;
        for (uint32_t i = 0; i < m_tableSize; ++i) {
            m_cells[i].startIndex = currentOffset;
            m_cells[i].count = m_cellCounts[i]; // Store count for the query phase
            currentOffset += m_cellCounts[i];
            
            // Reset the counter so we can use it as a running offset in Step 4
            m_cellCounts[i] = 0; 
        }

        // 4. Scatter: Place the entity indices into the perfectly contiguous sorted array
        for (size_t i = 0; i < activeEntities; ++i) {
            uint32_t hash = m_entityCellHashes[i];
            
            // Look up where this cell starts, add the running offset, and insert
            uint32_t destIndex = m_cells[hash].startIndex + m_cellCounts[hash];
            m_sortedEntityIndices[destIndex] = static_cast<uint32_t>(i);
            
            m_cellCounts[hash]++; // Increment offset for the next entity in this cell
        }
    }

    // --- QUERY THE GRID ---
    // Returns a span of entity indices that share the same grid cell.
    FORCE_INLINE std::span<const uint32_t> GetEntitiesInCell(const Vector3DStack& position) const {
        int gridX = static_cast<int>(std::floor(position.data[0] * m_invCellSize));
        int gridY = static_cast<int>(std::floor(position.data[1] * m_invCellSize));
        int gridZ = static_cast<int>(std::floor(position.data[2] * m_invCellSize));

        uint32_t hash = HashCoords(gridX, gridY, gridZ);
        
        const SpatialHashCell& cell = m_cells[hash];
        if (cell.count == 0) return {};

        // Because we sorted the array, we just return a lightweight view of the memory!
        return std::span<const uint32_t>(&m_sortedEntityIndices[cell.startIndex], cell.count);
    }
};
