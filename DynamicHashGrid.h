#pragma once

#include "Math.h"
#include <vector>
#include <span>
#include <concepts>
#include <algorithm>
#include <print>
#include <bit> // Required for std::bit_ceil

// ==================================================================================
// TRANSIENT SPATIAL HASH GRID (NETWORK & AI)
// ==================================================================================
/*
    - Perfect for Sparse, Unbounded, Macro-Level Systems (e.g., small number of items [500 to 5,000] scattered randomly across vast distances).
    - Never use this for dense, high-frequency physics (e.g. 100,000 colliding particles) b/c its a performance killer.
    - Used for physical game objects (Crates, Enemies, Vehicles).
    - Maps unbounded 3D coordinates into a fixed-size 1D array of "Buckets".
*/

namespace Engine::Database {

    struct SpatialCell {
        uint32_t startIndex; // Where in the sorted entity array does this cell begin?
        uint32_t count;      // How many entities are in this cell?
    };

    // Concept to ensure whatever we store can be mapped back to a 3D coordinate
    template <typename T>
    concept HasSpatialPosition = requires(T a) {
        { a.position.x } -> std::convertible_to<float>;
        { a.position.y } -> std::convertible_to<float>;
        { a.position.z } -> std::convertible_to<float>;
    };

    template <typename PayloadType>
    class TransientSpatialGrid {
    private:
        float m_cellSize;
        float m_invCellSize;
        uint32_t m_tableMask; // Power-of-2 mask for ultra-fast bitwise modulo

        // The Grid Buckets
        std::vector<SpatialCell> m_cells;
        std::vector<PayloadType> m_sortedPayloads;
        
        // Internal buffers for the counting sort
        std::vector<uint32_t> m_hashes;
        std::vector<uint32_t> m_cellCounts;

        // --- SPATIAL HASH FUNCTION ---
        // Uses prime numbers to scramble the grid coordinates into a semi-random bucket
        FORCE_INLINE uint32_t HashCoords(int32_t x, int32_t y, int32_t z) const {
            const uint32_t p1 = 73856093;
            const uint32_t p2 = 19349663;
            const uint32_t p3 = 83492791;
            
            uint32_t hash = (static_cast<uint32_t>(x) * p1) ^ 
                            (static_cast<uint32_t>(y) * p2) ^ 
                            (static_cast<uint32_t>(z) * p3);
                            
            // Bitwise (&) is significantly faster than integer divison of modulo.
            return hash & m_tableMask; // Zero-cost hardware modulo
        }

    public:
        // Table size MUST be a power of 2 (e.g., 4096, 8192, 16384)
        void Initialize(float cellSize, uint32_t maxItems, uint32_t tableSize = 8192) {
            // Assert power of two for the bitmask
            if ((tableSize & (tableSize - 1)) != 0) {
                std::println("[ERROR] TransientSpatialGrid tableSize must be a power of 2.");
                return;
            }

            m_cellSize = cellSize;
            m_invCellSize = 1.0f / cellSize;
            m_tableMask = tableSize - 1;

            m_cells.resize(tableSize);
            m_cellCounts.resize(tableSize);
            m_sortedPayloads.resize(maxItems);
            m_hashes.resize(maxItems);
        }

        // --- THE O(N) BATCH BUILDER ---
        void Build(std::span<const PayloadType> items, std::span<const Vector3D> positions) {
            size_t count = items.size();

            // 1. Reset cell counts to zero
            std::fill(m_cellCounts.begin(), m_cellCounts.end(), 0);

            // 2. Compute Hashes & Count elements per cell
            for (size_t i = 0; i < count; ++i) {
                // Scalar float-to-int conversion mapped to SSE/NEON via compiler
                int32_t gridX = static_cast<int32_t>(positions[i].x * m_invCellSize);
                int32_t gridY = static_cast<int32_t>(positions[i].y * m_invCellSize);
                int32_t gridZ = static_cast<int32_t>(positions[i].z * m_invCellSize);

                uint32_t hash = HashCoords(gridX, gridY, gridZ);
                m_hashes[i] = hash;
                m_cellCounts[hash]++;
            }

            // 3. Prefix Sum (Calculate memory offsets)
            uint32_t offset = 0;
            for (uint32_t i = 0; i <= m_tableMask; ++i) {
                m_cells[i] = { offset, m_cellCounts[i] };
                offset += m_cellCounts[i];
                m_cellCounts[i] = 0; // Reset for scatter pass
            }

            // 4. Scatter (Perfect memory alignment)
            for (size_t i = 0; i < count; ++i) {
                uint32_t hash = m_hashes[i];
                uint32_t dest = m_cells[hash].startIndex + m_cellCounts[hash];
                
                m_sortedPayloads[dest] = items[i];
                m_cellCounts[hash]++;
            }
        }

        // --- ZERO-ALLOCATION QUERY ---
        [[nodiscard]] FORCE_INLINE std::span<const PayloadType> QueryCell(const Vector3D& pos) const {
            int32_t gridX = static_cast<int32_t>(pos.x * m_invCellSize);
            int32_t gridY = static_cast<int32_t>(pos.y * m_invCellSize);
            int32_t gridZ = static_cast<int32_t>(pos.z * m_invCellSize);

            uint32_t hash = HashCoords(gridX, gridY, gridZ);
            const SpatialCell& cell = m_cells[hash];

            if (cell.count == 0) return {};
            return std::span<const PayloadType>(&m_sortedPayloads[cell.startIndex], cell.count);
        }
    };
}

namespace Engine::Systems {

    // ==============================================================
    // USE CASE 1: AI SENSORY & PERCEPTION
    // ==============================================================
    struct AIStimulus {
        uint32_t sourceEntityID;
        float intensity; // e.g., how loud the gunshot was
        enum class Type : uint8_t { Visual, Audio, Damage } type;
    };

    class AIPerceptionSystem {
        Engine::Database::TransientSpatialGrid<AIStimulus> m_stimulusGrid;

    public:
        void Initialize() {
            // 10-meter cells. Maximum 5000 active events per frame.
            m_stimulusGrid.Initialize(10.0f, 5000, 4096); 
        }

        void ProcessAITick(const Vector3D& aiPosition) {
            // The AI instantly queries its cell to "hear" or "see" everything nearby
            auto events = m_stimulusGrid.QueryCell(aiPosition);
            
            for (const auto& stimulus : events) {
                if (stimulus.type == AIStimulus::Type::Audio) {
                    // Trigger investigation behavior...
                }
            }
        }
    };

    // ==============================================================
    // USE CASES 2: NETWORK INTEREST & REPLICATION
    // ==============================================================
    struct NetworkEntity {
        uint32_t networkID;
        uint16_t prefabID;
    };

    class NetworkStream {
    public:
        void Serialize(uint32_t data) { /* Network byte packing logic */ }
    };

    class NetworkReplicationSystem {
        Engine::Database::TransientSpatialGrid<NetworkEntity> m_relevanceGrid;

    public:
        void Initialize() {
            // 50-meter cells. Max 100,000 replicated objects.
            m_relevanceGrid.Initialize(50.0f, 100000, 16384); 
        }

        void ReplicateToPlayer(const Vector3D& playerPos, class NetworkStream& stream) {
            // 1. Gather all entities in the player's 50m sector
            auto relevantEntities = m_relevanceGrid.QueryCell(playerPos);

            // 2. Serialize ONLY the entities the player can actually see
            for (const auto& entity : relevantEntities) {
                stream.Serialize(entity.networkID);
            }
            
            // 3. To handle multiple cells (e.g., 150m view distance), just query 
            // the adjacent gridX/Y/Z offsets and stream those spans too!
        }
    };
}

// ==================================================================================
// USE CASE 3: INFINITE CHUNK HASH DATABASE (PERSISTENT DATA)
// ==================================================================================

namespace Engine::Database {

    struct ChunkKey {
        int32_t x, y, z;
        
        // C++20/26 Spaceship operator for free equality generation
        auto operator<=>(const ChunkKey&) const = default;
    };

    struct WorldChunk {
        ChunkKey key;
        float* terrainData; // Pointer to massive VRAM/RAM buffers
        bool isActive;
    };

    class WorldChunkDatabase {
    private:
        static constexpr uint32_t EMPTY_HASH = 0xFFFFFFFF;
        
        struct HashSlot {
            uint32_t hash = EMPTY_HASH;
            ChunkKey key;
            uint32_t chunkIndex; // Index into m_denseChunks
        };

        std::vector<HashSlot> m_hashTable;
        std::vector<WorldChunk> m_denseChunks; // Contiguous array of actual loaded chunks
        
        uint32_t m_tableMask;
        uint32_t m_activeChunks;

        FORCE_INLINE uint32_t HashInteger(const ChunkKey& key) const {
            uint32_t h = (static_cast<uint32_t>(key.x) * 73856093) ^ 
                         (static_cast<uint32_t>(key.y) * 19349663) ^ 
                         (static_cast<uint32_t>(key.z) * 83492791);
            return h & m_tableMask;
        }

    public:
        void Initialize(uint32_t maxChunks = 4096) {
            // Keep table 2x larger than max chunks to prevent linear probing pileups
            uint32_t tableSize = std::bit_ceil(maxChunks * 2); 
            m_tableMask = tableSize - 1;

            m_hashTable.resize(tableSize);
            m_denseChunks.reserve(maxChunks);
            m_activeChunks = 0;
        }

        // --- O(1) CACHE-FRIENDLY INSERTION ---
        void InsertChunk(const ChunkKey& key, WorldChunk chunk) {
            uint32_t hash = HashInteger(key);
            uint32_t index = hash;

            // Linear Probing: If a slot is taken by a different chunk, step forward
            while (m_hashTable[index].hash != EMPTY_HASH) {
                if (m_hashTable[index].key == key) {
                    std::println("[WARNING] Chunk already exists at ({}, {}, {})", key.x, key.y, key.z);
                    return;
                }
                index = (index + 1) & m_tableMask; // Wrap around safely
            }

            // Store the chunk in the dense contiguous array
            m_denseChunks.push_back(chunk);

            // Register the lookup
            m_hashTable[index] = { hash, key, static_cast<uint32_t>(m_denseChunks.size() - 1) };
            m_activeChunks++;
        }

        // --- O(1) CACHE-FRIENDLY LOOKUP ---
        [[nodiscard]] WorldChunk* GetChunk(const ChunkKey& key) {
            uint32_t hash = HashInteger(key);
            uint32_t index = hash;

            while (m_hashTable[index].hash != EMPTY_HASH) {
                if (m_hashTable[index].key == key) {
                    return &m_denseChunks[m_hashTable[index].chunkIndex];
                }
                index = (index + 1) & m_tableMask;
            }

            return nullptr; // Chunk is not loaded in memory
        }
    };
}
