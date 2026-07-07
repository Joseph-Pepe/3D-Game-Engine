#pragma once

#include <vector>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <print>
#include <immintrin.h> // AVX and AVX2 intrinsics
#include <memory>
#include <cstdint>
#include <cstring> // Required for std::memcpy
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <stdexcept>

// Engine Dependencies
#include "../Math.h"
#include "../imgui.h"
#include "../STLContainers/SmallVector.h"
#include "../Memory.h"               // Required for AlignedAllocator
#include "../SIMD/SIMDVectorMath.h"  // Required for NATIVE_SIMD_BATCH_ALIGN, PhysicsChunkNative


// ==================================================================================
// 1. C++ 26 COMPILE-TIME STATIC REFLECTION
// ==================================================================================
/*
    - The compiler can now look at its own datastructures, interrogate its types and variable names and automatically generate the boilerplate for us at zero runtime cost.
    - Compile-Time Component Type IDs (for lock free o(1) array lookups).
    - Auto Generating ImGui inspector (so you never have to write UI code for your components again).

    - e.g., if PhysicsComponent needs a new boolean or drag float, all you need to do is add it to the struct.
    - C++26 compiler detects the new variable, assigns the memory offsets, and injects the new slider into your ImGui debug dashboard automatically on the next compile.

*/

#if __has_include(<meta>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    #include <meta>        // Required for C++26 reflection
    #define ENGINE_HAS_CXX26_META_REFLECTION 1
#else
    #define ENGINE_HAS_CXX26_META_REFLECTION 0
#endif


// ==================================================================================
// 1. THE COMPONENTS (POD STRUCTS)
// ==================================================================================
// No macros, no inheritance, just pure Plain Old Data (POD) structs.

struct TransformComponent {
    Vector3DScalar position{0.0f, 0.0f, 0.0f};
    Vector3DScalar rotation{0.0f, 0.0f, 0.0f};
    Vector3DScalar scale{1.0f, 1.0f, 1.0f};
};

struct PhysicsComponent {
    Vector3DScalar velocity{0.0f, 0.0f, 0.0f};
    float mass = 1.0f;
    float friction = 0.5f;
    bool isStatic = false;
};

// 1. Add the Bridge Component (Auto-inspected by C++26 Reflection)
struct ParticleEmitterComponent {
    int activeParticles = 100000;
    float gravityPull = 5.0f;
    bool isAwake = true;

    // Pointer to the heavy silicon-level math manager
    Engine::Physics::ParticleMemoryBlock* memoryBlock = nullptr;
};

// ==================================================================================
// 2. C++26 AUTO-INSPECTOR (The Magic UI Generator)
// ==================================================================================
/*
    In older engines, adding a new variable to a component meant manually writing 
    ImGui::SliderFloat("Mass", &comp.mass, ...) in a UI file. 
    
    Using C++26 ^ (Reflection) and [: :] (Splicing), we ask the compiler to 
    iterate through the struct's variables at compile time and write the ImGui 
    code for us natively.
*/

template <typename T>
void DrawComponentUI(T& component, const char* componentName) {
    if (ImGui::CollapsingHeader(componentName, ImGuiTreeNodeFlags_DefaultOpen)) {

        #if ENGINE_HAS_CXX26_META_REFLECTION
            // C++26: Unroll the struct members at compile-time
            constexpr auto members = std::meta::nonstatic_data_members_of(^T);

            // Zero runtime branching. It physically emits the ImGui calls into the binary.
            [: expand(members) :] >> [&]<auto member>{
                // Get the actual string name of the variable (e.g., "velocity" or "mass")
                constexpr std::string_view name = std::meta::identifier_of(member);
                
                // Get the type of the variable
                constexpr std::meta::info member_type = std::meta::type_of(member);

                // C++26 SPLICING: [:member:] converts the reflection info back into actual C++ memory access! Access the memory address directly
                auto& value = component.[:member:];

                // --- AUTO-DETECT TYPES AND DRAW THE CORRECT UI ---
                if constexpr (member_type == ^float) {
                    ImGui::DragFloat(name.data(), &value, 0.1f);
                } 
                else if constexpr (member_type == ^int) {
                    ImGui::DragInt(name.data(), &value, 1);
                }
                else if constexpr (member_type == ^bool) {
                    ImGui::Checkbox(name.data(), &value);
                }
                else if constexpr (member_type == ^Vector3DScalar) {
                    // Because ImGui expects a float[3], we can safely cast our POD struct
                    ImGui::DragFloat3(name.data(), &value.x, 0.1f);
                }
            };
        #endif
    }
}

// ==================================================================================
// 3. COMPILE-TIME TYPE INDEXING
// ==================================================================================
/*
    Traditionally, ECS engines use `static atomic<int> type_counter++` to give components 
    unique IDs. This is dangerous because it evaluates at runtime and can break across 
    DLL boundaries (causing a Transform component to be ID 1 in the Engine, but ID 3 in the Game).
    
    We define our master list of components here. We use C++26 pack indexing 
    to assign them permanent, deterministic integer IDs at compile time.
*/

template <typename T, typename Tuple>
struct ComponentIndex;

template <typename T, typename... Types>
struct ComponentIndex<T, std::tuple<Types...>> {
    // C++20/26 constexpr trick to find the index of a type in a parameter pack
    static constexpr std::size_t value = []() consteval {
        constexpr std::array<bool, sizeof...(Types)> matches = {std::is_same_v<T, Types>...};
        for (std::size_t i = 0; i < matches.size(); ++i) {
            if (matches[i]) return i;
        }
        return static_cast<std::size_t>(-1); // Type not found
    }();
};

struct PositionComponent { float x, y, z; };
struct AITargetComponent { float targetX, targetY, targetZ; };
struct AIMovementComponent { float speed; };

// 1. The Global Master List (Used for the 64-bit Archetype Graph Mask)! Register all components in this tuple.
using GlobalComponentRegistry = std::tuple<
    TransformComponent, 
    PhysicsComponent, 
    ParticleEmitterComponent,
    PositionComponent,
    AITargetComponent,
    AIMovementComponent
>;

// 2. The Sparse List (Used ONLY for instantiating std::vector arrays)
using SparseComponentRegistry = std::tuple<
    ParticleEmitterComponent 
>;

// Distinct compile-time ID generators
template <typename T>
constexpr uint32_t GetGlobalComponentID() {
    constexpr std::size_t idx = ComponentIndex<T, GlobalComponentRegistry>::value;
    
    // If you forget to register a component, the compiler will now halt and print this exact message!
    static_assert(idx != static_cast<std::size_t>(-1), 
        "CRITICAL ERROR: Component type is not registered in GlobalComponentRegistry!");
        
    return static_cast<uint32_t>(idx); // return static_cast<uint32_t>(ComponentIndex<T, GlobalComponentRegistry>::value);
}

template <typename T>
constexpr uint32_t GetSparseComponentID() {
    constexpr std::size_t idx = ComponentIndex<T, SparseComponentRegistry>::value;
    
    static_assert(idx != static_cast<std::size_t>(-1), 
        "CRITICAL ERROR: Component type is not registered in SparseComponentRegistry!");
        
    return static_cast<uint32_t>(idx); // return static_cast<uint32_t>(ComponentIndex<T, SparseComponentRegistry>::value);
}

// ==================================================================================
// STORAGE CLASS TRAITS
// ==================================================================================
/*
    [AOS (Array of Structs)]

        - Used for simple logical components.
        - e.g., gameplay logic, AI, and player input.

    [AoSoA (Array of Structs of Arrays)]

        - Used for heavy math/physics components.
        - e.g., physics updates with 100% pure silicon throughput.
*/

// 1. Define a trait to determine how a component is stored (AOS or AoSoA)
template <typename T>
struct ComponentStorageTrait {
    // Default to standard AoS (Array of Structs)
    using StorageType = std::vector<T>;
    static constexpr bool is_aosoa = false;
};

// 2. Specialize the trait for your heavy math components
template <>
struct ComponentStorageTrait<PhysicsComponent> {
    // Override storage to use your AoSoA chunks
    using StorageType = std::vector<Engine::Physics::PhysicsChunkNative>;
    static constexpr bool is_aosoa = true;
};

/*
    [Sparse Set Storage]

        - Used for components that are added/removed constantly or sparse components that only a few entities possess.
        - e.g., StunnedTag, CameraFocus, Inventory.
        - Great for adding and removing components on the fly.
        - Cache misses when dealing with multiple components that need to run because these components are stored in isolated arrays, independent of eachother.

    [Archetype Storage]

        - Used for components that are iterated over sequentially every single frame and require perfect SIMD alignment.
        - e.g., changing [Transform, Physics, MeshRenderer] is rare.
        - Best for crunching millions of physics particles.
*/

enum class StorageBackend {
    Archetype,
    SparseSet
};

template <typename T>
struct ComponentTrait {
    // Default everything to Sparse Set (safer for random gameplay tags)
    static constexpr StorageBackend backend = StorageBackend::SparseSet;
};

// Specialize your heavy SIMD components to force them into the Archetype Graph
template <> 
struct ComponentTrait<TransformComponent> {
    static constexpr StorageBackend backend = StorageBackend::Archetype;
};

template <> 
struct ComponentTrait<PhysicsComponent> {
    static constexpr StorageBackend backend = StorageBackend::Archetype;
};

// ==================================================================================
// 5. ECS ARCHETYPE GRAPH MEMORY MODEL
// ==================================================================================
/*
    - An archetype is a container of chunks that stores a fixed number of entities, and its components in tightly packed, parallel arrays.
    - An entity only needs one entry in a central directory.
    - This directory knows which archetype the entity belongs to, and where it sits inside that archetype's memory chunks.
    - It groups entities with the exact same component signature into the same chunks of memory (i.e., tightly packed shared memory blocks).
    - CPU prefetcher loads both components into L1/L2 cache seamlessly.
    - Because an entity's components are strictly packed together in a chunk, you must move the entire entity to a different archetype.
    
    - [Transform] = to move the Object
    - [Physics] = the velocity
    - e.g., Archetype A (Transform + Physics) where [Pos, Pos, Pos] and [Vel, Vel, Vel] are packed side-by-side.
    - e.g., Archetype B (Transform + Physics + ParticleEmitter) stored in a completely separate memory block.
*/

using ComponentMask = uint64_t; // Supports up to 64 unique component types
using Entity = uint32_t;

// Max unique component types in the engine
constexpr uint32_t MAX_COMPONENTS = 64;
constexpr Entity MAX_ENTITIES = 100000;
constexpr uint32_t ENTITIES_PER_CHUNK = 1024;

// The central map for every alive entity
struct EntityRecord {
    ComponentMask archetypeSignature; // e.g., 0b011 (Transform + Physics)
    uint32_t chunkIndex;              // Which memory block in the Archetype?
    uint32_t laneIndex;               // Which row inside that block?
};

// Global lookup table
std::vector<EntityRecord> entityDirectory(MAX_ENTITIES);

// --- FLAT MEMORY CHUNK ---
struct ArchetypeChunk {
    uint32_t activeCount = 0;

    // REVERSE LOOKUP: (Maps Lane Index -> Entity ID), [small_vector ensures this stays inline until it overflows]
    Engine::STLContainer::small_vector<Entity, ENTITIES_PER_CHUNK> entities;   
    
    // 1. ONE single contiguous heap allocation for the entire chunk (and is guaranteed to be aligned for AVX2/AVX-512).
    std::vector<std::byte, /*Engine::Memory::AlignedAllocator*/ AlignedAllocator<std::byte, Engine::Physics::NATIVE_SIMD_BATCH_ALIGN>> rawMemory;
    
    // Type-erased memory for the components. 
    // 2. Zero-cost views into the flat memory block for each component type
    Engine::STLContainer::small_vector<std::span<std::byte>, MAX_COMPONENTS> componentBuffers;

    void AllocateFlatMemory(const std::vector<size_t>& strides) {
        size_t totalBytes = 0;
        for (size_t stride : strides) totalBytes += stride * ENTITIES_PER_CHUNK;
        
        rawMemory.resize(totalBytes);
        
        // Slice the raw memory into spans
        size_t offset = 0;
        for (size_t stride : strides) {
            size_t size = stride * ENTITIES_PER_CHUNK;
            componentBuffers.push_back(std::span<std::byte>(rawMemory.data() + offset, size));
            offset += size;
        }
    }
};

class Archetype {
public:
    ComponentMask signature;
    std::vector<uint32_t> componentIDs; // Which components are in this archetype?
    std::vector<size_t> componentStrides; // Size of each component (AoS or AoSoA lane)
    
    // The actual memory blocks
    std::vector<ArchetypeChunk> chunks;

    // ==========================================
    // EDGE GRAPH
    // ==========================================
    /*
        - Archetypes are treated as nodes in a state machine.
        - Edge: pre-computed pointer to the next archetype.
        - O(1) lookup: raw array of 64 pointers for our edges.
    */

    // If an entity in this archetype adds Component N, follow addEdges[N]
    std::array<Archetype*, MAX_COMPONENTS> addEdges{nullptr};
    
    // If an entity in this archetype removes Component N, follow removeEdges[N]
    std::array<Archetype*, MAX_COMPONENTS> removeEdges{nullptr};

    Archetype(ComponentMask sig) : signature(sig) {}

    Archetype(ComponentMask sig, std::vector<uint32_t> ids, std::vector<size_t> strides) 
        : signature(sig), componentIDs(ids), componentStrides(strides) {}
};

// ==================================================================================
// ARCHETYPE LAYOUT GENERATOR
// ==================================================================================
template <typename... Types>
constexpr auto GenerateComponentStrideTable(std::tuple<Types...>) {
    // Creates a compile-time array where Index == ComponentID, Value == Size in Bytes
    return std::array<size_t, sizeof...(Types)>{
        sizeof(typename ComponentStorageTrait<Types>::StorageType::value_type)...
    };
}

// Global static table of exact memory sizes for every Archetype component
constexpr auto GlobalComponentStrides = GenerateComponentStrideTable(GlobalComponentRegistry{});

class ArchetypeManager {
private:
    // Owns the memory of all unique archetypes in the engine
    std::vector<std::unique_ptr<Archetype>> allArchetypes;

    // Archetype based on a bitmask signature used to search and append to flat vectors.
    Archetype* GetOrCreateArchetype(ComponentMask targetSignature) {
        // 1. Does it already exist? (Fast linear scan)
        for (size_t i = 0; i < directoryKeys.size(); ++i) {
            if (directoryKeys[i] == targetSignature) return directoryValues[i];
        }

        // 2. If not, allocate it, initialize its memory chunks, and register it.
        auto newArchetype = std::make_unique<Archetype>(targetSignature);
        Archetype* ptr = newArchetype.get();
        
        // --- THE CRITICAL FIX: POPULATE THE METADATA ---
        // Iterate through all 64 possible component bits (initialize componentIDs and Strides here based on the mask)
        for (uint32_t compID = 0; compID < MAX_COMPONENTS; ++compID) {
            // If the bit for this component is flipped to 1...
            if ((targetSignature & (1ULL << compID)) != 0) {
                ptr->componentIDs.push_back(compID);
                
                // Pull the exact byte size from our constexpr lookup table
                ptr->componentStrides.push_back(GlobalComponentStrides[compID]); 
            }
        }
        
        // 3. Register in our flat directory arrays
        directoryKeys.push_back(targetSignature);
        directoryValues.push_back(ptr);
        
        allArchetypes.push_back(std::move(newArchetype));
        
        return ptr;
    }

public:
    // Flat arrays guarantee data is stored contiguously.
    std::vector<ComponentMask> directoryKeys;
    std::vector<Archetype*> directoryValues;

    ArchetypeManager() {
        // Always create the "Empty" archetype (Entity with no components) at boot
        GetOrCreateArchetype(0);
    }

    // Lookup for the ECS to find an entity's current home using linear scan of vector instead of a map.
    Archetype* GetArchetype(ComponentMask signature) {
        // Linear scan of a contiguous vector is significantly faster than std::unordered_map 
        // node-chasing for datasets under ~500 elements.
        for (size_t i = 0; i < directoryKeys.size(); ++i) {
            if (directoryKeys[i] == signature) return directoryValues[i];
        }
        return nullptr;
    }

    // ==========================================
    // GRAPH TRAVERSAL: The O(1) Transition
    // ==========================================
    Archetype* GetNextArchetype_Add(Archetype* current, uint32_t componentID) {
        // FAST PATH: The edge is already cached. O(1) return.
        if (current->addEdges[componentID] != nullptr) {
            return current->addEdges[componentID];
        }

        // SLOW PATH: First time this transition has ever happened.
        // Calculate what the new signature should be.
        ComponentMask newSignature = current->signature | (1ULL << componentID);

        // Find or create the destination archetype
        Archetype* nextArchetype = GetOrCreateArchetype(newSignature);

        // CACHE THE EDGES (Link the graph in both directions)
        current->addEdges[componentID] = nextArchetype;
        nextArchetype->removeEdges[componentID] = current;

        return nextArchetype;
    }
};

// ============================================================================================
// 4. THE ECS REGISTRY (HYBRID: SPARSE SET ARCHITECTURE & ARCHETYPE GRAPH: SoA / AoSoA Storage)
// ============================================================================================

class ECS {
private:

    // --- HEAVY STORAGE ---
    ArchetypeManager archetypeManager;

    // --- LIGHT STORAGE (Sparse Sets) --- We only create sparse sets for components marked as StorageBackend::SparseSet
    template <typename... Types>
    struct SparseStorage {
        // DENSE ARRAYS: Automatically aligned for SIMD! The actual packed data.
        std::tuple<std::vector<typename ComponentStorageTrait<Types>::StorageType::value_type, 
                   /*Engine::Memory::AlignedAllocator*/AlignedAllocator<typename ComponentStorageTrait<Types>::StorageType::value_type, Engine::Physics::NATIVE_SIMD_BATCH_ALIGN>>...> denseArrays;
        
        // 2. SPARSE ARRAYS: Maps Entity ID -> logical dense index (0, 1, 2, 3...), [if an entity doesn't have the component, its value is -1].
        // Alias maps every 'Type' in the pack to a uint32_t vector
        template <typename> 
        using SparseVec = std::vector<uint32_t>;

        // 2. SPARSE ARRAYS: Now the compiler knows to expand this once for every type in 'Types'
        std::tuple<SparseVec<Types>...> sparseArrays;

        uint32_t denseEntityCount = 0;
        
        // [...]: C++26 Pack Indexing allows us to auto-generate vectors for every component in the tuple.
        SparseStorage() {
            auto init_sparse = [&]<std::size_t... I>(std::index_sequence<I...>) {
                // Fold expression safely resizes every vector in the tuple
                (std::get<I>(sparseArrays).resize(MAX_ENTITIES, static_cast<uint32_t>(-1)), ...);
            };
            init_sparse(std::index_sequence_for<Types...>{});
        }
    };

    // Use the Sparse Registry to generate the arrays!
    SparseStorage<ParticleEmitterComponent /*, TransformComponent, PhysicsComponent, other sparse components*/> sparseManager;

    // Bitmask array: Each entity has a 32-bit signature showing which components it owns.
    // e.g., 0b011 means it has Transform (ID 0) and Physics (ID 1).
    std::vector<EntityRecord> entityDirectory;
    uint32_t nextEntity = 0;

    // Ensures when UI or gameplay systems modify a proxy struct, it scatters the data safely back into the correct SIMD lanes without corrupting neighboring entities.
    template <typename T>
    void WriteToArchetypeChunk(Archetype* arch, uint32_t chunkIndex, uint32_t laneIndex, uint32_t globalID, const T& component) {
        // 1. Locate the component's byte buffer in this archetype using C++20/26 ranges
        // auto it = std::ranges::find(arch->componentIDs, globalID);
        // if (it == arch->componentIDs.end()) {
        //     throw std::runtime_error("Critical ECS Failure: Component ID not found in Archetype signature.");
        // }
        // size_t bufferIndex = std::distance(arch->componentIDs.begin(), it);

        // 1. Locate the component's byte buffer with DIRECT INDEX LOOP which compiles down to faster assembly for smaller arrays (e.g., ComponentIDs). 
        size_t bufferIndex = 0;
        for (; bufferIndex < arch->componentIDs.size(); ++bufferIndex) {
            if (arch->componentIDs[bufferIndex] == globalID) break;
        }
        if (bufferIndex == arch->componentIDs.size()) {
            throw std::runtime_error("Critical ECS Failure: Component ID not found.");
        }

        auto& byteBuffer = arch->chunks[chunkIndex].componentBuffers[bufferIndex];

        // 2. COMPILE-TIME BRANCH: Scatter (AoSoA) vs Direct Write (AoS)
        if constexpr (ComponentStorageTrait<T>::is_aosoa) {
            
            // --- AoSoA WRITE PATH ---
            using ChunkType = typename ComponentStorageTrait<T>::StorageType::value_type; 

            // DYNAMIC SCALING
            constexpr uint32_t BATCH_SIZE = Engine::Physics::NATIVE_BATCH_SIZE;
            
            uint32_t subChunkIndex = laneIndex / BATCH_SIZE;
            uint32_t innerLane = laneIndex % BATCH_SIZE;
            
            auto* chunkNative = reinterpret_cast<ChunkType*>(byteBuffer.data() + (subChunkIndex * sizeof(ChunkType)));
            
            // Scatter the proxy data back into the SIMD lanes
            if constexpr (std::is_same_v<T, PhysicsComponent>) {
                chunkNative->velX[innerLane] = component.velocity.x;
                chunkNative->velY[innerLane] = component.velocity.y;
                chunkNative->velZ[innerLane] = component.velocity.z;
                chunkNative->mass[innerLane] = component.mass;
                chunkNative->friction[innerLane] = component.friction;

                // Cast the proxy boolean into a SIMD-friendly float mask
                chunkNative->isStaticMask[innerLane] = component.isStatic ? 1.0f : 0.0f;
            } else {
                throw std::runtime_error("AoSoA proxy injection missing for this type.");
            }

        } else {
            
            // --- AoS WRITE PATH ---
            size_t stride = sizeof(T); 
            auto* ptr = reinterpret_cast<T*>(byteBuffer.data() + (laneIndex * stride));
            
            // Overwrite the contiguous memory block with the updated struct
            *ptr = component;
        }
    }

    // ==================================================
    // TYPE-AWARE ARCHETYPE READER
    // ==================================================

    template <typename T>
    auto ReadFromArchetypeChunk(Archetype* arch, uint32_t chunkIndex, uint32_t laneIndex, uint32_t globalID) -> std::conditional_t<ComponentStorageTrait<T>::is_aosoa, T, T&> {
        // 1. Locate the component's byte buffer in this archetype using C++20/26 ranges
        // auto it = std::ranges::find(arch->componentIDs, globalID);
        // if (it == arch->componentIDs.end()) {
        //     throw std::runtime_error("Critical ECS Failure: Component ID not found in Archetype signature.");
        // }
        // size_t bufferIndex = std::distance(arch->componentIDs.begin(), it);

        // 1. Locate the component's byte buffer with DIRECT INDEX LOOP which compiles down to faster assembly for smaller arrays (e.g., ComponentIDs). 
        size_t bufferIndex = 0;
        for (; bufferIndex < arch->componentIDs.size(); ++bufferIndex) {
            if (arch->componentIDs[bufferIndex] == globalID) break;
        }
        if (bufferIndex == arch->componentIDs.size()) {
            throw std::runtime_error("Critical ECS Failure: Component ID not found.");
        }


        auto& byteBuffer = arch->chunks[chunkIndex].componentBuffers[bufferIndex];

        // 2. COMPILE-TIME BRANCH: AoSoA vs AoS
        if constexpr (ComponentStorageTrait<T>::is_aosoa) {
            
            // --- AoSoA READ PATH (e.g., PhysicsComponent from Engine::Physics::PhysicsChunkNative) ---
            using ChunkType = typename ComponentStorageTrait<T>::StorageType::value_type; 

            constexpr uint32_t BATCH_SIZE = Engine::Physics::NATIVE_BATCH_SIZE;
            
            uint32_t subChunkIndex = laneIndex / BATCH_SIZE;
            uint32_t innerLane = laneIndex % BATCH_SIZE;
            
            // Cast the byte array to our aligned SIMD chunk type
            auto* chunkNative = reinterpret_cast<ChunkType*>(byteBuffer.data() + (subChunkIndex * sizeof(ChunkType)));
            
            // Reconstruct and return the proxy POD struct (Copy)
            if constexpr (std::is_same_v<T, PhysicsComponent>) {
                T proxy;
                proxy.velocity.x = chunkNative->velX[innerLane];
                proxy.velocity.y = chunkNative->velY[innerLane];
                proxy.velocity.z = chunkNative->velZ[innerLane];
                proxy.mass = chunkNative->mass[innerLane];
                proxy.friction = chunkNative->friction[innerLane];
                
                // Cast the SIMD float mask back to a standard C++ boolean for the UI/Proxy
                proxy.isStatic = (chunkNative->isStaticMask[innerLane] > 0.5f);
                return proxy;
            } else {
                // If you add more AoSoA types later, you handle their reconstruction here
                throw std::runtime_error("AoSoA proxy reconstruction missing for this type.");
            }

        } else {
            
            // --- AoS READ PATH (e.g., TransformComponent) ---
            // Direct memory mapping. We use sizeof(T) to guarantee type-safe strides.
            size_t stride = sizeof(T); 
            auto* ptr = reinterpret_cast<T*>(byteBuffer.data() + (laneIndex * stride));
            
            // Return a direct memory reference so ImGui (or systems) can mutate it in-place
            return *ptr;
        }
    }

    // ==================================================
    // RAW MEMORY MANIPULATION (COPYABLE SILICON BYTES)
    // ==================================================

    void AllocateLane(Entity e, Archetype* destArchetype, uint32_t& outChunk, uint32_t& outLane) {
        // 1. If we have no chunks, or the last chunk is full, allocate a new memory block
        if (destArchetype->chunks.empty() || destArchetype->chunks.back().activeCount == ENTITIES_PER_CHUNK) {
            ArchetypeChunk newChunk;
            
            newChunk.AllocateFlatMemory(destArchetype->componentStrides);
            
            // Allocate contiguous byte buffers for every component type in this archetype
            for (size_t stride : destArchetype->componentStrides) {
                newChunk.componentBuffers.emplace_back(stride * ENTITIES_PER_CHUNK);
            }
            destArchetype->chunks.push_back(std::move(newChunk));
        }

        outChunk = static_cast<uint32_t>(destArchetype->chunks.size() - 1);
        outLane = destArchetype->chunks.back().activeCount++;
        
        // Store the reverse lookup!
        destArchetype->chunks[outChunk].entities[outLane] = e; 
    }

    void MoveEntityData(Archetype* srcArchetype, uint32_t srcChunk, uint32_t srcLane,
                            Archetype* destArchetype, uint32_t destChunk, uint32_t destLane) {
        
        // If coming from the "Empty" Archetype (Signature 0), there is no data to move.
        if (srcArchetype->signature == 0) return;

        // Iterate through every component the OLD archetype had
        for (size_t srcIndex = 0; srcIndex < srcArchetype->componentIDs.size(); ++srcIndex) {
            
            uint32_t compID = srcArchetype->componentIDs[srcIndex];
            size_t stride = srcArchetype->componentStrides[srcIndex];

            // Find where this component lives in the NEW archetype's arrays
            auto destIt = std::find(destArchetype->componentIDs.begin(), destArchetype->componentIDs.end(), compID);
            size_t destIndex = std::distance(destArchetype->componentIDs.begin(), destIt);

            // Calculate exact memory addresses
            void* srcPtr = srcArchetype->chunks[srcChunk].componentBuffers[srcIndex].data() + (srcLane * stride);
            void* destPtr = destArchetype->chunks[destChunk].componentBuffers[destIndex].data() + (destLane * stride);

            // Blistering fast silicon copy
            std::memcpy(destPtr, srcPtr, stride);
        }
    }

    void FillHoleInChunk(Archetype* srcArchetype, uint32_t srcChunk, uint32_t srcLane) {
        // We don't do swap-and-pop on the empty archetype
        if (srcArchetype->signature == 0) return; 

        ArchetypeChunk& chunk = srcArchetype->chunks[srcChunk];
        uint32_t lastLane = chunk.activeCount - 1;

        // If the entity we moved wasn't the very last one in the chunk, we have a hole to fill
        if (srcLane != lastLane) {
            
            // 1. Move the data of the LAST entity into the HOLE
            for (size_t i = 0; i < srcArchetype->componentIDs.size(); ++i) {
                size_t stride = srcArchetype->componentStrides[i];
                
                void* holePtr = chunk.componentBuffers[i].data() + (srcLane * stride);
                void* lastPtr = chunk.componentBuffers[i].data() + (lastLane * stride);
                
                std::memcpy(holePtr, lastPtr, stride);
            }

            // 2. Move the Reverse Lookup Entity ID
            Entity movedEntity = chunk.entities[lastLane];
            chunk.entities[srcLane] = movedEntity;

            // 3. Update the Global Directory so the ECS knows the moved entity has a new lane!
            entityDirectory[movedEntity].laneIndex = srcLane;
        }

        // Shrink the active count (destroying the now-duplicated last lane)
        chunk.activeCount--;
    }

public:
    ECS() {
        entityDirectory.resize(MAX_ENTITIES);
    }

    Entity CreateEntity() {
        Entity e = nextEntity++;
        // Initialize the entity in the "Empty" archetype
        entityDirectory[e] = {0, 0, 0}; 
        return e;
    }

    // Handles writing logical struct data into memory (AoS or AoSoA lane)
    template <typename T>
    void SetComponent(Entity e, const T& component) {
        if constexpr (ComponentTrait<T>::backend == StorageBackend::Archetype) {
            // ROUTE TO ARCHEPTYPE
            constexpr uint32_t globalID = GetGlobalComponentID<T>(); // Use Global ID
            EntityRecord& record = entityDirectory[e];
            Archetype* arch = archetypeManager.GetArchetype(record.archetypeSignature);
            
            // Casting the type-erased std::byte array back to T
            WriteToArchetypeChunk(arch, record.chunkIndex, record.laneIndex, globalID, component);

        } else {
            // ROUTE TO SPARSE SET
            constexpr uint32_t sparseID = GetSparseComponentID<T>(); // USE SPARSE ID!
            uint32_t denseIndex = std::get<sparseID>(sparseManager.sparseArrays)[e];

            if constexpr (ComponentStorageTrait<T>::is_aosoa) {
                // DYNAMIC SCALING
                constexpr uint32_t BATCH_SIZE = Engine::Physics::NATIVE_BATCH_SIZE;

                uint32_t chunkIndex = denseIndex / BATCH_SIZE;
                uint32_t laneIndex = denseIndex % BATCH_SIZE;
                auto& chunk = std::get<sparseID>(sparseManager.denseArrays)[chunkIndex];
                
                // Scatter the data into the SIMD lanes
                chunk.velX[laneIndex] = component.velocity.x;
                chunk.velY[laneIndex] = component.velocity.y;
                chunk.velZ[laneIndex] = component.velocity.z;
                chunk.mass[laneIndex] = component.mass;
                chunk.friction[laneIndex] = component.friction;

                // Cast the boolean to the SIMD float mask
                chunk.isStaticMask[laneIndex] = component.isStatic ? 1.0f : 0.0f;
            } else {
                std::get<sparseID>(sparseManager.denseArrays)[denseIndex] = component;
            }
        }
    }

    void MoveEntityToNewArchetype(Entity e, uint32_t newComponentID) {
        EntityRecord& record = entityDirectory[e];
        Archetype* currentArchetype = archetypeManager.GetArchetype(record.archetypeSignature);

        // 1. Traverse the Graph
        Archetype* destinationArchetype = archetypeManager.GetNextArchetype_Add(currentArchetype, newComponentID);

        // 2. Allocate space in the destination archetype's chunks
        uint32_t destChunkIndex;
        uint32_t destLaneIndex;
        AllocateLane(e, destinationArchetype, destChunkIndex, destLaneIndex);

        // 3-4. Move the data (memcpy the old components to the new chunk)
        MoveEntityData(currentArchetype, record.chunkIndex, record.laneIndex,
                    destinationArchetype, destChunkIndex, destLaneIndex);

        // 5. Swap and Pop the hole left in the old archetype
        FillHoleInChunk(currentArchetype, record.chunkIndex, record.laneIndex);

        // 6. Update the central directory so the engine knows where the entity lives now
        record.archetypeSignature = destinationArchetype->signature;
        record.chunkIndex = destChunkIndex;
        record.laneIndex = destLaneIndex;
    }

    template <typename T>
    void AddComponent(Entity e, T component) {
        // C++17/26 Compile-Time Routing
        if constexpr (ComponentTrait<T>::backend == StorageBackend::Archetype) {
            constexpr uint32_t globalID = GetGlobalComponentID<T>();
            // 1. ROUTE TO ARCHEPTYPE GRAPH
            // Physically move the entity to the new Archetype chunk
            MoveEntityToNewArchetype(e, globalID);

            // 2. Safely write the new component using your type-aware C++26 writer!
            SetComponent(e, component);
        } else {
            // 2. ROUTE TO SPARSE SET (lightning-fast sparse set insertion)
            constexpr uint32_t sparseID = GetSparseComponentID<T>();
            auto& denseArray = std::get<sparseID>(sparseManager.denseArrays);
            auto& sparseArray = std::get<sparseID>(sparseManager.sparseArrays);

            // 1. If it already has the component, just update the dense data
            if (sparseArray[e] != static_cast<uint32_t>(-1)) {
                SetComponent(e, component);
            } 
            // 2. Otherwise, pack it tightly at the end of the dense array
            else {
                if constexpr (ComponentStorageTrait<T>::is_aosoa) {
                    constexpr uint32_t BATCH_SIZE = Engine::Physics::NATIVE_BATCH_SIZE;

                    // 1. Use the centralized counter to determine where we are
                    uint32_t currentCount = sparseManager.denseEntityCount;
                    
                    // 2. If the current count perfectly divides into the batch size, it means all existing chunks are 100% full, and we need a new one.
                    if (currentCount % BATCH_SIZE == 0) {
                         denseArray.push_back(typename ComponentStorageTrait<T>::StorageType::value_type{});
                    }

                    // 3. Calculate exact placement based on the master count
                    uint32_t chunkIndex = currentCount / BATCH_SIZE;
                    uint32_t laneIndex = currentCount % BATCH_SIZE;

                    // sparseArray[e] = (chunkIndex * BATCH_SIZE) + laneIndex;

                    // 4. Record the lookup, increment the master tracker, and write the data
                    sparseArray[e] = currentCount;
                    sparseManager.denseEntityCount++; // Increment safely!
                    SetComponent(e, component); // Write data into the newly reserved lane
                } else {
                    // Standard AoS Packing
                    uint32_t newIndex = static_cast<uint32_t>(denseArray.size());
                    denseArray.push_back(component);
                    sparseArray[e] = newIndex; // Map the Entity to its packed index
                }
            }
        }
    }

    template <typename T>
    // Returns a Reference for AoS, but a Copy for AoSoA to satisfy C++ return deduction
    auto GetComponent(Entity e) -> std::conditional_t<ComponentStorageTrait<T>::is_aosoa, T, T&> {
        if constexpr (ComponentTrait<T>::backend == StorageBackend::Archetype) {
            // --- 1. ROUTE TO ARCHETYPE GRAPH ---
            constexpr uint32_t globalID = GetGlobalComponentID<T>(); // Use Global ID
            EntityRecord& record = entityDirectory[e];
            Archetype* arch = archetypeManager.GetArchetype(record.archetypeSignature);
            
            // Find the byte offset for this specific component type within the archetype
            // (You will need to calculate this based on componentStrides in the Archetype class)
            // For now, this is the placeholder where you cast the type-erased std::byte memory back to T:
            return ReadFromArchetypeChunk<T>(arch, record.chunkIndex, record.laneIndex, globalID);
        } else {
            // --- 2. ROUTE TO SPARSE SET ---
            constexpr uint32_t sparseID = GetSparseComponentID<T>(); // USE SPARSE ID!

            // Look up where this entity's data lives in the packed array
            uint32_t denseIndex = std::get<sparseID>(sparseManager.sparseArrays)[e];

            if constexpr (ComponentStorageTrait<T>::is_aosoa) {

                // AoSoA Path: Calculate Chunk and Lane dynamically
                constexpr uint32_t BATCH_SIZE = Engine::Physics::NATIVE_BATCH_SIZE;

                // AoSoA Path: Calculate Chunk and Lane
                uint32_t chunkIndex = denseIndex / BATCH_SIZE;
                uint32_t laneIndex = denseIndex % BATCH_SIZE;

                auto& chunk = std::get<sparseID>(sparseManager.denseArrays)[chunkIndex];
                
                // For UI/Single Entity logic, we reconstruct the POD struct on the fly: Gather the data from the SIMD lanes into a temporary struct
                T proxy;
                proxy.velocity.x = chunk.velX[laneIndex];
                proxy.velocity.y = chunk.velY[laneIndex];
                proxy.velocity.z = chunk.velZ[laneIndex];
                proxy.mass = chunk.mass[laneIndex];
                proxy.friction = chunk.friction[laneIndex];

                // Cast the float mask back to boolean
                proxy.isStatic = (chunk.isStaticMask[laneIndex] > 0.5f);
                return proxy; 
            } else {
                // AoS Path: Standard direct reference return to the tightly packed data
                return std::get<sparseID>(sparseManager.denseArrays)[denseIndex];
            }
        }
    }

    template <typename T>
    auto& GetSparseDenseArray() {
        static_assert(ComponentTrait<T>::backend == StorageBackend::SparseSet, "Only for sparse sets.");
        constexpr uint32_t sparseID = GetSparseComponentID<T>();
        return std::get<sparseID>(sparseManager.denseArrays);
    }

    template <typename T>
    bool HasComponent(Entity e) {
        if constexpr (ComponentTrait<T>::backend == StorageBackend::Archetype) {
            constexpr uint32_t globalID = GetGlobalComponentID<T>(); // Use Global ID
            // Check the 64-bit archetype signature in the entity directory
            return (entityDirectory[e].archetypeSignature & (1ULL << globalID)) != 0;
        } else {
            constexpr uint32_t sparseID = GetSparseComponentID<T>(); // USE SPARSE ID!
            // Check if the sparse map has a valid index
            return std::get<sparseID>(sparseManager.sparseArrays)[e] != static_cast<uint32_t>(-1);
        }
    }

    // ==================================================================================
    // ECS QUERY ENGINE (The Data Filter)
    // ==================================================================================
    // Hunts through the ArchetypeManager and extracts spans of memory that matches the requested signature.
    // Example: auto query = ecs.Query<TransformComponent, PhysicsComponent>();

    template <typename... QueryTypes>
    struct QueryResult {
        size_t count = 0;
        
        // Parallel arrays of pointers pointing to the exact start of the memory chunks
        std::vector<std::tuple<typename ComponentStorageTrait<QueryTypes>::StorageType::value_type*...>> chunks;
        
        // How many valid entities exist inside each returned chunk
        std::vector<uint32_t> chunkActiveCounts;
    };

    template <typename... QueryTypes>
    QueryResult<QueryTypes...> Query() {
        QueryResult<QueryTypes...> result;

        // 1. Generate the bitmask signature for the components the system is asking for
        constexpr ComponentMask queryMask = (0ULL | ... | (1ULL << GetGlobalComponentID<QueryTypes>()));

        // 2. Scan the Archetype Manager to find every Archetype that matches the mask
        for (size_t i = 0; i < archetypeManager.directoryKeys.size(); ++i) {
            ComponentMask archMask = archetypeManager.directoryKeys[i];
            
            // Bitwise AND check: Does this archetype contain ALL the components we asked for?
            if ((archMask & queryMask) == queryMask) {
                Archetype* arch = archetypeManager.directoryValues[i];

                // 3. We found a matching Archetype! Now extract its actual memory chunks.
                for (auto& chunk : arch->chunks) {
                    if (chunk.activeCount == 0) continue; // Skip empty chunks
                    
                    result.count += chunk.activeCount;
                    result.chunkActiveCounts.push_back(chunk.activeCount);

                    // 4. Extract the exact memory pointer for each requested component
                    auto chunkPointers = std::make_tuple(
                        GetRawPointerFromChunk<QueryTypes>(arch, &chunk)...
                    );
                    
                    result.chunks.push_back(chunkPointers);
                }
            }
        }
        return result;
    }

    // --- WRITE-BACK PROXY STRATEGY FOR UI: RENDER THE ENTIRE UI FOR AN ENTITY ---
    void DrawInspector(Entity e) {
        if (HasComponent<TransformComponent>(e)) {
            // AoS returns a direct reference, ImGui modifies it directly in memory
            auto& transform = GetComponent<TransformComponent>(e);
            DrawComponentUI(transform, "Transform");
        }
        
        if (HasComponent<PhysicsComponent>(e)) {
            // AoSoA returns a copy. ImGui modifies the contiguous struct layout.
            auto physicsCopy = GetComponent<PhysicsComponent>(e);
            DrawComponentUI(physicsCopy, "Physics");
            
            // Write the modified data back into the scattered SIMD lanes
            SetComponent(e, physicsCopy); 
        }

        if (HasComponent<ParticleEmitterComponent>(e)) {
            auto& emitter = GetComponent<ParticleEmitterComponent>(e);
            DrawComponentUI(emitter, "Particle Emitter");
        }
    }
private:
    // extracts a correctly casted pointer from a specific chunk.
    template <typename T>
    auto* GetRawPointerFromChunk(Archetype* arch, ArchetypeChunk* chunk) {
        constexpr uint32_t globalID = GetGlobalComponentID<T>();

        // Find the byte buffer
        size_t bufferIndex = 0;
        for (; bufferIndex < arch->componentIDs.size(); ++bufferIndex) {
            if (arch->componentIDs[bufferIndex] == globalID) break;
        }

        using ReturnType = typename ComponentStorageTrait<T>::StorageType::value_type;
        return reinterpret_cast<ReturnType*>(chunk->componentBuffers[bufferIndex].data());
    }
};
