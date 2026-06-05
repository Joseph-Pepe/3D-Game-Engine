#pragma once

#include <vector>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <print>
#include <immintrin.h> // AVX and AVX2 intrinsics

// Engine Dependencies
#include "Math.h"
#include "imgui.h"

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


// C++26 Reflection Header
#include <meta> 


// ==================================================================================
// 1. THE COMPONENTS (Zero Boilerplate)
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
        
        // C++26: Get the reflection info object for the struct 'T'
        constexpr std::meta::info type_info = ^T;

        // C++26: 'template for' unrolls this loop at COMPILE TIME. 
        // Zero runtime branching. It physically emits the ImGui calls into the binary.
        template for (constexpr auto member : std::meta::data_members_of(type_info)) {
            
            // Get the actual string name of the variable (e.g., "velocity" or "mass")
            constexpr std::string_view name = std::meta::name_of(member);
            
            // Get the type of the variable
            constexpr std::meta::info member_type = std::meta::type_of(member);

            // C++26 SPLICING: [:member:] converts the reflection info back into actual C++ memory access!
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
        }
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

using ComponentRegistry = std::tuple<TransformComponent, PhysicsComponent>;

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

template <typename T>
constexpr uint32_t GetComponentID() {
    return static_cast<uint32_t>(ComponentIndex<T, ComponentRegistry>::value);
}

// ==================================================================================
// THE AOS & AoSoA CHUNK COMPONENT
// ==================================================================================

// 256 Entities per chunk. Must be a multiple of the SIMD width (8 for AVX2).
constexpr uint32_t ENTITIES_PER_CHUNK = 256; 

// alignas(64) ensures the chunk starts on a hardware cache-line boundary
struct alignas(64) PhysicsChunk {
    // How many entities in this specific chunk are currently alive? (0 to 256)
    uint32_t activeCount = 0; 

    // --- The Internal SoA Data ---
    // Because ENTITIES_PER_CHUNK is 256 (a multiple of 8), we don't need to worry about the SIMD Padding Trap!
    
    // Position (Vector3)
    alignas(32) float posX[ENTITIES_PER_CHUNK];
    alignas(32) float posY[ENTITIES_PER_CHUNK];
    alignas(32) float posZ[ENTITIES_PER_CHUNK];

    // Velocity (Vector3)
    alignas(32) float velX[ENTITIES_PER_CHUNK];
    alignas(32) float velY[ENTITIES_PER_CHUNK];
    alignas(32) float velZ[ENTITIES_PER_CHUNK];
};

// A single instance of this struct represents 8 discrete physical objects! alignas(32) ensures the memory address is perfectly aligned for AVX load instructions
struct alignas(32) PhysicsChunk8 {
    // Number of active entities in this specific chunk (0 to 8)
    uint32_t activeCount = 0;

    // Tightly packed arrays. Perfectly sized to instantly load into __m256 registers.

    // --- Position (8 entities) ---
    float posX[8] = {0};
    float posY[8] = {0};
    float posZ[8] = {0};

    // --- Velocity (8 entities) ---
    float velX[8] = {0};
    float velY[8] = {0};
    float velZ[8] = {0};
    
    // --- Mass (8 entities) ---
    float mass[8] = {1.0f};

    // If an object is static, it shouldn't be in the physics chunk at all!
};

// ==================================================================================
// AOS & AoSoA COMPONENTS
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
    using StorageType = std::vector<PhysicsChunk8>;
    static constexpr bool is_aosoa = true;
};



// ==================================================================================
// 4. THE ECS REGISTRY (Sparse Set / SoA Storage)
// ==================================================================================

using Entity = uint32_t;
constexpr Entity MAX_ENTITIES = 100000;

class ECS {
private:
    // C++26: Pack Indexing allows us to auto-generate vectors for every component in the tuple
    template <typename... Types>
    struct Storage {
        // DENSE ARRAYS: Now dynamically resolves to std::vector<T> OR std::vector<Chunk8>, the actual packed component data (No gaps!) 
        std::tuple<typename ComponentStorageTrait<Types>::StorageType...> denseArrays;
        
        // 2. SPARSE ARRAYS: Maps Entity ID -> logical dense index (0, 1, 2, 3...)
        // If an entity doesn't have the component, its value is -1.
        std::tuple<std::vector<uint32_t>...> sparseArrays;
        
        Storage() {
            auto init_sparse = []<std::size_t... I>(std::tuple<std::vector<uint32_t>...>& tup, std::index_sequence<I...>) {
                (std::get<I>(tup).resize(MAX_ENTITIES, static_cast<uint32_t>(-1)), ...);
            };
            init_sparse(sparseArrays, std::index_sequence_for<Types...>{});
        }
    };

    Storage<TransformComponent, PhysicsComponent> componentStorage;

    // Bitmask array: Each entity has a 32-bit signature showing which components it owns.
    // e.g., 0b011 means it has Transform (ID 0) and Physics (ID 1).
    std::vector<uint32_t> entitySignatures;
    uint32_t nextEntity = 0;

public:
    ECS() {
        entitySignatures.resize(MAX_ENTITIES, 0);
    }

    Entity CreateEntity() {
        return nextEntity++;
    }

    template <typename T>
    void AddComponent(Entity e, T component) {
        constexpr uint32_t compID = GetComponentID<T>();
        
        // Add the component to the bitmask, mark the signature
        entitySignatures[e] |= (1 << compID);

        auto& denseArray = std::get<compID>(componentStorage.denseArrays);
        auto& sparseArray = std::get<compID>(componentStorage.sparseArrays);

        // 1. If it already has the component, just update the dense data
        if (sparseArray[e] != static_cast<uint32_t>(-1)) {
            denseArray[sparseArray[e]] = component;
        } 
        // 2. Otherwise, pack it tightly at the end of the dense array
        else {
            uint32_t newIndex = static_cast<uint32_t>(denseArray.size());
            denseArray.push_back(component);
            sparseArray[e] = newIndex; // Map the Entity to its packed index
        }
    }

    template <typename T>
    decltype(auto) GetComponent(Entity e) {
        constexpr uint32_t compID = GetComponentID<T>();
        
        // Look up where this entity's data lives in the packed array
        uint32_t denseIndex = std::get<compID>(componentStorage.sparseArrays)[e];

        if constexpr (ComponentStorageTrait<T>::is_aosoa) {
            // AoSoA Path: Calculate Chunk and Lane
            uint32_t chunkIndex = denseIndex / 8;
            uint32_t laneIndex = denseIndex % 8;
            
            auto& chunkArray = std::get<compID>(componentStorage.denseArrays);
            
            // Return a proxy object or extract a temporary scalar struct.
            // For UI/Single Entity logic, we reconstruct the POD struct on the fly:
            T proxy;
            proxy.velocity.x = chunkArray[chunkIndex].velocityX[laneIndex];
            proxy.velocity.y = chunkArray[chunkIndex].velocityY[laneIndex];
            proxy.velocity.z = chunkArray[chunkIndex].velocityZ[laneIndex];
            proxy.mass = chunkArray[chunkIndex].mass[laneIndex];
            return proxy; 
        } else {
            // AoS Path: Standard direct reference return to the tightly packed data
            return std::get<compID>(componentStorage.denseArrays)[denseIndex];
        }
    }

    template <typename T>
    bool HasComponent(Entity e) {
        constexpr uint32_t compID = GetComponentID<T>();
        return (entitySignatures[e] & (1 << compID)) != 0;
    }

    // --- RENDER THE ENTIRE UI FOR AN ENTITY IN ONE LINE ---
    void DrawInspector(Entity e) {
        if (HasComponent<TransformComponent>(e)) {
            DrawComponentUI(GetComponent<TransformComponent>(e), "Transform");
        }
        
        if (HasComponent<PhysicsComponent>(e)) {
            DrawComponentUI(GetComponent<PhysicsComponent>(e), "Physics");
        }
    }
};

// 1. Add the Bridge Component (Auto-inspected by C++26 Reflection)
struct ParticleEmitterComponent {
    int activeParticles = 100000;
    float gravityPull = 5.0f;
    bool isAwake = true;

    // Pointer to the heavy silicon-level math manager
    ParticlePhysicsSOA* physicsEngine = nullptr; 
};

// 2. Register it in the Master Tuple
using ComponentRegistry = std::tuple<TransformComponent, PhysicsComponent, ParticleEmitterComponent>;
