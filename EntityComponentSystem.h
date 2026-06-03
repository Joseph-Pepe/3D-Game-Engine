#pragma once

#include <vector>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <print>



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
// 4. THE ECS REGISTRY (Sparse Set / SoA Storage)
// ==================================================================================

using Entity = uint32_t;
constexpr Entity MAX_ENTITIES = 100000;

class ECS {
private:
    // C++26: Pack Indexing allows us to auto-generate vectors for every component in the tuple
    template <typename... Types>
    struct Storage {
        std::tuple<std::vector<Types>...> arrays;
        
        Storage() {
            // Reserve memory for all arrays instantly
            auto reserve_all = []<std::size_t... I>(std::tuple<std::vector<Types>...>& tup, std::index_sequence<I...>) {
                (std::get<I>(tup).reserve(MAX_ENTITIES), ...);
            };
            reserve_all(arrays, std::index_sequence_for<Types...>{});
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
        
        // Add the component to the bitmask signature
        entitySignatures[e] |= (1 << compID);

        // Ensure the vector is large enough, then copy the data
        auto& compArray = std::get<compID>(componentStorage.arrays);
        if (compArray.size() <= e) {
            compArray.resize(e + 1);
        }
        compArray[e] = component;
    }

    template <typename T>
    T& GetComponent(Entity e) {
        constexpr uint32_t compID = GetComponentID<T>();
        return std::get<compID>(componentStorage.arrays)[e];
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
