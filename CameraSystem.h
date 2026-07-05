#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <print>

#include <numbers> // C++20/26 Standardized Math Constants

#include "Math.h"
#include "BVHGrid.h"
#include "GPURHI.h"

// Check if the compiler has shipped the C++26 inplace_vector
#if __has_include(<inplace_vector>)
    #include <inplace_vector> // C++26 API provides a vector that stores data locally without ever touching the heap allocator.
    /*
        // Replaces (std::vector). Zero heap allocations. Data is perfectly contiguous on the stack.
        // Extremely cache friendly for your SIMD wrappers.
        std::inplace_vector<Vector3D, 64> localCluster;
    */

    // Alias the standard version into the engine namespace
    namespace Engine {
        template <typename T, std::size_t Capacity>
        using inplace_vector = std::inplace_vector<T, Capacity>;
    }
    #define ENGINE_HAS_CXX26_INPLACE_VECTOR 1
#else
    // Fallback to your custom implementation until MSVC updates
    #include "STLContainers/InplaceVector.h"

    namespace Engine {
        template <typename T, std::size_t Capacity>
        using inplace_vector = Engine::STLContainer::inplace_vector<T, Capacity>;
    }

    #define ENGINE_HAS_CXX26_INPLACE_VECTOR 0
#endif

#if __has_include(<meta>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    #include <meta>        // Required for C++26 reflection
    #define ENGINE_HAS_CXX26_META_REFLECTION 1

    // C++26 Reflection: Zero-overhead Enum to String
    template <typename E>
    constexpr std::string_view EnumToString(E value) {
        std::string_view result = "<unknown>";
        // [: :] is the C++26 splice operator, ^^ reflects the type
        [: expand(std::meta::enumerators_of(^^E)) :] >> [&]<auto e>{
            if (value == [:e:]) result = std::meta::identifier_of(e);
        };
        return result;
    }
#else
    #define ENGINE_HAS_CXX26_META_REFLECTION 0
#endif

// ==================================================================================
// 3D CAMERA & Catmull-Rom Spline
// ==================================================================================
/*
    - 3D camera is a single entity. 
    - Use Catmull-Rom Spline for cinematic paths for buttery smooth fly-throughs, and orbital paths without that rigid snapping of linear interpolation.
    - P(t) = (1/2) (2P_1 + t(-P_0 + P_2) + t^2(2P_0 - 5P_1 + 4P_2 - P_3) + t^3(-P_0 + 3P_1 - 3P_2 + P_3)), where it evaluates 4 points (P_0, P_1, P_2, P_3)
*/

// --- MATH UTILITIES FOR CAMERA PATHING (SPLINES & LINEAR ALGEBRA) ---
FORCE_INLINE Vector3DStack Lerp(const Vector3DStack& a, const Vector3DStack& b, float t) {
    // V = A + t * (B - A)
    return a + ((b - a) * t);
}

// 128-bit PURE SIMD Lerp (Executes entirely inside SSE/AVX registers)
FORCE_INLINE Vector3D Lerp(const Vector3D& a, const Vector3D& b, float t) {
    // V = A + t * (B - A)
    return a + ((b - a) * t);
}

FORCE_INLINE Vector3DStack CatmullRom(const Vector3DStack& p0, const Vector3DStack& p1, 
                                      const Vector3DStack& p2, const Vector3DStack& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    // Evaluates in a few CPU cycles using standard ALUs
    Vector3DStack v0 = p1 * 2.0f;
    Vector3DStack v1 = (p2 - p0) * t;
    Vector3DStack v2 = (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2;
    Vector3DStack v3 = (p1 * 3.0f - p0 - p2 * 3.0f + p3) * t3;

    return (v0 + v1 + v2 + v3) * 0.5f;
}

FORCE_INLINE Vector3D CatmullRom(const Vector3D& p0, const Vector3D& p1, 
                                 const Vector3D& p2, const Vector3D& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    // Evaluates in a few CPU cycles using standard SIMD FMA
    Vector3D v0 = p1 * 2.0f;
    Vector3D v1 = (p2 - p0) * t;
    Vector3D v2 = (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2;
    Vector3D v3 = (p1 * 3.0f - p0 - p2 * 3.0f + p3) * t3;

    return (v0 + v1 + v2 + v3) * 0.5f;
}

// Strongly typed movement strictly prevents invalid input
// enum class CameraMove {
//     FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN
// };

// Replaces 'enum class CameraMove'
struct CameraInputAxes {
    float MoveX = 0.0f; // Right (+1.0f) / Left (-1.0f)
    float MoveY = 0.0f; // Up (+1.0f) / Down (-1.0f)
    float MoveZ = 0.0f; // Forward (+1.0f) / Backward (-1.0f)

    // Reset every frame before polling hardware input
    void Clear() {
        MoveX = 0.0f;
        MoveY = 0.0f;
        MoveZ = 0.0f;
    }
};

// ===============================================
// ENTITY COMPONENT SYSTEM (ECS)
// ===============================================
/*
    - Is a component based architecture that decouples the data (where it is) from its logic (how it moves).
    
      1. Only one camera exists.
      2. The cinematic system drives it until the cutscene ends.
      3. The player's input controller dynamically attaches to it and takes over.
*/

// --- 1. CAMERA COMPONENT (PURE DATA) ---
// This is attached to your Entity. It has zero movement logic. It knows nothing about splines, keyboards, or gamepads.
// It only knows its physical properties and how to build its matrices using optimized SIMD functions.
struct alignas(16) CameraComponent {
    Vector3D Position;       // 16 bytes
    Quaternion Orientation;  // 16 bytes (Identity by default)

    float FOV = 90.0f;
    float AspectRatio = 16.0f / 9.0f;
    float NearClip = 0.1f;
    float FarClip = 10000.0f;

    // Default initialization
    CameraComponent(Vector3D startPos = Vector3D(0.0f, 0.0f, 0.0f)) {
        Position = startPos;
        Orientation = Quaternion(); // {0,0,0,1}
    }

    // --- PURE SIMD QUATERNION TO VIEW MATRIX (100% SIMD Matrix Generation) ---
    // Calculating the conjugated rotation and translation directly into columns without branching or extracting to an intermediate transform struct.
    Matrix4x4_SIMD GetViewMatrix() const {
        // 1. To get a View Matrix, we need the INVERSE of the camera's rotation.
        Quaternion invQ = Orientation.Conjugate();

        // 2. Precompute Fused terms for the Rotation Matrix
        float x2 = invQ.x() + invQ.x(), y2 = invQ.y() + invQ.y(), z2 = invQ.z() + invQ.z();
        float xx = invQ.x() * x2, xy = invQ.x() * y2, xz = invQ.x() * z2;
        float yy = invQ.y() * y2, yz = invQ.y() * z2, zz = invQ.z() * z2;
        float wx = invQ.w() * x2, wy = invQ.w() * y2, wz = invQ.w() * z2;

        // 3. Build the Rotation Axes (Right, Up, Forward)
        Vector3D r(1.0f - (yy + zz), xy - wz, xz + wy, 0.0f);
        Vector3D u(xy + wz, 1.0f - (xx + zz), yz - wx, 0.0f);
        Vector3D f(xz - wy, yz + wx, 1.0f - (xx + yy), 0.0f);

        // 4. Calculate SIMD Translation: T = -R * Position
        float tx = -r.dot(Position);
        float ty = -u.dot(Position);
        float tz = -f.dot(Position);
        __m128 translation = _mm_set_ps(1.0f, tz, ty, tx);

        // 5. Store directly into the Matrix format
        Matrix4x4_SIMD mat;
        mat.col[0] = r.reg;
        mat.col[1] = u.reg;
        mat.col[2] = f.reg;
        mat.col[3] = translation;

        return mat;
    }

    void PrintTelemetry() const {
        // Dynamically calculate the Forward vector from the Quaternion
        Vector3D localForward(0.0f, 0.0f, -1.0f, 0.0f);
        Vector3D currentFront = Orientation.RotateVector(localForward);
        
        // Calculate the absolute target position
        Vector3D absoluteTarget = Position + currentFront;

        std::println("Cam Pos: [{}, {}, {}] | Target: [{}, {}, {}]", 
                     Position.x(), Position.y(), Position.z(),
                     absoluteTarget.x(), absoluteTarget.y(), absoluteTarget.z());
    }
};

// ===============================================
// THIRD-PERSON CAMERA (ECS)
// ===============================================
/*
    - Use a spring arm that acts like an invisible boom-pole that is no longer attached directly to the player's coordinates.
    - It evaluates a raycast from the player to the camera.
    - If a ray hits a wall, the Spring arm instantly pulls the camera forward to prevent it from clipping through the geometry.
*/

enum class SpringArmOcclusionMode : uint8_t {
    PullForward,    // Classic: Camera zooms in to prevent clipping
    FadeOccluders   // Isometric: Camera stays static, walls become transparent
};

// --- 1. SPRING ARM COMPONENT (PURE DATA) ---
// Attach this to the Player/Vehicle Entity alongside the CameraComponent
struct alignas(16) SpringArmComponent {
    float TargetArmLength = 300.0f;
    float ProbeRadius = 15.0f; // Size of the camera to prevent clipping through tight corners

    // Offsets
    Vector3D TargetOffset = Vector3D(0.0f, 50.0f, 0.0f); // e.g., Look at the character's head, not their feet
    Vector3D SocketOffset = Vector3D(0.0f, 0.0f, 0.0f);  // Over-the-shoulder offset

    // Frame-rate Independent Lag Options
    bool bEnableCameraLag = true;
    float CameraLagSpeed = 15.0f; 

    // --- OCCLUSION SETTINGS ---
    SpringArmOcclusionMode OcclusionMode = SpringArmOcclusionMode::PullForward;
    
    // Internal state tracking for the smoothing math
    Vector3D PreviousDesiredPosition;
    
    SpringArmComponent() {
        PreviousDesiredPosition = Vector3D(0.0f, 0.0f, 0.0f);
    }
};

// --- 2. SYSTEM: SPRING ARM INPUT CONTROLLER ---
class SpringArmController {
public:
    // Frame-Rate Independent Exponential Decay Lerp
    // Mathematically guarantees identical smoothing curves at 30Hz, 60Hz, and 144Hz.
    static FORCE_INLINE Vector3D DecayLerp(const Vector3D& current, const Vector3D& target, float decaySpeed, float deltaTime) {
        // formula: current = target + (current - target) * exp2(-decaySpeed * dt)
        float decayFactor = std::exp2(-decaySpeed * deltaTime);
        return target + ((current - target) * decayFactor);
    }

    // Evaluates the Spring Arm and directly mutates the CameraComponent
    void Update(
        const Vector3D& playerPosition, 
        const Quaternion& playerRotation, // Where the player is aiming
        SpringArmComponent& arm, 
        CameraComponent& camera, 
        const SceneTLAS& physicsScene, 
        RenderSystem& renderer,
        float deltaTime) 
    {
        // 1. Calculate the actual target focal point (e.g., Character's head)
        Vector3D worldTargetOffset = playerRotation.RotateVector(arm.TargetOffset);
        Vector3D aimPoint = playerPosition + worldTargetOffset;

        // 2. Establish the back-vector (Where the camera WANTS to be)
        // A spring arm extends strictly backwards (-Z) relative to the rotation
        Vector3D localArmDirection(0.0f, 0.0f, 1.0f, 0.0f); 
        Vector3D worldArmDirection = playerRotation.RotateVector(localArmDirection);

        // 3. Calculate the over-the-shoulder offset
        Vector3D worldSocketOffset = playerRotation.RotateVector(arm.SocketOffset);

        // 4. Calculate the desired un-obstructed position
        Vector3D desiredPosition = aimPoint + (worldArmDirection * arm.TargetArmLength) + worldSocketOffset;

        // --- CAMERA SMOOTHING (LAG) ---
        if (arm.bEnableCameraLag) {
            desiredPosition = DecayLerp(arm.PreviousDesiredPosition, desiredPosition, arm.CameraLagSpeed, deltaTime);
        }
        arm.PreviousDesiredPosition = desiredPosition;


        // --- COLLISION RESOLUTION (BACKFACE CULLING) ---
        // If we cast from AimPoint (Player) to DesiredPosition (Camera), the ray hits the 
        // INSIDE of the wall. If your BVH culls backfaces, the ray will miss the wall entirely!
        // We MUST cast from the Camera to the Player.
        // Ensure your level geometry uses thick walls so the ray doesn't pass through culled backfaces!

        // 1. Vector pointing FROM Player TO Camera
        Vector3D rayDirection = desiredPosition - aimPoint;
        float desiredDistance = std::sqrt(rayDirection.dot(rayDirection));


        if (desiredDistance > 1e-4f) {
            rayDirection = rayDirection * (1.0f / desiredDistance); // Normalize

            // 2. Origin is the Player, shooting toward the Camera
            Ray cameraRay(aimPoint, rayDirection); // Cast from Camera -> Player

            if (arm.OcclusionMode == SpringArmOcclusionMode::PullForward) {
                // Single hit closest resolution
                RayHit hitResult;

                // Probe the BVH! We hit something! Check if the hit is closer than our desired arm length.
                if (physicsScene.Raycast(cameraRay, hitResult) && hitResult.t < desiredDistance) {
                    // 3. Pull the camera in to the hit point, moving from the PLAYER outward
                    float clampedDistance = std::max(0.0f, hitResult.t - arm.ProbeRadius);
                    desiredPosition = aimPoint + (rayDirection * clampedDistance);
                }
            } 
            else {
                // --- MULTI-HIT OCCLUSION FADING ---
                // Camera ignores physics and stays exactly where it is.
                // We ask the BVH for every mesh between the camera and the player.
                uint32_t occludedInstances[16]; // Max 16 walls to prevent array bloat
                std::span<uint32_t> hitSpan(occludedInstances, 16);
                
                uint32_t hitCount = physicsScene.RaycastMulti(cameraRay, desiredDistance, hitSpan);
                
                // Route these hits to the Material System
                for (uint32_t i = 0; i < hitCount; ++i) {
                    renderer.RequestOcclusionFade(occludedInstances[i]);
                }
            }
        }

        // 5. Finalize the Camera State
        camera.Position = desiredPosition;
        camera.Orientation = playerRotation; // Lock camera orientation to the arm's drive rotation
    }
};

/*
    - Render / Material System  (Smooth Fading)
    - The renderer flags that specific mesh instance with a target opacity.
    - DecayLerps the current opacity toward the target.


    struct DitheredFadeState {
        float CurrentOpacity = 1.0f;
        bool bWasOccludingThisFrame = false;
    };

    // Flat array mapping 1:1 with your TLAS BVHInstances
    std::vector<DitheredFadeState> InstanceFadeStates; 

    // Called by the SpringArmController
    void RequestOcclusionFade(uint32_t instanceIndex) {
        InstanceFadeStates[instanceIndex].bWasOccludingThisFrame = true;
    }

    // Called by the JobSystem right before pushing uniforms to Vulkan/DX12
    void UpdateFades(float deltaTime) {
        for (auto& state : InstanceFadeStates) {
            float targetOpacity = state.bWasOccludingThisFrame ? 0.2f : 1.0f;
            
            // Only do math if it isn't fully opaque
            if (state.CurrentOpacity != targetOpacity) {
                // DecayLerp ensures it fades smoothly over ~0.2 seconds
                state.CurrentOpacity = SpringArmController::DecayLerp(state.CurrentOpacity, targetOpacity, 20.0f, deltaTime);
                
                // Snap to 1.0 to prevent micro-calculations forever
                if (state.CurrentOpacity > 0.99f) state.CurrentOpacity = 1.0f; 
            }

            // Reset the flag for the NEXT frame. If the Spring Arm doesn't 
            // flag it again next frame, it will organically fade back to 1.0f.
            state.bWasOccludingThisFrame = false; 
        }
    }
*/

// ===============================================
// FIRST-PERSON CAMERA (ECS)
// ===============================================

// --- 2. SYSTEM: FREE LOOK INPUT CONTROLLER ---
// FreeLookController: Is a first-person (spectator/noclip) camera.
// Injects input accumulation into a generic CameraComponent. Strictly handles user input and applies mathematical deltas to any CameraComponent you hand it. 
class FreeLookController {
public:
    float MovementSpeed = 800.0f;
    float MouseSensitivity = 0.15f;

    // WASD Movement (0=Forward, 1=Backward, 2=Left, 3=Right, 4=Up, 5=Down)
    // void ProcessKeyboard(CameraMove direction, float deltaTime) {
    //     float velocity = MovementSpeed * deltaTime;

    //     // ===============================================
    //     // BRANCHLESS JUMP TABLES (SWITCH) 
    //     // ===============================================
    //     /*
    //         - An if-statement used would cause the CPU branch predictor to constantly try to guess which key the user was pressing.
    //         - Occassionally it would guess wrong and flush its execution pipeline (a penalty of ~15 cycles).
            
    //         - A switch statement is optimized into a highly efficient jump table.
    //         - Executes the exact movement in a single instruction cycle without evaluating the other conditions.
    //         - Calcultes an offset based on enum integer and performs a single, direct memory jump to the correct logic.
    //     */

    //     #if ENGINE_HAS_CXX26_META_REFLECTION
    //         // Automatically prints: "Input: FORWARD" without writing a massive switch block
    //         std::println("Input: {}", EnumToString(direction));
    //     #endif

    //     // Branchless Jump Table via Switch
    //     switch (direction) {
    //         case CameraMove::FORWARD:  Position = Position + (Front * velocity); break;
    //         case CameraMove::BACKWARD: Position = Position - (Front * velocity); break;
    //         case CameraMove::LEFT:     Position = Position - (Right * velocity); break;
    //         case CameraMove::RIGHT:    Position = Position + (Right * velocity); break;
    //         case CameraMove::UP:       Position = Position + (WorldUp * velocity); break;
    //         case CameraMove::DOWN:     Position = Position - (WorldUp * velocity); break;
    //     }
    // }

    // 'UpdatePosition' Replaces 'ProcessKeyboard' by using normalized vectors to move with a capped maximum speed in all directions.
    // Notice we pass the component by reference. The controller mutates the data.

    // --- TRUE 6-DOF MOVEMENT ---
    void UpdatePosition(CameraComponent& camera, const CameraInputAxes& input, float deltaTime) {
        
        // 1. Define the base coordinate axes
        Vector3D localRight(1.0f, 0.0f, 0.0f, 0.0f);
        Vector3D localUp(0.0f, 1.0f, 0.0f, 0.0f);
        Vector3D localForward(0.0f, 0.0f, -1.0f, 0.0f);

        // 2. Rotate the base axes by the camera's current Quaternion
        Vector3D camRight   = camera.Orientation.RotateVector(localRight);
        Vector3D camUp      = camera.Orientation.RotateVector(localUp);
        Vector3D camForward = camera.Orientation.RotateVector(localForward);

        // 3. True branchless accumulation
        // If an input axis is 0.0f, that vector component naturally zeroes out.
        Vector3D moveDirection = 
            (camForward * input.MoveZ) + 
            (camRight   * input.MoveX) + 
            (camUp      * input.MoveY);

        // 4. Prevent the "Diagonal Exploit"
        // If a player presses Forward(1) and Right(1), the vector length becomes ~1.414.
        // We must normalize the direction vector if its length exceeds 1.0.
        float squaredLength = moveDirection.dot(moveDirection);
        if (squaredLength > 1.0f) {
            float invLen = 1.0f / std::sqrt(squaredLength);
            moveDirection = moveDirection * invLen;
        }

        // 5. Apply Velocity
        camera.Position = camera.Position + (moveDirection * (MovementSpeed * deltaTime));
    }

    // --- GIMBAL-LOCK FREE ROTATION ---
    void ProcessMouseMovement(CameraComponent& camera, float xoffset, float yoffset) {
        float yawAngle   = -xoffset * MouseSensitivity;
        float pitchAngle = -yoffset * MouseSensitivity;

        // 1. Create Delta Quaternions from the mouse input
        // Pitch rotates around the LOCAL X Axis (1,0,0)
        Quaternion pitchDelta = Quaternion::AngleAxis(pitchAngle, Vector3D(1.0f, 0.0f, 0.0f, 0.0f));
        
        // Yaw rotates around the GLOBAL Y Axis (0,1,0) to prevent the camera from "rolling" diagonally 
        // If you want a Spaceship/Flight Sim camera, change this to Local Y!
        Quaternion yawDelta = Quaternion::AngleAxis(yawAngle, Vector3D(0.0f, 1.0f, 0.0f, 0.0f));

        // 2. Combine the Rotations via Hamilton Product
        // Order matters! Global Yaw pre-multiplies, Local Pitch post-multiplies.
        camera.Orientation = yawDelta * camera.Orientation * pitchDelta;

        // 3. Hardware Normalize to prevent floating-point drift over time
        camera.Orientation.Normalize();
    }

// private:
    // ===============================================
    // PRECISION LOSS PREVENTION 
    // ===============================================
    /*
        - Its best to prevent tiny floating-point truncation errors from compounding, causing the camera to slowly drift off its perfect axis.
        - std::numbers::pi_v<float> means the compiler uses the absolute maximum IEEE-754 floating-point precision available for the specific hardware architecture to represent pi.
        - This ensures perfect rotational stability no matter how long the player spins the camera.
    */
    // void UpdateCameraVectors(CameraComponent& camera) {
    //     Vector3DScalar front;

    //     // [C++20]: Constants at maximum hardware precision (Convert to radians)
    //     // [std::numbers::pi_v<float>]: Prevents floating-point truncation issues during rapid camera rotations.
    //     float yawRad = Yaw * (std::numbers::pi_v<float> / 180.0f);
    //     float pitchRad = Pitch * (std::numbers::pi_v<float> / 180.0f);

    //     // Spherical coordinates to Cartesian coordinates
    //     front.x = std::cos(yawRad) * std::cos(pitchRad);
    //     front.y = std::sin(pitchRad);
    //     front.z = std::sin(yawRad) * std::cos(pitchRad);

    //     // Normalize Front
    //     float lenF = std::sqrt(front.dot(front));
    //     camera.Front = front * (1.0f / lenF);

    //     // Re-calculate Right and Up
    //     camera.Right = camera.Front.cross(camera.WorldUp);
    //     float lenR = std::sqrt(camera.Right.dot(camera.Right));
    //     camera.Right = camera.Right * (1.0f / lenR);

    //     camera.Up = camera.Right.cross(camera.Front);
    //     float lenU = std::sqrt(camera.Up.dot(camera.Up));
    //     camera.Up = camera.Up * (1.0f / lenU);
    // }
};

/*
    // Main Loop - OLD WAY [ENUMS]
    // if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    //     camera.ProcessKeyboard(CameraMove::FORWARD, dt);
    // if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    //     camera.ProcessKeyboard(CameraMove::BACKWARD, dt);

    // MAIN LOOP - NEW WAY WITH CONTINUOUS STATE ACCUMULATION [BRANCHLESS VECTOR MATH]
    CameraInputAxes currentInput;
    currentInput.Clear();

    // Accumulate axes (creates perfectly balanced -1 to 1 scales)
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) currentInput.MoveZ += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) currentInput.MoveZ -= 1.0f;

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) currentInput.MoveX += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) currentInput.MoveX -= 1.0f;

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) currentInput.MoveY += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) currentInput.MoveY -= 1.0f;

    // Dispatch a single, branchless update
    camera.UpdatePosition(currentInput, dt);
*/

// --- 3. SYSTEM: CINEMATIC TRACK SPLINE CONTROLLER ---
// Injects mathematical spline evaluation into a generic CameraComponent. Track that acts as an invisible rail that can hijack any generic CameraComponent and drag it along the Catmull-Rom Spline.
class CinematicTrackController {
private:
    // C++26 (std::inplace_vector): Stored directly on the stack/arena. Zero heap fragmentation. Replaces std::vector
    // #if ENGINE_HAS_CXX26_INPLACE_VECTOR
    //     // Specify the maximum number of waypoints (e.g., 64). This pre-allocates exactly 1024 bytes (64 * 16 bytes) directly inside the class footprint.
    //     std::inplace_vector<Vector3DStack, 64> controlPoints;
    //     std::inplace_vector<Vector3DStack, 64> lookAtTargets;
    // #else
    //     // Kept as Vector3DStack because CatmullRom uses heavy vector math operations where SSE acceleration actually benefits the 4-point polynomial evaluation.
    //     std::vector<Vector3DStack> controlPoints;
    //     std::vector<Vector3DStack> lookAtTargets;
    // #endif

    // Guaranteed zero-allocation stack storage using our custom C++26 engine container
    Engine::STLContainer::inplace_vector<Vector3DStack, 64> controlPoints;
    Engine::STLContainer::inplace_vector<Vector3DStack, 64> lookAtTargets;

    float currentProgress = 0.0f;  // Global timeline progress (0.0 to 1.0)
    float traversalSpeed = 0.1f;   // Percentage of track completed per second

public:
    // Add a physical coordinate for the camera to fly through
    void AddWaypoint(const Vector3DStack& pos, const Vector3DStack& lookAt) {
        controlPoints.push_back(pos);
        lookAtTargets.push_back(lookAt);
    }

    // Set how fast the camera completes the entire track
    void SetSpeed(float speed) { traversalSpeed = speed; }

    // Evaluates the spline and overwrites the camera's transform data. Runs once per frame on the Main UI/Render Thread
    void Update(CameraComponent& camera, float deltaTime) {
        size_t count = controlPoints.size();
        if (count < 2) return;  // Need at least 2 points to Lerp, 4 to perfectly Spline

        // Advance global timeline
        currentProgress += traversalSpeed * deltaTime;
        if (currentProgress > 1.0f) currentProgress = 1.0f; 

        // 1. Calculate which segment of the spline we are currently in
        // A track with 5 points has 4 physical segments.
        float segmentCount = static_cast<float>(count - 1);
        float scaledProgress = currentProgress * segmentCount;

        // Truncate float to get the active array index
        int currentIndex = static_cast<int>(scaledProgress);

        // Local t is the progress strictly between the current node and the next node
        float localT = scaledProgress - static_cast<float>(currentIndex);

        // 2. Fetch the 4 Control Points (Clamp to bounds to prevent segfaults)
        int i0 = std::max(0, currentIndex - 1);
        int i1 = currentIndex;
        int i2 = std::min(static_cast<int>(count - 1), currentIndex + 1);
        int i3 = std::min(static_cast<int>(count - 1), currentIndex + 2);

        // 3. Compute Smooth Spline Position
        Vector3DStack splinePos = CatmullRom(controlPoints[i0], controlPoints[i1], controlPoints[i2], controlPoints[i3], localT);
        
        // Instantly load the 16-byte aligned stack data into a SIMD Vector3D
        // .asPoint() ensures W = 1.0f for spatial translation
        camera.Position = Vector3D(_mm_load_ps(splinePos.data)).asPoint(); 

        // 4. Overwrite Component Direction via Spline Target
        Vector3DStack splineTarget = Lerp(lookAtTargets[i1], lookAtTargets[i2], localT);
        
        // Load target directly into a SIMD register
        Vector3D targetVec(_mm_load_ps(splineTarget.data));
        
        // Calculate the new directional vector purely on the silicon
        Vector3D newFront = targetVec - camera.Position;
        newFront = newFront.asDirection(); // Enforce W = 0.0f
        
        float lenSq = newFront.dot(newFront);
        if (lenSq > 1e-8f) {
            newFront = newFront * (1.0f / std::sqrt(lenSq));
            
            // Instantly snap the camera's quaternion to look down the spline!
            camera.Orientation = Quaternion::FromDirection(newFront);
        }
    }
};

/*
    - Instantiate CameraComponent once.
    - Can hot swap who owns the camera based on the game state without deleting or allocating any new objects.  

    // 1. Create the singular Entity Data
    CameraComponent mainCamera(Vector3DScalar(0.0f, 200.0f, 1000.0f));

    // 2. Create the Systems
    FreeLookController playerController;
    CinematicTrackController introCutscene;
    introCutscene.AddWaypoint(...); // Add points

    bool isInCutscene = true;

    // 3. Main Engine Loop
    while (Engine.IsRunning()) {
        
        // Engine State Router
        if (isInCutscene) {
            // The Cutscene has full control over the camera
            introCutscene.Update(mainCamera, dt);
            
            // Example check: did we finish the track?
            if (introCutscene.IsFinished()) {
                isInCutscene = false; // Seamlessly hand control back to the player
            }
        } 
        else {
            // Accumulate player input
            CameraInputAxes currentInput = ProcessHardwareInput(); 
            
            // The Player has full control over the camera
            playerController.ProcessMouseMovement(mainCamera, mouseX, mouseY);
            playerController.UpdatePosition(mainCamera, currentInput, dt);
        }

        // 4. Send to Renderer (Completely agnostic to who moved it)
        Matrix4x4_SIMD viewMat = mainCamera.GetViewMatrix();
        Renderer::SubmitViewMatrix(viewMat);
    }
*/

// ===============================================
// CAMERA STATE MACHINE (ECS)
// ===============================================

// Defines the physical parameters for a specific camera state
struct CameraStateProfile {
    float FOV = 90.0f;
    float TargetArmLength = 300.0f;
    
    // Looks at the character's upper chest/head
    Vector3D TargetOffset = Vector3D(0.0f, 60.0f, 0.0f); 
    
    // Pushes the camera left/right for over-the-shoulder framing
    Vector3D SocketOffset = Vector3D(0.0f, 0.0f, 0.0f);  
    
    // How fast we transition INTO this state
    float TransitionSpeed = 12.0f; 
};

// --- ECS COMPONENT ---
// Attach this to your Player Entity alongside the SpringArm and CameraComponent
struct alignas(16) CameraStateComponent {
    CameraStateProfile TargetState;

    // Hardcoded AAA Profiles (In a real engine, you'd load these from a JSON/XML config)
    CameraStateProfile ProfileExploration;
    CameraStateProfile ProfileAiming;
    CameraStateProfile ProfileSprinting;
    CameraStateProfile ProfileCrawling;

    CameraStateComponent() {
        // 1. Default Exploration
        ProfileExploration.FOV = 85.0f;
        ProfileExploration.TargetArmLength = 280.0f;
        ProfileExploration.TargetOffset = Vector3D(0.0f, 50.0f, 0.0f);
        ProfileExploration.SocketOffset = Vector3D(0.0f, 0.0f, 0.0f); // Center aligned
        ProfileExploration.TransitionSpeed = 8.0f; // Smooth, relaxed return to default

        // 2. Aiming the Gun
        ProfileAiming.FOV = 55.0f; // Zoomed in for precision
        ProfileAiming.TargetArmLength = 80.0f; // Pulled in tight to the shoulder
        ProfileAiming.TargetOffset = Vector3D(0.0f, 65.0f, 0.0f); // Look higher up at the reticle
        ProfileAiming.SocketOffset = Vector3D(55.0f, 0.0f, 0.0f); // Shift right to clear the center screen
        ProfileAiming.TransitionSpeed = 18.0f; // Fast, snappy transition for combat responsiveness

        // 3. Sprinting
        ProfileSprinting.FOV = 100.0f; // Widen FOV for sense of speed
        ProfileSprinting.TargetArmLength = 350.0f; // Pull back slightly
        ProfileSprinting.TargetOffset = Vector3D(0.0f, 40.0f, 0.0f); // Character leans forward, lower target
        ProfileSprinting.SocketOffset = Vector3D(0.0f, 0.0f, 0.0f);
        ProfileSprinting.TransitionSpeed = 5.0f; // Gradual buildup

        // Initialize
        TargetState = ProfileExploration;
    }

    void SetState(const CameraStateProfile& newState) {
        TargetState = newState;
    }
};

// Reads the current SpringArmController length and evaluates the physics.
// Grabs the target properties amd smoothly interpolates the current CameraComponent and SpringArmComponent values towards them usiung frame-rate independent decay. 
class CameraStateController {
public:
    // Scalar Frame-Rate Independent Exponential Decay Lerp
    static FORCE_INLINE float DecayLerp(float current, float target, float decaySpeed, float deltaTime) {
        return target + (current - target) * std::exp2(-decaySpeed * deltaTime);
    }

    // Vector Frame-Rate Independent Exponential Decay Lerp
    static FORCE_INLINE Vector3D DecayLerp(const Vector3D& current, const Vector3D& target, float decaySpeed, float deltaTime) {
        float decayFactor = std::exp2(-decaySpeed * deltaTime);
        return target + ((current - target) * decayFactor);
    }

    // Mutates the live components to match the target state dynamically over time
    void Update(CameraStateComponent& state, SpringArmComponent& arm, CameraComponent& camera, float deltaTime) {
        
        float speed = state.TargetState.TransitionSpeed;

        // 1. Interpolate the Camera Lens properties
        camera.FOV = DecayLerp(camera.FOV, state.TargetState.FOV, speed, deltaTime);

        // 2. Interpolate the Spring Arm dimensions
        arm.TargetArmLength = DecayLerp(arm.TargetArmLength, state.TargetState.TargetArmLength, speed, deltaTime);
        
        arm.TargetOffset = DecayLerp(arm.TargetOffset, state.TargetState.TargetOffset, speed, deltaTime);
        arm.SocketOffset = DecayLerp(arm.SocketOffset, state.TargetState.SocketOffset, speed, deltaTime);
    }
};

/*
    // 1. Instantiate the ECS Data
    CameraComponent mainCamera;
    SpringArmComponent playerSpringArm;
    CameraStateComponent playerCameraState;

    // 2. Instantiate Systems
    CameraStateController stateController;
    SpringArmController armController;

    // 3. Main Loop
    while (Engine.IsRunning()) {
        
        // --- A. GAMEPLAY INPUT ROUTING ---
        // Example: Player holds Right Mouse Button to aim
        if (Input::IsButtonHeld(MOUSE_BUTTON_RIGHT)) {
            playerCameraState.SetState(playerCameraState.ProfileAiming);
        } 
        else if (Input::IsButtonHeld(KEY_SHIFT)) {
            playerCameraState.SetState(playerCameraState.ProfileSprinting);
        } 
        else {
            playerCameraState.SetState(playerCameraState.ProfileExploration);
        }

        // --- B. CAMERA PIPELINE EXECUTION ---
        // Order matters! 

        // Step 1: Smoothly mutate the FOV and Arm lengths toward the target profile
        stateController.Update(playerCameraState, playerSpringArm, mainCamera, dt);

        // Step 2: Now that the arm has its new frame-interpolated length, calculate collision and final position
        armController.Update(
            player.Position, 
            player.Rotation, 
            playerSpringArm, 
            mainCamera, 
            physicsScene, 
            renderer, 
            dt
        );

        // --- C. RENDER ---
        Matrix4x4_SIMD viewMat = mainCamera.GetViewMatrix();
        Renderer::SubmitViewMatrix(viewMat);
        // Submit mainCamera.FOV to Projection Matrix...
    }
*/

// ======================================================
// CINEMATIC FIXED | SPATIAL RAIL CAMERA SYSTEM (ECS)
// =====================================================
/*
    - Camera driven by the player's spatial position in the world instead of by time.
    - Its locked to static nodes or moves along fixed tracks (rails).
    - It dynamically slides and rotates to frame the player based on where the player stands.
*/

enum class FixedCameraMode : uint8_t {
    StaticStation, // Stationary camera that rotates to track player
    LinearRail     // Camera slides along a track and rotates to track player
};

// Contiguous, cache-aligned data structure for a cinematic tracking zone
struct alignas(16) CinematicZoneComponent {
    uint32_t ZoneID = 0;
    Vector3D TriggerMin;
    Vector3D TriggerMax;
    
    FixedCameraMode Mode = FixedCameraMode::StaticStation;
    
    // Rail bounds (If StaticStation, RailStart is the camera's fixed position)
    Vector3D RailStart;
    Vector3D RailEnd;
    
    float FOV = 60.0f;          // Cinematic cameras often use tighter FOVs (e.g., 45-60)
    float TrackingLag = 15.0f;   // Smoothing weight for tracking speed
};

// --- SYSTEM: CINEMATIC RAIL CONTROLLER ---
class CinematicRailController {
public:
    // Performance Optimization: Cache-friendly processing loop
    void Update(
        const Vector3D& playerPosition,
        const Engine::STLContainer::inplace_vector<CinematicZoneComponent, 32>& activeZones,
        CameraComponent& outCamera,
        float deltaTime)
    {
        // 1. Spatial Partitioning / Trigger Volume Scan
        // Linear scan over stack-allocated vector is faster than tree traversal for small counts
        const CinematicZoneComponent* currentZone = nullptr;
        for (const auto& zone : activeZones) {
            if (IsPointInVolume(playerPosition, zone.TriggerMin, zone.TriggerMax)) {
                currentZone = &zone;
                break; 
            }
        }

        // Fallback if player leaves all defined tracking volumes
        if (!currentZone) return;

        Vector3D targetCamPos;

        if (currentZone->Mode == FixedCameraMode::StaticStation) {
            targetCamPos = currentZone->RailStart;
        } 
        else if (currentZone->Mode == FixedCameraMode::LinearRail) {
            // --- SIMD SPATIAL PROJECTION onto the Camera Rail ---
            __m128 p = playerPosition.reg;
            __m128 a = currentZone->RailStart.reg;
            __m128 b = currentZone->RailEnd.reg;

            __m128 ab = _mm_sub_ps(b, a); // Vector from Rail Start to Rail End
            __m128 ap = _mm_sub_ps(p, a); // Vector from Rail Start to Player

            // Dot Product: ap . ab
            __m128 dotAP_AB = _mm_dp_ps(ap, ab, 0x77); 
            // Dot Product: ab . ab (Squared Length of Rail)
            __m128 dotAB_AB = _mm_dp_ps(ab, ab, 0x77);

            float scalarTop = _mm_cvtss_f32(dotAP_AB);
            float scalarBottom = _mm_cvtss_f32(dotAB_AB);

            // Prevent division by zero if the rail has no length
            float t = (scalarBottom > 1e-6f) ? (scalarTop / scalarBottom) : 0.0f;
            t = std::clamp(t, 0.0f, 1.0f); // Clamp camera onto the physical rail bounds

            // Position = RailStart + t * (RailEnd - RailStart)
            targetCamPos = Vector3D(_mm_add_ps(a, _mm_mul_ps(_mm_set1_ps(t), ab)));
        }

        // 2. Frame-Rate Independent Camera Lag (Smooth transitions along the rail)
        float decayFactor = std::exp2(-currentZone->TrackingLag * deltaTime);
        outCamera.Position = targetCamPos + ((outCamera.Position - targetCamPos) * decayFactor);

        // 3. Dynamic Framing / LookAt Rotation
        // Calculate the direction pointing from the smooth camera position directly to the player
        Vector3D lookDirection = playerPosition - outCamera.Position;
        lookDirection = lookDirection.asDirection(); // Force W component to 0.0f

        float lengthSq = lookDirection.dot(lookDirection);
        if (lengthSq > 1e-6f) {
            lookDirection = lookDirection * (1.0f / std::sqrt(lengthSq)); // Safe Normalize
            
            // Re-use your ultra-fast, trig-free direction to quaternion solver
            Quaternion targetOrientation = Quaternion::FromDirection(lookDirection);
            
            // Soft blend the camera's rotational shift to match player movement cleanly
            outCamera.Orientation = Quaternion::Slerp(outCamera.Orientation, targetOrientation, 1.0f - decayFactor);
        }

        // Pass the cinematic zone lens configuration down to the camera state
        outCamera.FOV = currentZone->FOV;
    }

private:
    static FORCE_INLINE bool IsPointInVolume(const Vector3D& p, const Vector3D& min, const Vector3D& max) {
        // High throughput branchless boundary check using SIMD mask comparisons
        __m128 point = p.reg;
        __m128 boxMin = min.reg;
        __m128 boxMax = max.reg;

        __m128 geMin = _mm_cmpge_ps(point, boxMin);
        __m128 leMax = _mm_cmple_ps(point, boxMax);
        __m128 inside = _mm_and_ps(geMin, leMax);

        int mask = _mm_movemask_ps(inside);
        return (mask & 0x07) == 0x07; // Check X, Y, and Z lanes simultaneously
    }
};

// ======================================================
// CAMERA RELATIVE INPUT ENGINE
// =====================================================
/*
    - Character movement system must dynamically transform the player's hardware input (X/Y input) by the Camera's View Matrix.
*/

struct alignas(16) RawInputState {
    // Gamepad left thumbstick or Keyboard WASD normalized axes (-1.0f to 1.0f)
    float MoveAxisX = 0.0f;
    float MoveAxisY = 0.0f;
    
    // Mouse deltas or Gamepad right thumbstick
    float LookDeltaX = 0.0f;
    float LookDeltaY = 0.0f;
    
    bool IsGamepad = false;
};

class CameraRelativeInputEngine {
public:
    // Computes a branchless, frame-rate independent world movement vector relative to the camera viewport
    [[nodiscard]] FORCE_INLINE Vector3D ComputeWorldMovement(const RawInputState& input, const CameraComponent& camera) noexcept {
        // High throughput early-out for dead-zones
        float sqMag = (input.MoveAxisX * input.MoveAxisX) + (input.MoveAxisY * input.MoveAxisY);
        if (sqMag < 0.01f) return Vector3D(0.0f, 0.0f, 0.0f);

        // Extract camera directional axes from its orientation quaternion
        // Assuming a standard Y-Up coordinate system
        Vector3D camForward = camera.Orientation.GetForwardVector();
        Vector3D camRight   = camera.Orientation.GetRightVector();

        // Project camera vectors onto the horizontal gameplay plane (Y = 0) to prevent 
        // characters from flying or digging into the ground when looking up/down
        camForward.setY(0.0f);
        camRight.setY(0.0f);

        // Perform fast safe normalization on the planar vectors using SIMD registers
        __m128 fReg = camForward.reg;
        __m128 rReg = camRight.reg;

        __m128 fDot = _mm_dp_ps(fReg, fReg, 0x77);
        __m128 rDot = _mm_dp_ps(rReg, rReg, 0x77);

        // Branchless reciprocal square root approximation with one iteration of Newton-Raphson refinement
        __m128 fRsqrt = _mm_rsqrt_ps(fDot);
        __m128 rRsqrt = _mm_rsqrt_ps(rDot);

        camForward.reg = _mm_mul_ps(fReg, fRsqrt);
        camRight.reg   = _mm_mul_ps(rReg, rRsqrt);

        // Accumulate input dimensions to compute final world velocity vector
        // MoveAxisY maps to Forward/Backward, MoveAxisX maps to Left/Right
        Vector3D worldDirection = (camForward * input.MoveAxisY) + (camRight * input.MoveAxisX);

        // Clamp vector magnitude to 1.0f to prevent diagonal acceleration exploits
        float mag = std::sqrt((worldDirection.x() * worldDirection.x()) + (worldDirection.z() * worldDirection.z()));
        if (mag > 1.0f) {
            worldDirection = worldDirection * (1.0f / mag);
        }

        return worldDirection;
    }
};

// ======================================================
// ZONE BLENDING 
// =====================================================
/*
    - Smooths crossfades between distinct camera angles (e.g., transition from a fixed overview to a side-scrolling rail track).
    - Perfroms an automated runtime blend.
*/

// Holds active crossfading contexts between old and new camera tracks
struct alignas(16) CameraBlendTracker {
    CinematicZoneComponent SourceZone;
    CinematicZoneComponent TargetZone;
    float BlendAlpha = 1.0f;       // 0.0f = Source, 1.0f = Target Completely Active
    float BlendSpeed = 2.0f;       // Speed factor of the structural crossfade
    bool IsBlending = false;
};

class SpatialRailCameraSystem {
private:
    CameraBlendTracker m_BlendState;

public:
    // Fully SIMD accelerated zone parsing logic
    void Update(
        const Vector3D& playerPosition,
        const Engine::STLContainer::inplace_vector<CinematicZoneComponent, 32>& levelZones,
        CameraComponent& outCamera,
        float deltaTime) noexcept 
    {
        // 1. Locate the single primary active camera zone based on spatial volume intersection
        const CinematicZoneComponent* primaryZone = nullptr;
        for (const auto& zone : levelZones) {
            if (IsPointInVolume(playerPosition, zone.TriggerMin, zone.TriggerMax)) {
                primaryZone = &zone;
                break;
            }
        }

        if (!primaryZone) return; // Retain current frame parameters if out of bounds

        // 2. Manage the Camera Transition State Machine
        if (!m_BlendState.IsBlending) {
            // Check if the player stepped into a new tracking environment
            if (m_BlendState.TargetZone.ZoneID != primaryZone->ZoneID) {
                // If it's the initial run, just snap to it instantly
                if (m_BlendState.TargetZone.ZoneID == 0) {
                    m_BlendState.TargetZone = *primaryZone;
                } else {
                    // Trigger a smooth crossfade state machine sequence
                    m_BlendState.SourceZone = m_BlendState.TargetZone;
                    m_BlendState.TargetZone = *primaryZone;
                    m_BlendState.BlendAlpha = 0.0f;
                    m_BlendState.IsBlending = true;
                }
            }
        } else {
            // Progress the crossfade state machine cleanly
            m_BlendState.BlendAlpha += m_BlendState.BlendSpeed * deltaTime;
            if (m_BlendState.BlendAlpha >= 1.0f) {
                m_BlendState.BlendAlpha = 1.0f;
                m_BlendState.IsBlending = false;
            }
        }

        // 3. Resolve the Camera Transforms (Evaluate single tracking or crossfaded environments)
        CameraComponent targetEvaluatedState;
        
        if (!m_BlendState.IsBlending) {
            EvaluateZoneTransform(playerPosition, m_BlendState.TargetZone, targetEvaluatedState);
            
            // Apply standard frame-rate independent tracking lag smoothing
            float decayFactor = std::exp2(-m_BlendState.TargetZone.TrackingLag * deltaTime);
            outCamera.Position = targetEvaluatedState.Position + ((outCamera.Position - targetEvaluatedState.Position) * decayFactor);
            outCamera.Orientation = Quaternion::Slerp(outCamera.Orientation, targetEvaluatedState.Orientation, 1.0f - decayFactor);
            outCamera.FOV = targetEvaluatedState.FOV;
        } 
        else {
            // Solve BOTH spatial curves simultaneously to blend between them cleanly mid-air
            CameraComponent sourceTrackResult;
            CameraComponent targetTrackResult;
            
            EvaluateZoneTransform(playerPosition, m_BlendState.SourceZone, sourceTrackResult);
            EvaluateZoneTransform(playerPosition, m_BlendState.TargetZone, targetTrackResult);
            
            // Linear interpolate camera vectors across tracking zones
            float t = m_BlendState.BlendAlpha;
            outCamera.Position = Vector3D::Lerp(sourceTrackResult.Position, targetTrackResult.Position, t);
            outCamera.Orientation = Quaternion::Slerp(sourceTrackResult.Orientation, targetTrackResult.Orientation, t);
            outCamera.FOV = sourceTrackResult.FOV + (targetTrackResult.FOV - sourceTrackResult.FOV) * t;
        }
    }

private:
    // Evaluates a specific zone's mathematical rail or static station mapping logic
    void EvaluateZoneTransform(const Vector3D& playerPos, const CinematicZoneComponent& zone, CameraComponent& outState) const noexcept {
        outState.FOV = zone.FOV;

        if (zone.Mode == FixedCameraMode::StaticStation) {
            outState.Position = zone.RailStart;
        } 
        else if (zone.Mode == FixedCameraMode::LinearRail) {
            __m128 p = playerPos.reg;
            __m128 a = zone.RailStart.reg;
            __m128 b = zone.RailEnd.reg;

            __m128 ab = _mm_sub_ps(b, a);
            __m128 ap = _mm_sub_ps(p, a);

            float scalarTop = _mm_cvtss_f32(_mm_dp_ps(ap, ab, 0x77));
            float scalarBottom = _mm_cvtss_f32(_mm_dp_ps(ab, ab, 0x77));

            float factor = (scalarBottom > 1e-6f) ? (scalarTop / scalarBottom) : 0.0f;
            factor = std::clamp(factor, 0.0f, 1.0f);

            outState.Position = Vector3D(_mm_add_ps(a, _mm_mul_ps(_mm_set1_ps(factor), ab)));
        }

        // Standard direct LookAt resolution algorithm
        Vector3D lookDir = (playerPos - outState.Position).asDirection();
        float lengthSq = lookDir.dot(lookDir);
        if (lengthSq > 1e-6f) {
            lookDir = lookDir * (1.0f / std::sqrt(lengthSq));
            outState.Orientation = Quaternion::FromDirection(lookDir);
        }
    }

    static FORCE_INLINE bool IsPointInVolume(const Vector3D& p, const Vector3D& min, const Vector3D& max) noexcept {
        __m128 point = p.reg;
        __m128 boxMin = min.reg;
        __m128 boxMax = max.reg;
        int mask = _mm_movemask_ps(_mm_and_ps(_mm_cmpge_ps(point, boxMin), _mm_cmple_ps(point, boxMax)));
        return (mask & 0x07) == 0x07;
    }
};

/*
void Engine::UpdateCameraPipeline(float deltaTime) {
    // Step 1: Gather raw hardware inputs (Keyboard, Mouse, Gamepad)
    RawInputState rawInput = InputSystem::PollHardware();

    // Step 2: Determine if the player is in a fixed cinematic zone or standard exploration
    auto& activeZones = LevelManager::GetActiveCinematicZones(); 
    bool inCinematicZone = !activeZones.empty();

    if (inCinematicZone) {
        // Run God of War style tracking and blending
        m_CinematicSystem.Update(m_Player.Position, activeZones, m_MainCamera, deltaTime);
        
        // Translate movement vectors relative to the computed cinematic frame
        m_Player.MovementVelocity = m_InputEngine.ComputeWorldMovement(rawInput, m_MainCamera);
    } 
    else {
        // Run Tomb Raider style spring arm profiling and input orientation
        m_StateController.Update(m_PlayerCameraState, m_PlayerSpringArm, m_MainCamera, deltaTime);
        
        // Standard third-person relative input (relative to spring arm rotation)
        m_Player.MovementVelocity = m_InputEngine.ComputeWorldMovement(rawInput, m_MainCamera);
        
        // Resolve camera placement and collision checks against the environment physics
        m_SpringArmSystem.Update(m_Player.Position, m_Player.Rotation, m_PlayerSpringArm, m_MainCamera, m_PhysicsScene, deltaTime);
    }

    // Step 3: Apply final resolved transforms to character physics and view matrices
    m_PlayerPhysics.ApplyVelocity(m_Player.MovementVelocity, deltaTime);
    m_Renderer.SetViewMatrix(m_MainCamera.ComputeViewMatrix());
}
*/
