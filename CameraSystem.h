#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <print>

#include <numbers> // C++20/26 Standardized Math Constants

#include "Math.h"

#if __has_include(<inplace_vector>)
    /*
        // Replaces (std::vector). Zero heap allocations. Data is perfectly contiguous on the stack.
        // Extremely cache friendly for your SIMD wrappers.
        std::inplace_vector<Vector3D, 64> localCluster;
    */
    #include <inplace_vector> // C++26 API provides a vector that stores data locally without ever touching the heap allocator.
    #define ENGINE_HAS_CXX26_INPLACE_VECTOR 1
#else
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
    // Converted to Pure Scalar: Eliminates FPU-to-SSE shuffling penalties
    Vector3DScalar Position;
    Vector3DScalar Front;
    Vector3DScalar Up;
    Vector3DScalar Right;
    Vector3DScalar WorldUp;

    float FOV = 90.0f;
    float AspectRatio = 16.0f / 9.0f;
    float NearClip = 0.1f;
    float FarClip = 10000.0f;

    // Default initialization
    CameraComponent(Vector3DScalar startPos = Vector3DScalar(0.0f, 0.0f, 0.0f)) {
        Position = startPos;
        Front = Vector3DScalar(0.0f, 0.0f, -1.0f);
        Up = Vector3DScalar(0.0f, 1.0f, 0.0f);
        Right = Vector3DScalar(1.0f, 0.0f, 0.0f);
        WorldUp = Vector3DScalar(0.0f, 1.0f, 0.0f); // Assumes Y is up
    }

    // 100% SIMD Matrix Generation
    Matrix4x4_SIMD GetViewMatrix() const {
        // 1. Instantly hoist the aligned stack data into SSE registers
        // Unaligned load directly from the scalar's memory footprint
        // The CPU reads x, y, z, w sequentially starting from the address of 'x'
        Vector3D eyeVec(_mm_loadu_ps(&Position.x));

        // 2. Mapped operator+ in Vector3D class, this compiles down to a single _mm_add_ps instruction! SIMD Addition (target = position + front)
        Vector3D targetVec = eyeVec + Vector3D(_mm_loadu_ps(&Front.x)); 

        // FreeCamera already maintains a perfectly orthogonal Up vector
        Vector3D upVec(_mm_loadu_ps(&Up.x)); 
        
        // 3. Generate the matrix purely on the silicon
        return Matrix4x4_SIMD::LookAt_SIMD(eyeVec, targetVec, upVec);
    }

    void PrintTelemetry() const {
        // Calculate the absolute target position for telemetry (Position + Front)
        Vector3DScalar absoluteTarget = Position + Front;

        std::println("Cam Pos: [{}, {}, {}] | Target: [{}, {}, {}]", 
                     Position.x, Position.y, Position.z,
                     absoluteTarget.x, absoluteTarget.y, absoluteTarget.z);
    }
};

// --- 2. SYSTEM: FREE LOOK INPUT CONTROLLER ---
// Injects input accumulation into a generic CameraComponent. Strictly handles user input and applies mathematical deltas to any CameraComponent you hand it. 
class FreeLookController {
public:
    float Yaw = -90.0f, Pitch = 0.0f;
    float MovementSpeed = 800.0f, MouseSensitivity = 0.15f;

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

    // Replaces 'ProcessKeyboard' by using normalized vectors to move with a capped maximum speed in all directions.
    // Notice we pass the component by reference. The controller mutates the data.
    void UpdatePosition(CameraComponent& camera, const CameraInputAxes& input, float deltaTime) {
        // 1. True branchless accumulation
        // If an input axis is 0.0f, that vector component naturally zeroes out.
        Vector3DScalar moveDirection = 
            (camera.Front   * input.MoveZ) + 
            (camera.Right   * input.MoveX) + 
            (camera.WorldUp * input.MoveY);

        // 2. Prevent the "Diagonal Exploit"
        // If a player presses Forward(1) and Right(1), the vector length becomes ~1.414.
        // We must normalize the direction vector if its length exceeds 1.0.
        float squaredLength = moveDirection.dot(moveDirection);
        if (squaredLength > 1.0f) {
            float invLen = 1.0f / std::sqrt(squaredLength);
            moveDirection = moveDirection * invLen;
        }

        // 3. Apply final scaled velocity
        camera.Position = camera.Position + (moveDirection * (MovementSpeed * deltaTime));
    }

    void ProcessMouseMovement(CameraComponent& camera, float xoffset, float yoffset) {
        Yaw += (xoffset * MouseSensitivity);
        Pitch += (yoffset * MouseSensitivity);

        // Constrain pitch to prevent screen-flipping (Gimbal Lock Prevention)
        if (Pitch > 89.0f) Pitch = 89.0f;
        if (Pitch < -89.0f) Pitch = -89.0f;

        // Recompute the camera's directional axes based on the controller's spherical state
        UpdateCameraVectors(camera);
    }

private:
    // ===============================================
    // PRECISION LOSS PREVENTION 
    // ===============================================
    /*
        - Its best to prevent tiny floating-point truncation errors from compounding, causing the camera to slowly drift off its perfect axis.
        - std::numbers::pi_v<float> means the compiler uses the absolute maximum IEEE-754 floating-point precision available for the specific hardware architecture to represent pi.
        - This ensures perfect rotational stability no matter how long the player spins the camera.
    */
    void UpdateCameraVectors(CameraComponent& camera) {
        Vector3DScalar front;

        // [C++20]: Constants at maximum hardware precision (Convert to radians)
        // [std::numbers::pi_v<float>]: Prevents floating-point truncation issues during rapid camera rotations.
        float yawRad = Yaw * (std::numbers::pi_v<float> / 180.0f);
        float pitchRad = Pitch * (std::numbers::pi_v<float> / 180.0f);

        // Spherical coordinates to Cartesian coordinates
        front.x = std::cos(yawRad) * std::cos(pitchRad);
        front.y = std::sin(pitchRad);
        front.z = std::sin(yawRad) * std::cos(pitchRad);

        // Normalize Front
        float lenF = std::sqrt(front.dot(front));
        camera.Front = front * (1.0f / lenF);

        // Re-calculate Right and Up
        camera.Right = camera.Front.cross(camera.WorldUp);
        float lenR = std::sqrt(camera.Right.dot(camera.Right));
        camera.Right = camera.Right * (1.0f / lenR);

        camera.Up = camera.Right.cross(camera.Front);
        float lenU = std::sqrt(camera.Up.dot(camera.Up));
        camera.Up = camera.Up * (1.0f / lenU);
    }
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
    #if ENGINE_HAS_CXX26_INPLACE_VECTOR
        // Specify the maximum number of waypoints (e.g., 64). This pre-allocates exactly 1024 bytes (64 * 16 bytes) directly inside the class footprint.
        std::inplace_vector<Vector3DStack, 64> controlPoints;
        std::inplace_vector<Vector3DStack, 64> lookAtTargets;
    #else
        // Kept as Vector3DStack because CatmullRom uses heavy vector math operations where SSE acceleration actually benefits the 4-point polynomial evaluation.
        std::vector<Vector3DStack> controlPoints;
        std::vector<Vector3DStack> lookAtTargets;
    #endif

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
        // Overwrite Component Position via SIMD Spline
        Vector3DStack splinePos = CatmullRom(controlPoints[i0], controlPoints[i1], controlPoints[i2], controlPoints[i3], localT);
        camera.Position = Vector3DScalar(splinePos.data[0], splinePos.data[1], splinePos.data[2]);

        // 4. Overwrite Component Direction via Lerp
        // Compute Camera Rotation/Target
        // Look-at targets can usually just be linearly interpolated unless you want the camera panning to be distinctly eased.
        Vector3DStack splineTarget = Lerp(lookAtTargets[i1], lookAtTargets[i2], localT);
        Vector3DScalar targetScalar(splineTarget.data[0], splineTarget.data[1], splineTarget.data[2]);
        
        // Calculate the new Forward vector from the LookAt target
        Vector3DScalar newFront = targetScalar - camera.Position;
        float lenSq = newFront.dot(newFront);
        if (lenSq > 1e-8f) {
            newFront *= (1.0f / std::sqrt(lenSq));
            camera.Front = newFront;
        }

        // Re-orthogonalize the Right and Up vectors
        camera.Right = camera.Front.cross(camera.WorldUp);
        float lenR = std::sqrt(camera.Right.dot(camera.Right));
        camera.Right = camera.Right * (1.0f / lenR);

        camera.Up = camera.Right.cross(camera.Front);
        float lenU = std::sqrt(camera.Up.dot(camera.Up));
        camera.Up = camera.Up * (1.0f / lenU);
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
