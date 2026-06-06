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

// --- CINEMATIC CAMERA SYSTEM ---
class CinematicCamera {
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
    Vector3DStack position;
    Vector3DStack target;

    CinematicCamera() : position(0.0f, 0.0f, 0.0f), target(0.0f, 0.0f, -1.0f) {}

    // Add a physical coordinate for the camera to fly through
    void AddWaypoint(const Vector3DStack& pos, const Vector3DStack& lookAt) {
        controlPoints.push_back(pos);
        lookAtTargets.push_back(lookAt);
    }

    // Set how fast the camera completes the entire track
    void SetSpeed(float speed) { traversalSpeed = speed; }

    // Runs once per frame on the Main UI/Render Thread
    FORCE_INLINE void Update(float deltaTime) {
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
        position = CatmullRom(controlPoints[i0], controlPoints[i1], controlPoints[i2], controlPoints[i3], localT);

        // 4. Compute Camera Rotation/Target
        // Look-at targets can usually just be linearly interpolated unless you want 
        // the camera panning to be distinctly eased.
        target = Lerp(lookAtTargets[i1], lookAtTargets[i2], localT);
    }

    // Convert the SSE Vectors to Scalar to interface with the C++26 Matrix Math
    Matrix4 GetViewMatrix() const {
        return Matrix4::LookAt(
            Vector3DScalar(position.data[0], position.data[1], position.data[2]),
            Vector3DScalar(target.data[0], target.data[1], target.data[2]),
            Vector3DScalar(0.0f, 1.0f, 0.0f)
        );
    }

    // Builds the View Matrix to send to your Shader Pipeline
    // (Assuming standard math; easily swapped with GLM if you use it for matrices)
    void PrintTelemetry() const {
        std::println("Cam Pos: [{}, {}, {}] | Target: [{}, {}, {}]", 
                     position.data[0], position.data[1], position.data[2],
                     target.data[0], target.data[1], target.data[2]);
    }
};

// Strongly typed movement strictly prevents invalid input
enum class CameraMove {
    FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN
};

// --- INTERACTIVE FREE-LOOK CAMERA ---
class FreeCamera {

// ===============================================
// PRECISION LOSS PREVENTION 
// ===============================================
/*
    - Its best to prevent tiny floating-point truncation errors from compounding, causing the camera to slowly drift off its perfect axis.
    - std::numbers::pi_v<float> means the compiler uses the absolute maximum IEEE-754 floating-point precision available for the specific hardware architecture to represent pi.
    - This ensures perfect rotational stability no matter how long the player spins the camera.
*/

public:
    // Converted to Pure Scalar: Eliminates FPU-to-SSE shuffling penalties
    Vector3DScalar Position, Front, Up, Right, WorldUp;

    float Yaw = -90.0f, Pitch = 0.0f;
    float MovementSpeed = 800.0f, MouseSensitivity = 0.15f;

    FreeCamera(Vector3DScalar position = Vector3DScalar(0.0f, 200.0f, 1000.0f)) {
        Position = position;
        WorldUp = Vector3DScalar(0.0f, 1.0f, 0.0f);  // Assuming Y is up
        UpdateCameraVectors();
    }

    Matrix4 GetViewMatrix() const {
        return Matrix4::LookAt(Position, Position + Front, Up);
    }

    // WASD Movement (0=Forward, 1=Backward, 2=Left, 3=Right, 4=Up, 5=Down)
    void ProcessKeyboard(CameraMove direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;

        // ===============================================
        // BRANCHLESS JUMP TABLES (SWITCH) 
        // ===============================================
        /*
            - An if-statement used would cause the CPU branch predictor to constantly try to guess which key the user was pressing.
            - Occassionally it would guess wrong and flush its execution pipeline (a penalty of ~15 cycles).
            
            - A switch statement is optimized into a highly efficient jump table.
            - Executes the exact movement in a single instruction cycle without evaluating the other conditions.
            - Calcultes an offset based on enum integer and performs a single, direct memory jump to the correct logic.
            - Bypasses branch predictor.
        */

        #if ENGINE_HAS_CXX26_META_REFLECTION
            // Automatically prints: "Input: FORWARD" without writing a massive switch block
            std::println("Input: {}", EnumToString(direction));
        #endif

        // Branchless Jump Table via Switch
        switch (direction) {
            case CameraMove::FORWARD:  Position = Position + (Front * velocity); break;
            case CameraMove::BACKWARD: Position = Position - (Front * velocity); break;
            case CameraMove::LEFT:     Position = Position - (Right * velocity); break;
            case CameraMove::RIGHT:    Position = Position + (Right * velocity); break;
            case CameraMove::UP:       Position = Position + (WorldUp * velocity); break;
            case CameraMove::DOWN:     Position = Position - (WorldUp * velocity); break;
        }
    }

    void ProcessMouseMovement(float xoffset, float yoffset) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw += xoffset;
        Pitch += yoffset;

        // Constrain pitch to prevent screen-flipping (Gimbal Lock)
        if (Pitch > 89.0f) Pitch = 89.0f;
        if (Pitch < -89.0f) Pitch = -89.0f;

        UpdateCameraVectors();
    }

private:
    void UpdateCameraVectors() {
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
        Front = front * (1.0f / lenF);

        // Re-calculate Right and Up
        Right = Front.cross(WorldUp);
        float lenR = std::sqrt(Right.dot(Right));
        Right = Right * (1.0f / lenR);

        Up = Right.cross(Front);
        float lenU = std::sqrt(Up.dot(Up));
        Up = Up * (1.0f / lenU);
    }
};
