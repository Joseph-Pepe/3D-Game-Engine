#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <print>
#include "Math.h"
// ==================================================================================
// 3D CAMERA & Catmull-Rom Spline
// ==================================================================================
/*
    - 3D camera is a single entity. 
    - Use Catmull-Rom Spline for cinematic fly-throughs, and smooth orbital paths.
    - P(t) = (1/2) (2P_1 + t(-P_0 + P_2) + t^2(2P_0 - 5P_1 + 4P_2 - P_3) + t^3(-P_0 + 3P_1 - 3P_2 + P_3)), where it evaluates 4 points (P_0, P_1, P_2, P_3)
*/

// --- CINEMATIC CAMERA SYSTEM ---
class CinematicCamera {
private:
    std::vector<Vector3DStack> controlPoints;
    std::vector<Vector3DStack> lookAtTargets;
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

    // Builds the View Matrix to send to your Shader Pipeline
    // (Assuming standard math; easily swapped with GLM if you use it for matrices)
    void PrintTelemetry() const {
        std::println("Cam Pos: [{}, {}, {}] | Target: [{}, {}, {}]", 
                     position.data[0], position.data[1], position.data[2],
                     target.data[0], target.data[1], target.data[2]);
    }
};

// --- INTERACTIVE FREE-LOOK CAMERA ---
class FreeCamera {
public:
    Vector3DStack Position, Front, Up, Right, WorldUp;
    float Yaw = -90.0f, Pitch = 0.0f;
    float MovementSpeed = 800.0f, MouseSensitivity = 0.15f;

    FreeCamera(Vector3DStack position = Vector3DStack(0.0f, 200.0f, 1000.0f)) {
        Position = position;
        WorldUp = Vector3DStack(0.0f, 1.0f, 0.0f);  // Assuming Y is up
        UpdateCameraVectors();
    }

    Matrix4 GetViewMatrix() const {
        return Matrix4::LookAt(Position, Position + Front, Up);
    }

    // WASD Movement (0=Forward, 1=Backward, 2=Left, 3=Right, 4=Up, 5=Down)
    void ProcessKeyboard(int direction, float deltaTime) {
        float velocity = MovementSpeed * deltaTime;
        if (direction == 0) Position = Position + (Front * velocity);
        if (direction == 1) Position = Position - (Front * velocity);
        if (direction == 2) Position = Position - (Right * velocity);
        if (direction == 3) Position = Position + (Right * velocity);
        if (direction == 4) Position = Position + (WorldUp * velocity); 
        if (direction == 5) Position = Position - (WorldUp * velocity); 
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
        Vector3DStack front;

        // Convert to radians
        float yawRad = Yaw * (3.14159265359f / 180.0f);
        float pitchRad = Pitch * (3.14159265359f / 180.0f);

        // Spherical coordinates to Cartesian coordinates
        front.data[0] = std::cos(yawRad) * std::cos(pitchRad);
        front.data[1] = std::sin(pitchRad);
        front.data[2] = std::sin(yawRad) * std::cos(pitchRad);

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
