#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <print>

// --- MATH UTILITIES FOR CAMERA PATHING ---

FORCE_INLINE Vector3DStack Lerp(const Vector3DStack& a, const Vector3DStack& b, float t) {
    return a + ((b - a) * t);
}

FORCE_INLINE Vector3DStack CatmullRom(const Vector3DStack& p0, const Vector3DStack& p1, 
                                      const Vector3DStack& p2, const Vector3DStack& p3, float t) {
    float t2 = t * t;
    float t3 = t2 * t;

    Vector3DStack v0 = p1 * 2.0f;
    Vector3DStack v1 = (p2 - p0) * t;
    Vector3DStack v2 = (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2;
    Vector3DStack v3 = (p1 * 3.0f - p0 - p2 * 3.0f + p3) * t3;

    return (v0 + v1 + v2 + v3) * 0.5f;
}

// ==================================================================================
// 3D CAMERA & MATRIX MATH
// ==================================================================================

// --- 4x4 MATRIX MATH (Stack Allocated, Column-Major for OpenGL) ---
struct Matrix4 {
    float m[16] = {0}; // Initializes to all zeros

    static Matrix4 Identity() {
        Matrix4 mat;
        mat.m[0] = 1.0f; mat.m[5] = 1.0f; mat.m[10] = 1.0f; mat.m[15] = 1.0f;
        return mat;
    }

    static Matrix4 Perspective(float fovY_degrees, float aspect, float nearZ, float farZ) {
        Matrix4 mat;
        float fovY_rad = fovY_degrees * (3.14159265359f / 180.0f);
        float tanHalfFovY = std::tan(fovY_rad / 2.0f);

        mat.m[0] = 1.0f / (aspect * tanHalfFovY);
        mat.m[5] = 1.0f / tanHalfFovY;
        mat.m[10] = -(farZ + nearZ) / (farZ - nearZ);
        mat.m[11] = -1.0f;
        mat.m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
        return mat;
    }

    static Matrix4 LookAt(const Vector3DStack& eye, const Vector3DStack& target, const Vector3DStack& upVec) {
        Vector3DStack f = target - eye;
        float fLen = std::sqrt(f.dot(f));
        f = f * (1.0f / fLen);

        Vector3DStack r = f.cross(upVec);
        float rLen = std::sqrt(r.dot(r));
        r = r * (1.0f / rLen);

        Vector3DStack u = r.cross(f);

        Matrix4 mat = Identity();
        mat.m[0] = r.data[0];  mat.m[4] = r.data[1];  mat.m[8] = r.data[2];
        mat.m[1] = u.data[0];  mat.m[5] = u.data[1];  mat.m[9] = u.data[2];
        mat.m[2] = -f.data[0]; mat.m[6] = -f.data[1]; mat.m[10] = -f.data[2];

        mat.m[12] = -r.dot(eye);
        mat.m[13] = -u.dot(eye);
        mat.m[14] = f.dot(eye);
        return mat;
    }
};

// --- CINEMATIC CAMERA SYSTEM ---
class CinematicCamera {
private:
    std::vector<Vector3DStack> controlPoints;
    std::vector<Vector3DStack> lookAtTargets;
    float currentProgress = 0.0f; 
    float traversalSpeed = 0.1f;  

public:
    Vector3DStack position;
    Vector3DStack target;

    CinematicCamera() : position(0.0f, 0.0f, 0.0f), target(0.0f, 0.0f, -1.0f) {}

    void AddWaypoint(const Vector3DStack& pos, const Vector3DStack& lookAt) {
        controlPoints.push_back(pos);
        lookAtTargets.push_back(lookAt);
    }

    void SetSpeed(float speed) { traversalSpeed = speed; }

    FORCE_INLINE void Update(float deltaTime) {
        size_t count = controlPoints.size();
        if (count < 2) return; 

        currentProgress += traversalSpeed * deltaTime;
        if (currentProgress > 1.0f) currentProgress = 1.0f; 

        float segmentCount = static_cast<float>(count - 1);
        float scaledProgress = currentProgress * segmentCount;
        int currentIndex = static_cast<int>(scaledProgress);
        float localT = scaledProgress - static_cast<float>(currentIndex);

        int i0 = std::max(0, currentIndex - 1);
        int i1 = currentIndex;
        int i2 = std::min(static_cast<int>(count - 1), currentIndex + 1);
        int i3 = std::min(static_cast<int>(count - 1), currentIndex + 2);

        position = CatmullRom(controlPoints[i0], controlPoints[i1], controlPoints[i2], controlPoints[i3], localT);
        target = Lerp(lookAtTargets[i1], lookAtTargets[i2], localT);
    }

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
        WorldUp = Vector3DStack(0.0f, 1.0f, 0.0f); 
        UpdateCameraVectors();
    }

    Matrix4 GetViewMatrix() const {
        return Matrix4::LookAt(Position, Position + Front, Up);
    }

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

        if (Pitch > 89.0f) Pitch = 89.0f;
        if (Pitch < -89.0f) Pitch = -89.0f;

        UpdateCameraVectors();
    }

private:
    void UpdateCameraVectors() {
        Vector3DStack front;
        float yawRad = Yaw * (3.14159265359f / 180.0f);
        float pitchRad = Pitch * (3.14159265359f / 180.0f);

        front.data[0] = std::cos(yawRad) * std::cos(pitchRad);
        front.data[1] = std::sin(pitchRad);
        front.data[2] = std::sin(yawRad) * std::cos(pitchRad);
        
        float lenF = std::sqrt(front.dot(front));
        Front = front * (1.0f / lenF);

        Right = Front.cross(WorldUp);
        float lenR = std::sqrt(Right.dot(Right));
        Right = Right * (1.0f / lenR);

        Up = Right.cross(Front);
        float lenU = std::sqrt(Up.dot(Up));
        Up = Up * (1.0f / lenU);
    }
};
