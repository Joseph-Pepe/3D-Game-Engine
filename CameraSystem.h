#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <print>

#include <numbers> // C++20/26 Standardized Math Constants

#include "Math.h"
#include "BVHGrid.h"
#include "GPURHI.h"

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

// ==================================================================================
// EULER ANGLES (GIMBAL LOCK) & QUATERNIONS
// ==================================================================================
/*
    - Euler Angles: Yaw, Pitch, Roll (requires pitch to be clamped to prevent screen flipping).
    - Prevents smooth interpolation and complicates multi-axis rotations (gimbal lock).

      // Constrain pitch to prevent screen-flipping (Gimbal Lock Prevention)
      if (Pitch > 89.0f) Pitch = 89.0f;
      if (Pitch < -89.0f) Pitch = -89.0f;

    - Quaternions: Represents a rotation in 3D space using a 4D complex number (q = w + xi + yi + zi).
    - Eradicates gimbal lock entirely because it represents spherical rotation s directly (no snapping or axis flips). 
    - Allows us to combine rotations using pure SIMD Fused Multiply-Add (FMA) arithmetic.
    - Maps perfectly to a 128-bit register (Hamiltonian Product) using a handful of insruction cycles. 
*/

// ==================================================================================
// SIMD QUATERNION (128-bit)
// ==================================================================================
struct alignas(16) Quaternion {
    union {
        __m128 reg;
        struct { float x, y, z, w; };
    };

    FORCE_INLINE Quaternion() : reg(_mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f)) {} // Identity {0,0,0,1}
    FORCE_INLINE Quaternion(__m128 m) : reg(m) {}
    FORCE_INLINE Quaternion(float _x, float _y, float _z, float _w) : reg(_mm_set_ps(_w, _z, _y, _x)) {}

    // --- ANGLE AXIS CONVERSION ---
    // This is the ONLY time we use Trigonometry. Used when converting mouse/keyboard input to a rotation.
    static FORCE_INLINE Quaternion AngleAxis(float angleDegrees, const Vector3D& axis) {
        float halfAngleRad = (angleDegrees * (std::numbers::pi_v<float> / 180.0f)) * 0.5f;
        float s = std::sin(halfAngleRad);
        float c = std::cos(halfAngleRad);
        
        // Multiply the normalized axis by sin(half_angle)
        __m128 sinVec = _mm_set1_ps(s);
        __m128 axisScaled = _mm_mul_ps(axis.reg, sinVec);
        
        // Blend the Cosine value into the W lane (mask 0x08 = 1000 binary)
        __m128 wCos = _mm_set_ps(c, 0.0f, 0.0f, 0.0f);
        return Quaternion(_mm_blend_ps(axisScaled, wCos, 0x08));
    }

    // --- THE HAMILTON PRODUCT (SIMD QUATERNION MULTIPLICATION) ---
    // Combines two rotations into one. Executes in ~6 clock cycles on AVX2.
    FORCE_INLINE Quaternion operator*(const Quaternion& rhs) const {
        // Q1 = this (a, b, c, d) | Q2 = rhs (x, y, z, w)
        __m128 q1 = reg;
        __m128 q2 = rhs.reg;

        // Shuffle Q1
        __m128 w1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 x1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 y1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 z1 = _mm_shuffle_ps(q1, q1, _MM_SHUFFLE(2, 2, 2, 2));

        // Shuffle Q2 for the specific Hamilton cross-terms
        __m128 tmp0 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(3, 2, 1, 0)); // w, z, y, x
        __m128 tmp1 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(2, 3, 0, 1)); // z, w, x, y
        __m128 tmp2 = _mm_shuffle_ps(q2, q2, _MM_SHUFFLE(1, 0, 3, 2)); // y, x, w, z

        // FMA (Fused Multiply-Add/Sub) sequence to resolve the complex numbers
        __m128 res = _mm_mul_ps(w1, q2);
        
        // We use bitwise XOR to flip the signs for the subtraction terms in the Hamilton formula
        __m128 signX = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0, 0x80000000));
        __m128 signY = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0x80000000));
        __m128 signZ = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0x80000000, 0x80000000, 0));

        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(x1, tmp0), signX));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(y1, tmp1), signY));
        res = _mm_add_ps(res, _mm_xor_ps(_mm_mul_ps(z1, tmp2), signZ));

        return Quaternion(res);
    }

    // --- HARDWARE NORMALIZATION ---
    FORCE_INLINE void Normalize() {
        __m128 dot = _mm_dp_ps(reg, reg, 0xFF);
        __m128 invLen = _mm_rsqrt_ps(dot); // Hardware inverse square root
        reg = _mm_mul_ps(reg, invLen);
    }

    // --- CONJUGATE (INVERSE ROTATION) ---
    // Negates X, Y, and Z. Required to generate View Matrices!
    FORCE_INLINE Quaternion Conjugate() const {
        __m128 signMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0x80000000, 0x80000000));
        return Quaternion(_mm_xor_ps(reg, signMask));
    }
    
    // --- ROTATE VECTOR ---
    // Rotates a 3D vector by this quaternion: V' = Q * V * Q^-1
    FORCE_INLINE Vector3D RotateVector(const Vector3D& v) const {
        // Fast path for rotating a vector by a quaternion
        Vector3D qVec(x, y, z, 0.0f);
        Vector3D t = qVec.cross(v) * 2.0f;
        return v + (t * w) + qVec.cross(t);
    }

    // --- DIRECTION TO QUATERNION ---
    // Converts a normalized forward vector into a rotation without using Trigonometry.
    static FORCE_INLINE Quaternion FromDirection(const Vector3D& dir) {
        Vector3D baseForward(0.0f, 0.0f, -1.0f, 0.0f); 
        
        float dot = baseForward.dot(dir);
        
        // Edge Case: The camera needs to perfectly turn around 180 degrees
        if (dot < -0.9999f) {
            return Quaternion(0.0f, 1.0f, 0.0f, 0.0f); // 180-degree Yaw
        }
        
        // Build the Quaternion using the cross product axis and the half-way dot product
        Vector3D axis = baseForward.cross(dir);
        Quaternion q(axis.x, axis.y, axis.z, 1.0f + dot);
        q.Normalize();
        
        return q;
    }

    // --- DOT PRODUCT ---
    FORCE_INLINE float dot(const Quaternion& other) const {
        __m128 res = _mm_dp_ps(reg, other.reg, 0xFF);
        return _mm_cvtss_f32(res);
    }

    // --- NORMALIZED LERP (N-Lerp) ---
    // Insanely fast. Used when the angle between quaternions is extremely small.
    // Lerp draws a straight, linear chord (a line) across a rotation sphere's interior (through the 4D sphere), causing the camera's rotational speed to accelerate and decelerate slightly between waypoints.
    static FORCE_INLINE Quaternion Lerp(const Quaternion& q1, const Quaternion& q2, float t) {
        __m128 tReg = _mm_set1_ps(t);
        __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tReg);

        // res = (q1 * (1 - t)) + (q2 * t)
        __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, oneMinusT), _mm_mul_ps(q2.reg, tReg));
        
        Quaternion result(res);
        result.Normalize();
        return result;
    }

    // --- SPHERICAL LINEAR INTERPOLATION (SLERP) ---
    // Constant velocity rotation along the shortest path of the sphere.
    // Slerp traces the curve along the surface of a sphere, guarenteeing a perfectly constant velocity for CinematicTrackController.
    static FORCE_INLINE Quaternion Slerp(const Quaternion& q1, const Quaternion& q2, float t) {
        float cosOmega = q1.dot(q2);
        __m128 q2Reg = q2.reg;

        // 1. SHORTEST PATH ENFORCEMENT
        // If the dot product is negative, the quaternions point to opposite hemispheres.
        // We flip Q2 to force the camera to take the shortest physical rotation path.
        if (cosOmega < 0.0f) {
            cosOmega = -cosOmega;
            // Flip the sign bit of all 4 floats instantly using XOR
            __m128 negZero = _mm_set1_ps(-0.0f);
            q2Reg = _mm_xor_ps(q2Reg, negZero);
        }

        // 2. GIMBAL / PRECISION FALLBACK
        // If the quaternions are nearly identical (angle is basically 0), 
        // division by sin(Omega) will cause a NaN explosion. Fallback to N-Lerp.
        if (cosOmega > 0.9999f) {
            __m128 tReg = _mm_set1_ps(t);
            __m128 oneMinusT = _mm_sub_ps(_mm_set1_ps(1.0f), tReg);
            __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, oneMinusT), _mm_mul_ps(q2Reg, tReg));
            
            Quaternion result(res);
            result.Normalize();
            return result;
        }

        // 3. THE SPHERICAL MATH
        // Extract the angle (Omega) and calculate the transcendental weights
        float omega = std::acos(cosOmega);
        float invSinOmega = 1.0f / std::sin(omega);

        float weight0 = std::sin((1.0f - t) * omega) * invSinOmega;
        float weight1 = std::sin(t * omega) * invSinOmega;

        // 4. SIMD RE-ASSEMBLY
        __m128 w0Reg = _mm_set1_ps(weight0);
        __m128 w1Reg = _mm_set1_ps(weight1);

        // res = (q1 * w0) + (q2 * w1)
        __m128 res = _mm_add_ps(_mm_mul_ps(q1.reg, w0Reg), _mm_mul_ps(q2Reg, w1Reg));

        return Quaternion(res);
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
        float x2 = invQ.x + invQ.x, y2 = invQ.y + invQ.y, z2 = invQ.z + invQ.z;
        float xx = invQ.x * x2, xy = invQ.x * y2, xz = invQ.x * z2;
        float yy = invQ.y * y2, yz = invQ.y * z2, zz = invQ.z * z2;
        float wx = invQ.w * x2, wy = invQ.w * y2, wz = invQ.w * z2;

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
                     Position.x, Position.y, Position.z,
                     absoluteTarget.x, absoluteTarget.y, absoluteTarget.z);
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
