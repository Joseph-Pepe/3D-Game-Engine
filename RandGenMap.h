#pragma once

#include <cstdint>
#include <bit> // For C++20 std::bit_cast

#ifndef FORCE_INLINE
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
#endif

// Prevents global namespace pollution to prevent compiler errors.
namespace Engine::Math {

    // ===============================================
    // CRYPTOGRAPHIC RANDOM NUMBER GENERATOR & MAPPING
    // ===============================================
    /*
        - XorShift32 has statistical flaws, low-dimensional equidistribution errors.
        - For things like procedural generation, terrain noise, or monte carlo raytracing it will eventually produce visible banding or artifacting.

        // --- FAST STATELESS PRNG ---
        // Executes in ~1-2 clock cycles entirely inside the ALU registers.
        FORCE_INLINE uint32_t XorShift32(uint32_t& state) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return state;
        }

        - PCG32 (Permuted Congruential Generator) uses an LCG (Linear Congruential Generator), but applies a bitwise output permutation to destroy the predictability.
        - It is just as fast and mathematically superior to XorShift32.
        - constexpr allows these functions to procedurally generate terrain seeds or noise maps at compile-time, saving runtime CPU cycles.
    */

    // --- FAST STATELESS PRNG (PCG32) ---
    // Requires a 64-bit state, returns a perfectly distributed 32-bit random number.
    [[nodiscard]] FORCE_INLINE constexpr uint32_t PCG32(uint64_t& state) {
        uint64_t oldState = state;
        
        // 1. Advance internal state (Multiplier and Increment are PCG standards)
        state = oldState * 6364136223846793005ULL + 1ULL;
        
        // 2. Calculate output function (XSH RR - Xorshift High bits, Random Rotate)
        uint32_t xorshifted = static_cast<uint32_t>(((oldState >> 18u) ^ oldState) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldState >> 59u);
        
        // 3. Bitwise rotation 
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // --- DIVISION-LESS RANGE MAPPING ---
    // Maps a random 32-bit integer to [0, range) using multiplication instead of modulo.
    [[nodiscard]] FORCE_INLINE constexpr uint32_t MapToRange(uint32_t randomVal, uint32_t range) {
        // Extremely fast: Replaces a 15-cycle modulo operation with a 2-cycle multiply-shift.
        return static_cast<uint32_t>((static_cast<uint64_t>(randomVal) * static_cast<uint64_t>(range)) >> 32);
    }

    // --- FAST IEEE-754 FLOAT MAPPING [0.0, 1.0) ---
    // Takes a raw 32-bit random integer and constructs a float directly in the silicon.
    // Bypasses slow division (e.g., randomVal / MAX_UINT).
    [[nodiscard]] FORCE_INLINE constexpr float RandomFloat(uint32_t randomVal) noexcept {
        // 1. Take the top 23 bits of the random number (Mantissa precision for float32)
        // 2. Bitwise OR it with the IEEE-754 exponent for 1.0f (0x3F800000)
        uint32_t floatBits = (randomVal >> 9) | 0x3F800000;

        // 3. Cast the raw bits into a float (results in a range of [1.0f, 2.0f))
        // 4. Subtract 1.0f to shift the range to [0.0f, 1.0f)
        return std::bit_cast<float>(floatBits) - 1.0f;
    }

    // --- FAST FLOAT RANGE MAPPING [min, max) ---
    [[nodiscard]] FORCE_INLINE constexpr float RandomFloatRange(uint64_t& state, float min, float max) noexcept {
        uint32_t randVal = PCG32(state);
        float normalized = RandomFloat(randVal);
        return min + normalized * (max - min);
    }
}
