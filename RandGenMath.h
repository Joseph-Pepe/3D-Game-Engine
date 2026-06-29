#pragma once

#include <immintrin.h> // AVX, SSE (128-bit), MMX (64-bit).

// ===============================================
// CRYPTOGRAPHIC RANDOM NUMBER GENERATOR & MAPPING
// ===============================================
/*
    - XorShift32 has statistical flaws, low-dimensional equidistribution errors.
    - For things like procedural generation, terrain noise, or monte carlo raytracing it will eventually produce visible banding or artifacting.

    - PCG32 (Permuted Congruential Generator) uses an LCG (Linear Congruential Generator), but applies a bitwise output permutation ot destroy the predictability.
    - Its just as fast and better than XorShift32.
*/

// --- FAST STATELESS PRNG ---
// Executes in ~1-2 clock cycles entirely inside the ALU registers.
FORCE_INLINE uint32_t XorShift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// --- FAST STATELESS PRNG (PCG32) ---
// Requires a 64-bit state, returns a perfectly distributed 32-bit random number.
FORCE_INLINE uint32_t PCG32(uint64_t& state) {
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
// Maps a random 32-bit integer to [0, range) using multiplication instead of division.
FORCE_INLINE uint32_t MapToRange(uint32_t randomVal, uint32_t range) {
    return static_cast<uint32_t>((static_cast<uint64_t>(randomVal) * static_cast<uint64_t>(range)) >> 32);
}
