#pragma once

// --- ENGINE SYSTEM (CPU Control, Cache Locality, Dependency Management) ---
struct EngineSettings {
    int activeParticles = 100000;
    float tangentialSpeed = 8.0f;
    float gravityStrength = 5.0f;
    bool triggerRespawn = false; // Flag to reset the galaxy

    // --- Dimension Toggle ---
    bool is2DMode = true; 
    bool modeChanged = false; // Flag to tell the engine to squash/expand the Z-axis

    // --- Hardware Legacy Architectures ---
    bool isLegacyCPU = false; 
};

// EXTERN: Tells every file that includes this header that 'g_EngineSettings' exists, but prevents them from allocating their own duplicate copy of it.
// Trust that its being allocated in a different file.
extern EngineSettings g_EngineSettings;
