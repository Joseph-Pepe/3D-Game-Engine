#pragma once

#include <print>

// --- COMPILER INTRINSICS FOR CPUID ---
#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
    #include <intrin.h> // Required for __rdtsc(), __pdep_u32, __cpuid on MSVC
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
    #include <x86intrin.h> // For GCC/Clang
#endif

// ========================================
// HARDWARE DETECTION & DYNAMIC DISPATCH
// ========================================
struct HardwareCapabilities {
    bool hasAVX = false;
    bool hasAVX2 = false;
    bool hasFMA = false;
    bool hasBMI2 = false;
    bool hasAVX512F = false;  // Foundation (The base 512-bit instructions)
    bool hasAVX512DQ = false; // Double/Quadword (Needed for certain float conversions)
    bool hasAVX512VL = false; // Vector Length Extensions (Allows using AVX-512 instructions on 256-bit registers)
    
    // The specific AMD microcode trap you already identified
    bool isLegacyAMD_BMI2 = false; 

    static HardwareCapabilities Detect() {
        HardwareCapabilities caps;
        int cpuInfo[4] = {0};

        // 1. Get Maximum Supported Function Info
        #ifdef _MSC_VER
            __cpuid(cpuInfo, 0);
        #else
            __cpuid(0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
        #endif
        int maxFunction = cpuInfo[0];

        // Check if vendor is "AuthenticAMD"
        bool isAMD = (cpuInfo[1] == 0x68747541 && cpuInfo[3] == 0x69746E65 && cpuInfo[2] == 0x444D4163);

        if (maxFunction >= 1) {
            #ifdef _MSC_VER
                __cpuid(cpuInfo, 1);
            #else
                __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
            #endif

            // OSXSAVE (Bit 27 in ECX): Does the OS know how to handle wide registers?
            bool osUsesXSAVE = (cpuInfo[2] & (1 << 27)) != 0;
            caps.hasAVX = (cpuInfo[2] & (1 << 28)) != 0;
            caps.hasFMA = (cpuInfo[2] & (1 << 12)) != 0;

            if (maxFunction >= 7) {
                #ifdef _MSC_VER
                    __cpuidex(cpuInfo, 7, 0); // Need cpuidex to pass ECX=0
                #else
                    __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
                #endif

                caps.hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0;
                caps.hasBMI2 = (cpuInfo[1] & (1 << 8)) != 0;

                // AVX-512 Checks
                if (osUsesXSAVE) {
                    caps.hasAVX512F  = (cpuInfo[1] & (1 << 16)) != 0;
                    caps.hasAVX512DQ = (cpuInfo[1] & (1 << 17)) != 0;
                    caps.hasAVX512VL = (cpuInfo[1] & (1 << 31)) != 0;
                }
            }

            // Your existing AMD Microcode check
            if (isAMD) {
                #ifdef _MSC_VER
                    __cpuid(cpuInfo, 1);
                #else
                    __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
                #endif
                int baseFamily = (cpuInfo[0] >> 8) & 0xF;
                int extendedFamily = (cpuInfo[0] >> 20) & 0xFF;
                int family = baseFamily + (baseFamily == 0xF ? extendedFamily : 0);
                
                // Zen 1, Zen+, Zen 2 (Family <= 23) have microcoded BMI2
                caps.isLegacyAMD_BMI2 = (family <= 23);
            }
        }
        return caps;
    }
};

// Global Instance: C++17 INLINE Prevents linker crashes
inline HardwareCapabilities g_Hardware = HardwareCapabilities::Detect();

// ========================================
// HARDWARE DETECTION: BMI2 MICROCODE TRAP 
// ========================================
/*
    - On Intel (Haswell and newer) and AMD Zen 3 (Ryzen 5000+), the BMI2 _pdep_u32 instruction is wired directly into a dedicated execution port in the silicon.
    - It executes in ~3 clock cycles.

    - On AMD Zen 1, Zen+, Zen 2 (Ryzen 1000, 2000, and 3000 series), its implemented in microcode instead of a dedicated circuit.
    - Executes this instruction in ~18 to 50 clock cycles to complete.
    - When processing 100,000 particles it wastes ~3 to 4.5 million clock cycles per frame on older Ryzen processors.
    - Need to detect this hardware to not use BMI2 _pdep_u32, so we can reclaim 1.0 to 1.5 milliseconds of frame time.
    - This guarantees this engine runs deterministically across all x86 architectures.
*/

inline bool detect_hardware_CPUID_BMI2() {
    int cpuInfo[4] = {0};
    
    // 1. Get Vendor String
    #ifdef _MSC_VER
        __cpuid(cpuInfo, 0);
    #else
        __cpuid(0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    #endif

    // Check if vendor is "AuthenticAMD" (ebx, edx, ecx)
    if (cpuInfo[1] != 0x68747541 || cpuInfo[3] != 0x69746E65 || cpuInfo[2] != 0x444D4163) {
        return false; // Not AMD. Intel's BMI2 is fast in silicon.
    }

    // 2. Get CPU Family
    #ifdef _MSC_VER
        __cpuid(cpuInfo, 1);
    #else
        __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
    #endif

    
    int baseFamily = (cpuInfo[0] >> 8) & 0xF;
    int extendedFamily = (cpuInfo[0] >> 20) & 0xFF;
    int family = baseFamily + (baseFamily == 0xF ? extendedFamily : 0);

    // Zen 1, Zen+, and Zen 2 belong to Family 17h (23 in decimal).
    // Zen 3 is Family 19h (25) and has fast hardware BMI2.
    // Anything 23 or lower has the slow microcoded _pdep_u32.
    return (family <= 23);
}
