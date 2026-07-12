#pragma once

#include <print>
#include <immintrin.h> // Required for _xgetbv() and universal SIMD support

// --- COMPILER INTRINSICS FOR CPUID (x86_64 ONLY) ---
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h> // Required for _xgetbv() and universal SIMD support
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
        #include <intrin.h> // Required for __rdtsc(), __pdep_u32, __cpuid on MSVC
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
        #include <x86intrin.h> // For GCC/Clang
    #endif
    #define ENGINE_IS_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    // ARM architecture doesn't use CPUID or xgetbv
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
    #define ENGINE_IS_ARM 1
#else
    #error "Unsupported Architecture"
#endif

// C++26 Static Reflections to have the compiler automatically loop over the variables in the struct and generate print statements (i.e., scales automatically when new boolean fields are added in the struct).
#if __has_include(<meta>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    #include <meta>
    #define ENGINE_HAS_CXX26_META_REFLECTION 1
#else
    #define ENGINE_HAS_CXX26_META_REFLECTION 0
#endif

// ===========================================================
// HARDWARE DETECTION & DYNAMIC DISPATCH : BMI2 MICROCODE TRAP 
// ===========================================================
/*
    - We need to handle the runtime routing of execution paths based on the exact silicon the user works on.
    - Need to verify the Operating System is safe, detect silicon microcode traps, and route traffic to the correct assembly at runtime.
    - On Intel (Haswell and newer) and AMD Zen 3 (Ryzen 5000+), the BMI2 _pdep_u32 instruction is wired directly into a dedicated execution port in the silicon.
    - It executes in ~3 clock cycles.

    - On AMD Zen 1, Zen+, Zen 2 (Ryzen 1000, 2000, and 3000 series), its implemented in microcode (i.e., emulates _pdep_u32) instead of a dedicated circuit.
    - Executes this instruction in ~18 to 50 clock cycles to complete.
    - When processing 100,000 particles it wastes ~3 to 4.5 million clock cycles per frame on older Ryzen processors.
    - Need to detect this hardware to not use BMI2 _pdep_u32, so we can reclaim 1.0 to 1.5 milliseconds of frame time.
    - This guarantees this engine runs deterministically across all x86 architectures.
*/

struct HardwareCapabilities {
    bool hasAVX = false;
    bool hasAVX2 = false;
    bool hasFMA = false;
    bool hasBMI2 = false;
    bool hasAVX512F = false;  // Foundation (The base 512-bit instructions)
    bool hasAVX512DQ = false; // Double/Quadword (Needed for certain float conversions)
    bool hasAVX512VL = false; // Vector Length Extensions (Allows using AVX-512 instructions on 256-bit registers)

    bool hasNEON = false;

    bool isLegacyAMD_BMI2 = false;  // The specific AMD microcode trap identified

    static HardwareCapabilities Detect() {
        HardwareCapabilities caps;

        #if defined(ENGINE_IS_X86)
            int cpuInfo[4] = {0};

            // 1. Get Maximum Supported Function Info (Get Vendor String)
            #ifdef _MSC_VER
                __cpuid(cpuInfo, 0);
            #else
                __cpuid(0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
            #endif
            int maxFunction = cpuInfo[0];

            // Check if vendor is "AuthenticAMD" (ebx, edx, ecx), [False = Not AMD. Intel's BMI2 is fast in silicon]
            bool isAMD = (cpuInfo[1] == 0x68747541 && cpuInfo[3] == 0x69746E65 && cpuInfo[2] == 0x444D4163);

            if (maxFunction >= 1) {
                #ifdef _MSC_VER
                    __cpuid(cpuInfo, 1);
                #else
                    __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
                #endif

                // OSXSAVE (Bit 27 in ECX): Does the OS know how to handle wide registers?
                bool osUsesXSAVE = (cpuInfo[2] & (1 << 27)) != 0;
                bool osSavesYMM = false;
                bool osSavesZMM = false;

                // Ask the OS if it is actually saving the registers
                if (osUsesXSAVE) {
                    // Read the Extended Control Register (XCR0)
                    unsigned long long xcrFeatureMask = _xgetbv(0);
                    
                    // [C++14/26 Modernization]: Use binary literals instead of Hex for bitmask clarity
                    // Bit 1 = XMM (128-bit), Bit 2 = YMM (256-bit). Mask: 0000_0110
                    osSavesYMM = (xcrFeatureMask & 0b00000110) == 0b00000110; 
                    
                    // Bits 5, 6, 7 = OPMASK and ZMM (512-bit). Mask: 1110_0110
                    osSavesZMM = (xcrFeatureMask & 0b11100110) == 0b11100110;
                }

                // CPU supports AVX natively
                bool cpuHasAVX = (cpuInfo[2] & (1 << 28)) != 0;
                caps.hasFMA = (cpuInfo[2] & (1 << 12)) != 0;

                // Only enable AVX if BOTH the CPU and the OS support it
                caps.hasAVX = cpuHasAVX && osSavesYMM;

                if (maxFunction >= 7) {
                    #ifdef _MSC_VER
                        __cpuidex(cpuInfo, 7, 0); // Need cpuidex to pass ECX=0
                    #else
                        __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
                    #endif

                    caps.hasAVX2 = (cpuInfo[1] & (1 << 5)) != 0 && osSavesYMM;
                    caps.hasBMI2 = (cpuInfo[1] & (1 << 8)) != 0;

                    // Only enable AVX-512 if the OS saves ZMM state
                    if (osSavesZMM) {
                        caps.hasAVX512F  = (cpuInfo[1] & (1 << 16)) != 0;
                        caps.hasAVX512DQ = (cpuInfo[1] & (1 << 17)) != 0;
                        caps.hasAVX512VL = (cpuInfo[1] & (1 << 31)) != 0;
                    }
                }

                // AMD Microcode check for slow BMI2
                if (isAMD && caps.hasBMI2) {
                    // 2. Get CPU Family
                    #ifdef _MSC_VER
                        __cpuid(cpuInfo, 1);
                    #else
                        __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
                    #endif
                    int baseFamily = (cpuInfo[0] >> 8) & 0xF;
                    int extendedFamily = (cpuInfo[0] >> 20) & 0xFF;
                    int family = baseFamily + (baseFamily == 0xF ? extendedFamily : 0);
                    
                    // ===============================================
                    // HARDWARE | MICROCODED BMI2 DETECTION 
                    // ===============================================
                    /*
                        - Zen 1, Zen+, and Zen 2 belong to Family 17h (23 in decimal)
                        - Zen 1, Zen+, Zen 2 (Family <= 23) have microcoded BMI2
                        - Zen 3 is Family 19h (25) and has fast hardware BMI2.
                        - Anything 23 or lower has the slow microcoded _pdep_u32.
                    */
                    caps.isLegacyAMD_BMI2 = (family <= 23);
                }
            }
        #elif defined(ENGINE_IS_ARM)
            // ARM NEON is universally guaranteed on AArch64 systems (we do not need to query the OS or trap microcode bugs).
            caps.hasNEON = true;
        #endif
        return caps;
    }

    // --- C++26 REFLECTIVE TELEMETRY DUMP ---
    void PrintTelemetry() const {
        std::println("=== HARDWARE CAPABILITIES ===");
        
        #if ENGINE_HAS_CXX26_META_REFLECTION
            // C++26 Reflection: The compiler inspects the struct and generates the print statements automatically!
            // '^^' gets the meta-info of the struct. '[: :]' splices it back into executable code.
            [: expand(std::meta::nonstatic_data_members_of(^^HardwareCapabilities)) :] >> [&]<auto m>{
                std::println("{:<20}: {}", std::meta::identifier_of(m), this->[:m:] ? "YES" : "NO");
            };
        #else
            // C++23 Fallback
            std::println("AVX:                {}", hasAVX ? "YES" : "NO");
            std::println("AVX2:               {}", hasAVX2 ? "YES" : "NO");
            std::println("FMA:                {}", hasFMA ? "YES" : "NO");
            std::println("AVX-512 F:          {}", hasAVX512F ? "YES" : "NO");
            std::println("Legacy AMD BMI2:    {}", isLegacyAMD_BMI2 ? "YES (Warning: Microcode Trap)" : "NO");
        #endif
        
        std::println("=============================");
    }
};

// Global Instance: C++17 INLINE Prevents linker crashes
inline HardwareCapabilities g_Hardware = HardwareCapabilities::Detect();

// ==================================================================================
// DYNAMIC DISPATCH (RUNTIME HARDWARE ROUTING)
// ==================================================================================
/*
    - Evaluates the global HardwareCapabilities generated at startup by Hardware.h.
    - Routes the execution path to the fastest available instruction set.
    - Prevents "Illegal Instruction" crashes on legacy CPUs while maximizing performance on modern ones.
*/

// Template Driven Dynamic Dispatch: Full-System Multi-Versioning via Static Polymorphism.
namespace Engine::Physics {
    template <typename Abi>
    void UpdateParticlesTemplate(float* xs, float* ys, float* zs, size_t count, float deltaTime) {
        // Dynamically instantiate the exact SIMD width for this ABI
        using WideFloat = Engine::ISAArch::simd<float, Abi>;
        constexpr size_t stride = WideFloat::size();

        // Broadcast scalar to vector natively
        WideFloat dtBatch(deltaTime);

        // ... physics loop ...
    }
}

namespace Engine::GameEngine {
    // 1. The Generic Templated Physics Kernel, parameterized by 'Abi' (Application Binary Interface). The compiler will generate 5 different hardware versions.
    template <typename Abi>
    void UpdateEngineSubsystems(float* xs, float* ys, float* zs, size_t count, float deltaTime) {
        // 1. Run Physics (It uses the ABI)
        Engine::Physics::UpdateParticlesTemplate<Abi>(xs, ys, zs, count, deltaTime);

        // 2. Run Audio Mixing (It uses the ABI)
        Engine::Audio::MixTracks<Abi>(count);

        // 3. Run Culling (It uses the ABI)
        Engine::Rendering::CullFrustum<Abi>(count);
    }

    // 2. Define the type signature for our hardware math kernels (C++11)
    using GameEngineBackendFn = void(*)(float* xs, float* ys, float* zs, size_t count, float deltaTime);

    // 3. The active target backend function pointer initialized securely to a safe scalar fallback (allows the Engine to swap the backend based on the silicon at runtime).
    inline GameEngineBackendFn ExecuteGameEngineBackend = nullptr; // void (*ExecuteGameEngineBackend)(float*, float*, float*, size_t, float)
    
    // 4. The Dispatch Initializer called once during Engine Boot.
    void InitializeDynamicDispatch() {
        // Runs compile-time + runtime dynamic dispatch exactly once.
        #if defined(ENGINE_ARCH_AVX512)
            if (g_Hardware.hasAVX512F) {
                // Route to the AVX-512 instantiation of our template!
                ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::avx512>;
                std::println("[ROUTING] AVX-512 Backend Selected.");
                return;
            }
        #endif

        #if defined(ENGINE_ARCH_AVX2)
            if (g_Hardware.hasAVX2) {
                // Route to the AVX2 instantiation of our template!
                ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::avx2>;
                std::println("[ROUTING] AVX2 Backend Selected.");
                return;
            }
        #endif

        #if defined(ENGINE_ARCH_NEON)
            if (g_Hardware.hasNEON) {
                // Route to the ARM NEON instantiation of our template!
                ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::neon>;
                std::println("[ROUTING] ARM NEON Backend Selected.");
                return;
            }
        #endif

        #if defined(ENGINE_ARCH_SSE41)
            // Route to the SSE4.1 instantiation of our template!
            ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::sse41>;
            std::println("[ROUTING] SSE4.1 Legacy Backend Selected.");
            return;
        #endif

        // Absolute  baseline fallback
        ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::scalar>;
        std::println("[ROUTING] SCALAR Legacy Backend Selected.");
    }
}


/*
void EngineTick(float deltaTime) {
    // ... game logic ...
    
    // Jump directly to the optimized machine code for the host hardware. Zero branching!
    Engine::GameEngine::ExecuteGameEngineBackend(xs, ys, zs, count, deltaTime);
}
*/
