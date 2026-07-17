#pragma once

#include <print>
#include <span>

// #include "SIMD/SIMDCustomWrapper.h" 
#include "ParticleSystem.h"
#include "FiberjobSystem/JobSystem.h"

// --- COMPILER INTRINSICS FOR CPUID (x86_64 ONLY) ---
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h> // Required for _xgetbv() and universal SIMD support
    #ifdef _MSC_VER
        #define FORCE_INLINE __forceinline
        #include <intrin.h> // Required for __rdtsc(), __pdep_u32, __cpuid on MSVC
    #else
        #define FORCE_INLINE inline __attribute__((always_inline))
        #include <x86intrin.h> // For GCC/Clang
        #include <cpuid.h>     // Required for __cpuid and __cpuid_count
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
            // --- Hardware Auto-Detect ---
            std::println("AVX:                {}", hasAVX ? "YES" : "NO");
            std::println("AVX2:               {}", hasAVX2 ? "YES" : "NO");
            std::println("FMA:                {}", hasFMA ? "YES" : "NO");
            std::println("AVX-512 F:          {}", hasAVX512F ? "YES" : "NO");
            std::println("BMI2:               {}", hasBMI2 ? "YES" : "NO");
            std::println("Legacy AMD BMI2:    {}", isLegacyAMD_BMI2 ? "YES (BMI2 Microcoded)" : "NO (BMI2 CPU Detected)");
        #endif

        // g_EngineSettings.isLegacyCPU = g_Hardware.isLegacyAMD_BMI2;
        // if (g_EngineSettings.isLegacyCPU) {
        //     std::println("[WARNING] Slow BMI2 CPU detected. Forcing L1 Cache LUT fallback.");
        // }
        
        std::println("=============================");
    }
};

// Global Instance: C++17 INLINE Prevents linker crashes
inline HardwareCapabilities g_Hardware = HardwareCapabilities::Detect();

// ===================================================
// --- DYNAMIC DISPATCH (RUNTIME HARDWARE ROUTING) ---
// ===================================================
/*
    - Evaluates the global HardwareCapabilities generated at startup by Hardware.h.
    - Routes the execution path to the fastest available instruction set.
    - Prevents "Illegal Instruction" crashes on legacy CPUs while maximizing performance on modern ones.
*/

// Dynamic Dispatch: Full-System Multi-Versioning via Static Polymorphism.
namespace Engine::GameEngine {

    constexpr size_t stride = Engine::Physics::NativeFloatSIMDBatch::size();

    // Direct, compile-time linked subsystems update! SIMDVector3D ensures the memory layout is baked into the binary at compile-time (e.g., cannot pass a 32-bytes wide (AVX2) SIMDVector3D into SSE4.1 (16-bytes wide)).
    FORCE_INLINE void ExecuteGameEngineBackend(std::span<Engine::Physics::SIMDVector3D> pos, 
                                             std::span<Engine::Physics::SIMDVector3D> vel, 
                                            // std::span<Engine::Physics::SIMDVector3D> accVel,
                                             size_t count, float deltaTime, float gravityVal, 
                                             float mouseX, float mouseY, bool isMouseDown, 
                                             const Engine::Physics::ParticleSystem::ParticleHash* sortedHashes) {

        // =================================================
        // THREAD-LEVEL PARALLELISM & DATA-LEVEL PARALLELISM
        // =================================================
        /*
            - TLP (High-Level): Is handled by the job system, it slices a massive array (e.g., 100,000 particles) into a few large chunks and assigns those chunks to a thread pool (i.e., Outside Orchestrator).
            - DLP  (Low-Level): Handled by SIMD (AVX2/NEON), it takes a single chunk assigned to a thread and process 8 or 16 floats simultaneously inside the CPU registers (i.e., Inside Math Kernel). 
            - Threading and dispatching belongs strictly to the higher-level Engine Core or Game Loop, not buried in low-level kernels.
            - We use the JobSystem as an outer wrapper that decides which thread gets to run the sequential kernel.
            - Must dispatch two separate times to create a thread barrier (Sync Point).

              1. All threads calculate collisions and write to vel (Barrier: wait all to finish).
              2. All threads integrate and write to.
        */

        // A. Calculate how many SIMD batches we need to process (e.g., stride: 8 particles at a time (AVX2))
        size_t totalBatches = (count + stride - 1uz) / stride; 
        
        // B. Calculate a fair chunk size for the OS threads to prevent queue contention
        uint32_t threadCount = g_JobSystem.nextWorkerId.load(std::memory_order_relaxed);

        // Ensures the chunks stay large enough to exploit spatial locality inside each hardware core's L1/L2 caches. Ensure we don't dispatch 0 sized chunks if we have very few particles.
        uint32_t chunkSize = std::max(1u, static_cast<uint32_t>(totalBatches / (threadCount/* * 2 */)));

        // ========================================================
        // PHASE 1: NARROW PHASE COLLISIONS (READS POS, WRITES VEL)
        // ========================================================
        // 1. Calculate collisions entirely into velocity blocks first
        if (sortedHashes != nullptr) {
            // -- HIGH-LEVEL ORCHESTRATION (Outside the Math) --
            // 2. Dispatch to the Job System! The main thread drops a job into the queue. The worker threads wake up.
            g_JobSystem.DispatchAndWait(totalBatches, chunkSize, [&](uint32_t start, uint32_t end) {
                // --- SIMD MATH KERNEL (Radio Silence) -- 
                // 3. Threads process their assigned chunks simultaneously. This is the hot path. A single worker thread enters this loop and executes pure register math. 
                Engine::Physics::SolveCollisions(pos, vel, sortedHashes, count, start, end);
            });
        }

        // ============================================
        // PHASE 2: INTEGRATION (READS VEL, WRITES POS)
        // ============================================
        // 2. ONLY THEN integrate velocity data forward into positions
        g_JobSystem.DispatchAndWait(totalBatches, chunkSize, [&](uint32_t start, uint32_t end) {
            Engine::Physics::IntegrateParticles(pos, vel, count, deltaTime, gravityVal, mouseX, mouseY, isMouseDown, start, end);
        });
    }

    // Direct benchmark mapping
    FORCE_INLINE void ExecuteBenchmark(std::span<Engine::Physics::SIMDVector3D> pos, 
                                       std::span<Engine::Physics::SIMDVector3D> vel, 
                                       size_t count, float deltaTime, float gravityVal, int64_t repeats) {

        // Benchmarks run on a single dedicated background thread, so we pass 0 to totalBatches
        size_t totalBatches = (count + stride - 1uz) / stride;

        Engine::Physics::BenchmarkParticles(pos, vel, count, deltaTime, gravityVal, repeats, 0, totalBatches);
    }

    // Template Driven Dynamic Dispatch: Full-System Multi-Versioning via Static Polymorphism & dynamic function pointers (Used when using floats instead of SIMDVector3D).
    // // 1. The Generic Templated Physics Kernel, parameterized by 'Abi' (Application Binary Interface). The compiler will generate 5 different hardware versions.
    // //  template <typename Abi>
    // void UpdateEngineSubsystems(float* xs, float* ys, float* zs*, sfloat* vx, float* vy, float* vz, size_t count, float deltaTime, float gravityVal, float mouseX, float mouseY, bool isMouseDown, const uint32_t* cellStartOffsets, const uint32_t* sortedIndices) {

    //     // 1. Morton Sort ...
    //     // Let the Physics Module handle the entire simulation pipeline (Sort -> Collide -> Integrate)
    //     // Engine::Physics::EngineTick(livePhysics, deltaTime /*, mouseWorldX, mouseWorldY, applyMouseForce*/);
    //     // Engine::Physics::SortParticles(pos, ...);

    //     // 2. Collisions (SAFETY CHECK: Only run if the grid actually exists!)
    //     if (cellStartOffsets != nullptr && sortedIndices != nullptr) Engine::Physics::SolveCollisionsTemplate(pos, vel, cellStartOffsets, sortedIndices, count);

    //     // 3. Run Physics (It uses the ABI)
    //     Engine::Physics::IntegrateParticlesTemplate<Abi>(xs, ys, zs, vx, vy, vz, count, deltaTime, gravityVal, mouseX, mouseY, isMouseDown);

    //     // 4. Run Audio Mixing (It uses the ABI)
    //     // Engine::Audio::MixTracks<Abi>(count);

    //     // 5. Run Culling (It uses the ABI)
    //     // Engine::Rendering::CullFrustum<Abi>(count);
    // }

    // // 2. C++11: Define the type signature for our hardware math kernels. 
    // using GameEngineBackendFn = void(*)(float* xs,    float* ys,       float* zs*, 
    //                                     *float* vx,    float* vy,       float* vz*, 
    //                                     size_t count, float deltaTime, float gravityVal, 
    //                                     float mouseX, float mouseY,    bool isMouseDown, 
    //                                     const uint32_t* cellStartOffsets, 
    //                                     const uint32_t* sortedIndices);

    // // 3. The active target backend function pointer initialized securely to a safe scalar fallback (allows the Engine to swap the backend based on the silicon at runtime).                              
    // inline GameEngineBackendFn ExecuteGameEngineBackend = nullptr; // void (*ExecuteGameEngineBackend)(float*, float*, float*, size_t, float, ...)

    // // Benchmark  Signature
    // using BenchmarkBackendFn = void(*)(float* xs, float* ys, float* zs, float* vx, float* vy, float* vz, size_t count, float deltaTime, float gravityVal, int64_t repeats);
    // inline BenchmarkBackendFn ExecuteBenchmarkBackend = nullptr;
    
    // // 4. The Dispatch Initializer called once during Engine Boot.
    // void InitializeDynamicDispatch() {
    //     // Runs compile-time + runtime dynamic dispatch exactly once.
    //     #if defined(ENGINE_ARCH_AVX512)
    //         if (g_Hardware.hasAVX512F) {
    //             // Route to the AVX-512 instantiation of our template! Enables us to execute the function in main () like this [Engine::GameEngine::ExecuteBenchmarkBackend(...)]
    //             ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::avx512>;
    //             ExecuteBenchmarkBackend  = &Engine::Physics::BenchmarkParticlesTemplate<Engine::ISAArch::simd_abi::avx512>;
    //             std::println("[ROUTING] AVX-512 Backend Selected.");
    //             return;
    //         }
    //     #endif

    //     #if defined(ENGINE_ARCH_AVX2)
    //         if (g_Hardware.hasAVX2) {
    //             // Route to the AVX2 instantiation of our template!
    //             ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::avx2>;
    //             ExecuteBenchmarkBackend  = &Engine::Physics::BenchmarkParticlesTemplate<Engine::ISAArch::simd_abi::avx2>;
    //             std::println("[ROUTING] AVX2 Backend Selected.");
    //             return;
    //         }
    //     #endif

    //     #if defined(ENGINE_ARCH_NEON)
    //         if (g_Hardware.hasNEON) {
    //             // Route to the ARM NEON instantiation of our template!
    //             ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::neon>;
    //             ExecuteBenchmarkBackend  = &Engine::Physics::BenchmarkParticlesTemplate<Engine::ISAArch::simd_abi::neon>;
    //             std::println("[ROUTING] ARM NEON Backend Selected.");
    //             return;
    //         }
    //     #endif

    //     #if defined(ENGINE_ARCH_SSE41)
    //         // Route to the SSE4.1 instantiation of our template!
    //         ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::sse41>;
    //         ExecuteBenchmarkBackend  = &Engine::Physics::BenchmarkParticlesTemplate<Engine::ISAArch::simd_abi::sse41>;
    //         std::println("[ROUTING] SSE4.1 Legacy Backend Selected.");
    //         return;
    //     #endif

    //     // Absolute baseline fallback
    //     ExecuteGameEngineBackend = &UpdateEngineSubsystems<Engine::ISAArch::simd_abi::scalar>;
    //     ExecuteBenchmarkBackend  = &Engine::Physics::BenchmarkParticlesTemplate<Engine::ISAArch::simd_abi::scalar>;
    //     std::println("[ROUTING] SCALAR Legacy Backend Selected.");
    // }
}

// ==================================================================================
// MAIN GAME BOOTSTRAPPER (main.cpp)
// ==================================================================================

/*
void EngineTick(float deltaTime) {
    // ... game logic ...
    
    // Jump directly to the optimized machine code for the host hardware. Zero branching!
    Engine::GameEngine::ExecuteGameEngineBackend(xs, ys, zs, count, deltaTime);
}

// The absolute entry point from the Operating System that acts as an engine bootstrapper (i.e., wakes up the hardware, allocates memory arenas, boots the subsystems in a strictly controlled order, then hands control over to the game loop).
int main(int argc, char** argv) {
    
    // =================================================================
    // PHASE 1: SILICON & CORE WAKE-UP (Strictly Single-Threaded)
    // =================================================================
    
    // 1. Output the telemetry you built so we have a log of the host machine
    g_Hardware.PrintTelemetry();

    // 2. Lock in the dynamic dispatch pointers! This MUST happen before any worker threads are spawned.
    Engine::GameEngine::InitializeDynamicDispatch();

    // 3. Boot the Global Memory Arenas! Grab large contiguous chunks of RAM from the OS upfront.
    Engine::Memory::InitializeGlobalArenas();

    // =================================================================
    // PHASE 2: SUBSYSTEM BOOTSTRAPPING
    // =================================================================
    
    // Wake up the worker threads (e.g., spawn 15 threads on a 16-core CPU). They will instantly go to sleep waiting for jobs.
    std::println("[INIT] Booting Job System...");
    g_JobSystem.Initialize(); 

    // Boot Platform/OS specific layers
    Engine::Window::Create("Trinity Engine", 1920, 1080);
    Engine::Audio::Initialize();
    
    // Boot the GPU (Vulkan / DirectX 12)
    Engine::Renderer::Initialize();

    // =================================================================
    // PHASE 3: MOUNT GAME DATA
    // =================================================================
    // Load the initial level, compile shaders, load textures into VRAM.
    Engine::SceneManager::LoadScene("level_01.map");

    // =================================================================
    // PHASE 4: INFINITE GAME LOOP (The Heartbeat)
    // =================================================================
    Engine::Clock timer;

    while (Engine::Window::ShouldClose() == false) {
        float deltaTime = timer.GetDeltaTime();
        timer.Reset();

        // 1. Poll OS Events (Mouse, Keyboard, Window Resize)
        Engine::Window::PollEvents();

        // 2. Execute the dynamically routed Game Engine subsystems!
        // This jumps directly to your AVX-512, AVX2, or SSE compiled templates.
        Engine::GameEngine::ExecuteGameEngineBackend(
            globalPosX, globalPosY, globalPosZ, 
            activeEntityCount, 
            deltaTime
        );

        // 3. Submit command buffers to the GPU
        Engine::Renderer::RenderFrame();
    }

    // =================================================================
    // PHASE 5: GRACEFULLY SHUTDOWN
    // =================================================================
    // The window was closed. Shut down threads, flush GPU queues, and release RAM.
    g_JobSystem.Shutdown();
    Engine::Renderer::Shutdown();
    Engine::Memory::Shutdown();

    return 0; // Hand control back to Windows/Linux OS
}

// Windows-specific entry point interception
#ifdef _WIN32
    #include <windows.h>
    #include <stdlib.h> // Required for __argc and __argv

    int APIENTRY WinMain(HINSTANCE hInst, HINSTANCE hInstPrev, PSTR cmdline, int cmdshow) {
        // Windows specifically calls this. We just forward it to standard main.
        return main(__argc, __argv); 
    }
#endif
*/

// ==================================================================================
// PARTICLE SIMULATOR BOOTSTRAPPER (main.cpp)
// ==================================================================================
/*
#include "ParticleSystem.h"
#include "Hardware.h" // Your dynamic dispatcher

int main() {
    // 1. Boot Hardware & Job System
    g_Hardware.PrintTelemetry();
    Engine::GameEngine::InitializeDynamicDispatch();
    
    // 2. Initialize Game Data
    Engine::GameEngine::ParticleSystem particles;
    particles.Initialize(100000);
    particles.SpawnParticles(0, 100000, 1.5f);

    // 3. The Game Loop
    while (Engine::Window::ShouldClose() == false) {
        float dt = GetDeltaTime();

        // Pass the raw pointers into the dynamically routed hardware backend
        Engine::GameEngine::ExecuteGameEngineBackend(
            particles.pX.data(), particles.pY.data(), particles.pZ.data(),
            particles.vX.data(), particles.vY.data(), particles.vZ.data(),
            particles.activeCount, dt, 500.0f
        );

        Engine::Renderer::RenderFrame();
    }
    return 0;
}
*/
