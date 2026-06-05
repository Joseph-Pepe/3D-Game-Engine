#pragma once

#include <coroutine>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <cassert>
#include <bit>
#include <exception>
#include <chrono>
#include <new>
#include <immintrin.h>     // REQUIRED for _mm_pause, _mm_prefetch, _mm256_zeroupper

// --- HARDWARE INTRINSICS ---
#ifndef _WIN32_WINNT
    #define _WIN32_WINNT 0x0600 // Vista or later required for modern Fiber APIs
#endif

#if defined(_WIN32) || defined(_MSC_VER)
    #include <windows.h> // Required for OS-level Fibers
    #include <intrin.h>  // REQUIRED for __rdtsc on MSVC
#else
    #include <x86intrin.h> // REQUIRED for __rdtsc on GCC/Clang
    #error "POSIX ucontext or ASM fiber backend required for non-Windows platforms"
#endif

// The Job System needs the fast PRNG from Math.h for the work-stealing logic!
#include "Math.h"
#include "FixedFunction.h"

// ==================================================================================
// CROSS-PLATFORM CACHE LINE ALIGNMENT
// ==================================================================================
// Prevents compilation errors on strict compilers while maintaining false-sharing protection.
#if defined(__cpp_lib_hardware_interference_size)
    constexpr std::size_t CACHE_CHUNK_SIZE = std::hardware_destructive_interference_size;
#else
    constexpr std::size_t CACHE_CHUNK_SIZE = 64; // Standard L1 cache line size
#endif

// ==================================================================================
// COROUTINE MEMORY POOL (Zero-OS Allocation), GLOBAL JOB SYSTEM QUEUE 
// ==================================================================================

// --- GLOBAL JOB SYSTEM QUEUE --- 
// Forward declaration so our tasks can see the global queue
class JobSystem;
extern JobSystem g_JobSystem; // Use 'extern' here to promise the compiler that g_JobSystem exists.

// Used so a worker knows its own index without checking a map we initialize to 0, but will dynamically assign it.
inline thread_local uint32_t tl_workerIndex = 0;

// ==================================================================================
// HYBRID TASK SYSTEM: TAGGED POINTERS
// ==================================================================================
/*
    - Modern 64-bit Operating System (OS) pointers are aligned to atleast 8-bytes. 
    - Its lowest 3-bits of any memory address are always 000.
    - We can hijack the very last bit (Bit 0) to flag whether the job is a Coroutine or a Fiber.
*/

// Bit 0 = 0 -> Stackless Coroutine
// Bit 0 = 1 -> Stackful Fiber

inline void* EncodeCoroutineTask(std::coroutine_handle<> handle) {
    // Assert alignment guarantees bit 0 is empty
    assert((reinterpret_cast<uintptr_t>(handle.address()) & 1) == 0); 
    return handle.address();
}

inline void* EncodeFiberTask(void* fiberHandle) {
    assert((reinterpret_cast<uintptr_t>(fiberHandle) & 1) == 0);
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(fiberHandle) | 1);
}

inline bool IsFiberTask(void* taskPtr) {
    return (reinterpret_cast<uintptr_t>(taskPtr) & 1) != 0;
}

inline void* DecodeFiberHandle(void* taskPtr) {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(taskPtr) & ~1ULL);
}

// ==================================================================================
// 1. STACKLESS COROUTINES (High-Throughput Math)
// ==================================================================================

struct EngineJob {
    // Explicitly align the promise_type so the compiler offsets spilled __m256 YMM registers internally.
    struct alignas(32) promise_type {

        // --- 32-Byte Aligned Allocation Header ---
        // Forces the payload to remain strictly aligned for AVX2 instructions,
        // while tracking exactly which thread's pool this memory belongs to.
        struct alignas(32) AllocHeader {
            uint32_t originThreadId;
            char padding[28]; // Pad out to exactly 32 bytes
        };

        // RAII Wrapper guarantees cleanup when the thread dies
        struct CoroutinePool {
            std::vector<void*> blocks;
            std::size_t frameSize = 0; // Let the compiler tell US the size!

            ~CoroutinePool() {
                for (void* ptr : blocks) {
                    ::operator delete(ptr, std::align_val_t{32}); // Clean up
                }
            }
        };

        // MAGIC: 'inline static thread_local' allows this to exist safely inside the struct! 
        // Every CPU core gets its own private pool! No mutex needed!
        // Each thread gets its own self-cleaning pool!
        inline static thread_local CoroutinePool tl_coroutineFreePool;

        // 1. Thread-Local Zero-Lock Allocation with Origin Tagging
        static void* operator new(std::size_t size) {
            std::size_t totalSize = size + sizeof(AllocHeader);

            // Initialize the expected size on the very first allocation
            if (tl_coroutineFreePool.frameSize == 0) {
                tl_coroutineFreePool.frameSize = totalSize;
            }

            void* rawMemory = nullptr;

            // Safely reuse memory ONLY if the pool has blocks
            if (!tl_coroutineFreePool.blocks.empty() && totalSize == tl_coroutineFreePool.frameSize) {
                rawMemory = tl_coroutineFreePool.blocks.back();
                tl_coroutineFreePool.blocks.pop_back();
            } else {
                // Pool is empty, ask the OS. 
                // Using standard new since alignas(32) on the header preserves padding.
                rawMemory = ::operator new(totalSize, std::align_val_t{32});
            }

            // Write the tracking data to the header
            AllocHeader* header = static_cast<AllocHeader*>(rawMemory);
            header->originThreadId = tl_workerIndex;

            // Return the memory address immediately AFTER our 32-byte header
            return static_cast<void*>(header + 1);
        }

        // 2. OVERRIDE DELETE: Recycle or Destroy
        static void operator delete(void* ptr, std::size_t size) noexcept {
            // Step backwards in memory to read the header
            AllocHeader* header = static_cast<AllocHeader*>(ptr) - 1;
            std::size_t totalSize = size + sizeof(AllocHeader);

            // If the frame size matches, ANY thread can safely adopt this memory block!
            if (totalSize == tl_coroutineFreePool.frameSize) {
                tl_coroutineFreePool.blocks.push_back(header);
            } else {
                // Only ask the OS to destroy the memory if it's a completely mismatched size
                ::operator delete(header, std::align_val_t{32});
            }
        }

        // Capture the handle so we don't lose it (Fixes the memory leak!)
        EngineJob get_return_object() { 
            return EngineJob{std::coroutine_handle<promise_type>::from_promise(*this)}; 
        }
        
        // Start paused! The Job System decides when to fire this.
        std::suspend_always initial_suspend() { return {}; } 
        
        // MAGIC: Auto-destroy the coroutine state frame when the function finishes.
        std::suspend_never final_suspend() noexcept { return {}; } 
        
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
    };

    std::coroutine_handle<promise_type> handle;
};

// ---------------------------------------------------------
// THE AWAITER (YIELDING)
// ---------------------------------------------------------
struct YieldToJobSystem {
    // Return false to tell the compiler: "Suspend me immediately."
    bool await_ready() const noexcept { return false; }

    // Called the exact microsecond the coroutine pauses.
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> handle) const;

    // Called when the coroutine is picked back up.
    void await_resume() const noexcept {} 
};

// ==================================================================================
// 2. STACKFUL FIBERS (Deep Engine Logic & Fiber Pooling)
// ==================================================================================

enum class FiberState : uint32_t {
    Ready,
    Running,
    Yielded,
    Finished // The critical safe-to-delete state
};

// A custom wrapper for the OS Fiber. In a true engine, you'd pool these just like Coroutines.
// alignas(8) guarantees the memory address ends in 000, making it safe for Tagged Pointers!
struct alignas(8) FiberJob {
    void* handle = nullptr;
    FixedFunction<void(), 64> payload; // Zero-allocation, move-only function wrapper.
    std::atomic<FiberState> state{FiberState::Ready};

    FiberJob() {
        // Create a Fiber with a 64KB stack
        handle = CreateFiber(65536, FiberEntryPoint, this);
    }

    ~FiberJob() {
        if (handle) DeleteFiber(handle);
    }

    // DECLARE ONLY: The compiler doesn't need to know what it does yet.
    static void WINAPI FiberEntryPoint(void* lpParameter);

    // This is the static trampoline that the OS calls when the Fiber boots up
    // static void WINAPI FiberEntryPoint(void* lpParameter) {
    //     FiberJob* job = static_cast<FiberJob*>(lpParameter);

    //     // The OS Fiber context never dies. It just loops, waits for work, and yields.
    //     while (true) {
    //         // 1. Execute the user's deep logic
    //         if (job->payload) { // Check if the std::function contains a valid target
    //             job->payload();
    //         }

    //         // 2. Mark this FiberJob as safely completed
    //         job->state.store(FiberState::Finished, std::memory_order_release);

    //         // 3. Yield back to the JobSystem.
    //         // Execution literally freezes on this line until the pool assigns new work!
    //         JobSystem::YieldFiber(nullptr); 
    //     }
    // }
};

// ==================================================================================
// 3. UNIFIED LOCK-FREE SCHEDULER (Stores void*)
// ==================================================================================

// --- THE UNIFIED SCHEDULER ---
// std::hardware_destructive_interference_size: asks the target hardware how large its cache is, so its dynamically scaling to the CPU architecture (Intel/AMD x86, M1/M2/M3 ARM: 128-byte cache).
class alignas(CACHE_CHUNK_SIZE) WorkStealingQueue {
private:
    alignas(CACHE_CHUNK_SIZE) std::atomic<int64_t> top{0};    // Thieves steal from this cache line
    alignas(CACHE_CHUNK_SIZE) std::atomic<int64_t> bottom{0}; // The Owner pushes/pops from this cache line
    // std::vector<std::coroutine_handle<>> jobs;
    std::vector<void*> jobs; // Now stores Tagged Pointers!
    int64_t mask;

public:
    // DEFAULT CAPACITY: 8192 (A fixed-size, lock-free ring buffer must be a power of 2, [4096] or [8192])
    // The work stealing mechanism can pass jobs to any thread, in any order, dynamically, without ever causing a memory overlap.
    WorkStealingQueue(int64_t capacity = 8192) {
        assert(std::has_single_bit(static_cast<uint64_t>(capacity)) && "Capacity MUST be a power of 2!");
        jobs.resize(capacity);
        mask = capacity - 1; // Bitwise mask for hyper-fast wrapping
    }

    // Returns the exact size of the lock-free ring buffer
    int64_t Capacity() const {
        return mask + 1;
    }

    // Returns an estimate of pending jobs (Ring Buffer + Overflow)
    int64_t GetJobCount() const {
        // Load current indices
        int64_t b = bottom.load(std::memory_order_relaxed);
        int64_t t = top.load(std::memory_order_relaxed);
        
        // Calculate ring buffer occupancy (ensure we don't return negative if a steal is in progress)
        return (b > t) ? (b - t) : 0;
    }

    // ONLY the Owner Thread calls Push()
    void Push(void* job) {
        int64_t b = bottom.load(std::memory_order_relaxed);
        int64_t t = top.load(std::memory_order_acquire); // Read where the thieves are

        // STRICT ASSERTION: Fail fast if we blow past the ring buffer size.
        // This guarantees we never silently trigger a memory overwrite or OS allocation.
        assert(b - t < (mask + 1) && "FATAL: Job Queue Overflow! Increase queue capacity.");
        
        jobs[b & mask] = job;
        
        // Ensure the job data is written to RAM before we update the bottom pointer
        std::atomic_thread_fence(std::memory_order_release); 
        bottom.store(b + 1, std::memory_order_relaxed);
    }

    // ONLY the Owner Thread calls Pop()
    void* Pop() {
        int64_t b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        
        // Prevent CPU out-of-order execution from reading 'top' before 'bottom' is saved
        std::atomic_thread_fence(std::memory_order_seq_cst); 
        int64_t t = top.load(std::memory_order_relaxed);

        if (t <= b) {
            // Queue is not empty
            void* job = jobs[b & mask];
            
            if (t == b) {
                // This is the LAST job in the queue. 
                // A thief might be trying to steal this exact job right now!
                // We must race the thief using a Compare-And-Swap (CAS).
                if (!top.compare_exchange_strong(t, t + 1, 
                                                 std::memory_order_seq_cst, 
                                                 std::memory_order_relaxed)) {
                    // We lost the race. The thief stole it.
                    job = nullptr; 
                }
                bottom.store(b + 1, std::memory_order_relaxed); // Reset
            }
            return job;
        } else {
            // Queue was empty
            bottom.store(b + 1, std::memory_order_relaxed); 
            return nullptr;
        }
    }

    // ANY Thread can call Steal()
    void* Steal() {
        int64_t t = top.load(std::memory_order_acquire);        
        int64_t b = bottom.load(std::memory_order_acquire);

        if (t < b) {
            // There is work to steal!
            void* job = jobs[t & mask];
            
            // Attempt to steal it. If another thief grabs it first, CAS will fail.
            // Success CAS requires seq_cst, failure can safely be relaxed
            if (top.compare_exchange_strong(t, t + 1, 
                                            std::memory_order_seq_cst, 
                                            std::memory_order_relaxed)) {
                return job; // Success!
            }
        }
        return nullptr; // Failed to steal, or empty
    }
};

// ==================================================================================
// 4. THE MASTER THREAD POOL & FIBER POOL
// ==================================================================================

class JobSystem {
private:
    std::vector<std::thread> workers;

    // Read constantly by every thread. Keep it on its own isolated island! Must keep wake and sleep away from terminate.
    alignas(CACHE_CHUNK_SIZE) std::atomic<bool> terminate{false};

    // C++20: Atomic Futex for lock-free sleeping
    alignas(CACHE_CHUNK_SIZE) std::atomic<uint32_t> wakeSignal{0};
    
    // ISOLATED: Written frequently as threads sleep/wake.
    // We pad based on the target hardware (by ensuring the next variable is aligned) 
    // to guarantee it lives completely alone.
    alignas(CACHE_CHUNK_SIZE) std::atomic<int> sleepingThreads{0};

    // --- FIBER TRACKING ---
    // Thread-local storage so a running Fiber knows which Worker's core it is currently running on.
    inline static thread_local void* tl_mainWorkerFiber = nullptr;
    inline static thread_local FiberJob* tl_currentFiber = nullptr;

    // --- GLOBAL FIBER POOL ---
    // Pre-allocated storage for the actual fibers
    std::vector<std::unique_ptr<FiberJob>> fiberStorage;
    // Spinlock-protected vector of ready-to-use fibers
    alignas(CACHE_CHUNK_SIZE) std::atomic_flag fiberPoolLock = ATOMIC_FLAG_INIT;
    std::vector<FiberJob*> freeFibers;

    void ExecuteTask(void* job) {
        // Tagged Pointer Evaluation
        if (IsFiberTask(job)) {

            // 1. Decode to our specific FiberJob wrapper
            FiberJob* fiberJob = static_cast<FiberJob*>(DecodeFiberHandle(job));

            fiberJob->state.store(FiberState::Running, std::memory_order_relaxed);
            tl_currentFiber = fiberJob;

            // --- CONTEXT SWITCH ---
            // Jump to the OS-allocated fiber stack!
            SwitchToFiber(fiberJob->handle);

            // We resume the Worker Thread here when the Fiber yields!
            // ----------------------

            tl_currentFiber = nullptr;

            // 2. Evaluate what happened inside the Fiber
            FiberState resultingState = fiberJob->state.load(std::memory_order_acquire);

            if (resultingState == FiberState::Yielded) {
                // The fiber yielded mid-execution (e.g., waiting on IO). 
                // Do NOT recycle it. It will be scheduled again by whatever subsystem it is waiting on.
            } 
            else if (resultingState == FiberState::Finished) {
                // The infinite loop reached the end of the payload. Recycle it!
                ReleaseFiber(fiberJob);
            }
        } else { 
            // --- Stackless Coroutine Logic ---
            // PREFETCH THE COROUTINE FRAME!
            // Triggers the MESI transfer across the CPU cores in the background before the pipeline hits the indirect jump.
            _mm_prefetch((const char*)job, _MM_HINT_T0);

            // It's a Stackless Coroutine! 
            std::coroutine_handle<>::from_address(job).resume();
        }
    }
    
public:
    std::vector<std::unique_ptr<WorkStealingQueue>> queues;
    // ISOLATED: Read heavily by new threads.
    alignas(CACHE_CHUNK_SIZE) std::atomic<uint32_t> nextWorkerId{0};
    uint32_t maxQueues;

    struct alignas(CACHE_CHUNK_SIZE) ThreadMetrics {
        std::atomic<double> utilization{0.0};
        std::atomic<uint64_t> jobsCompleted{0};
        std::atomic<uint64_t> totalFlops{0}; // Isolated math tracker!
    };
    std::vector<std::unique_ptr<ThreadMetrics>> threadStats;

    JobSystem() {
        uint32_t hwThreads = std::thread::hardware_concurrency();

        if (hwThreads == 0) hwThreads = 4; // Safe fallback if OS lies to us
    
        // Reserve 1 core for the main UI thread
        if (hwThreads > 1) hwThreads -= 1;
        
        maxQueues = hwThreads + 8; 
        
        for (uint32_t i = 0; i < maxQueues; ++i) {
            queues.push_back(std::make_unique<WorkStealingQueue>());
        }
        
        RegisterThread(); // Main UI Thread = Index 0

        // PRE-ALLOCATE THE FIBER POOL (e.g., 256 Fibers)
        for (int i = 0; i < 256; ++i) {
            auto job = std::make_unique<FiberJob>();
            freeFibers.push_back(job.get());
            fiberStorage.push_back(std::move(job));
        }

        // The Main thread must also be a Fiber so it can context-switch!
        #if defined(_WIN32)
            tl_mainWorkerFiber = ConvertThreadToFiber(nullptr);
        #endif

        // Manually allocate the metrics structs
        for (uint32_t i = 0; i < maxQueues; ++i) {
            threadStats.push_back(std::make_unique<ThreadMetrics>());
        }
        
        for (uint32_t i = 1; i <= hwThreads; ++i) {
            workers.emplace_back([this, i]() {
                RegisterThread();
                
                // #ifdef _WIN32
                //     // Crucial: Pin workers to cores to keep caches hot
                //     SetThreadAffinityMask(GetCurrentThread(), (1ull << i));
                // #endif

                // Convert this standard OS Thread into a Fiber so it can context switch!
                tl_mainWorkerFiber = ConvertThreadToFiber(nullptr);
                
                // --- Initialize our lightweight 4-byte state ---
                // We use a simple hash of the worker index to ensure different starting seeds
                uint32_t rngState = (tl_workerIndex + 1) * 2654435761u;

                // 1. CACHE THIS LOCALLY! 
                // Thread counts don't dynamically change during gameplay.
                // This permanently removes the atomic load from your memory bus.
                uint32_t localActiveQueues = nextWorkerId.load(std::memory_order_relaxed);

                while (!terminate.load(std::memory_order_relaxed)) {
                    // Used for performance math.
                    auto telemetryStartTime = std::chrono::steady_clock::now();

                    // ==========================================
                    // HARDWARE CYCLE COUNTERS
                    // ==========================================
                    uint64_t chunkStartCycles = __rdtsc(); // __rdtsc() takes ~20 CPU cycles (<5 nanoseconds), std::chrono takes (40-100 nanoseconds) just to check the system clock.
                    uint64_t activeCycles = 0;
                    uint64_t jobsThisFrame = 0;

                    // Track telemetry for 100ms chunks to get a smooth, readable percentage
                    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - telemetryStartTime).count() < 0.1) {

                        // If the engine is dying, get out of the telemetry loop NOW
                        if (terminate.load(std::memory_order_relaxed)) break;

                        // 1. Check our own queue
                        void* job = queues[tl_workerIndex]->Pop();

                        // ==========================================
                        // THE WAKEUP PHASE 
                        // ==========================================

                        // 2. If our queue is empty, sweep ALL other queues starting from a random offset
                        if (!job) {
                            // uint32_t activeQueues = nextWorkerId.load(std::memory_order_relaxed);

                            /* [Branch Predictor]
                                - if-statements in loops only ruin performance when they are unpredictable (e.g., checking if a random particle hit a wall).
                                - When an if-statement is unpredictable, the CPU guesses wrong, has to flush its execution pipeline, and start over (massive penalty).
                                - e.g., an if-statment that is always evaluating to false due to a toggle, it will then execute the next 99,999 iterations of that loop with 100% prediction accuracy.
                            */

                            // results in a 2x performance boost (fixing schedule logic will improve GigaFlops throughput, 2 billion extra calculations per second).
                            // Instead of one core doing 100% of the work in 20 millisconds, we have 4 cores doing 25% of the work in 5 milliseconds.
                            // Sweep all queues in parallel
                            if (localActiveQueues > 1) {
                                // --- Fast random victim selection ---
                                uint32_t randomVal = XorShift32(rngState);
                                uint32_t startVictim = MapToRange(randomVal, localActiveQueues);

                                // OS context switches (sleep, wake back up) are brutally expensive (up to 1,500 microseconds).
                                // thread must be 100% certain the engine is devoid of work before it goes to sleep (i.e., loop guarantees that no jobs are left hiding).
                                // If the worker reach the end and fails to find a job, it goes to sleep.
                                for (uint32_t attempt = 0; attempt < localActiveQueues; ++attempt) {
                                    // --- Hyper-Fast Wrapping ---
                                    // Eliminates the ~15-40 clock cycle hardware division penalty of modulo version [(startVictim + attempt) % localActiveQueues], removing division saves millions of CPU cycles per frame.
                                    uint32_t victimIndex = startVictim + attempt;
                                    if (victimIndex >= localActiveQueues) {
                                        victimIndex -= localActiveQueues; 
                                    }

                                    if (victimIndex != tl_workerIndex) {
                                        job = queues[victimIndex]->Steal();
                                        if (job) break; // Found work! Stop sweeping.
                                    }
                                }
                            }
                        }

                        // 3. If we have a job, execute it!
                        if (job) {
                            // WE FOUND WORK: Start the hardware cycle counter!
                            uint64_t jobStartCycles = __rdtsc(); // __rdtsc() counts literal CPU clock pulses  exactly when the AVX instructions fire.
                            
                            // Let the system figure out if it's a Fiber or Coroutine!
                            ExecuteTask(job);

                            // job.resume(); // Execute the heavy AVX2 math

                            // End cycle counter and add to total active cycles
                            activeCycles += (__rdtsc() - jobStartCycles);
                            jobsThisFrame++;
                        } else {
                            // ==========================================
                            // THE SPIN-LOCK PHASE
                            // ==========================================

                            // If we STILL don't have a job, enter Adaptive Backoff
                            // Adaptive Backoff (Spin-lock): Spin for a long time (covering the whole physics step) before handing control back to the OS.
                            // Spin for a bit to see if work arrives (keeps core hot)
                            // 4,000,000 spins safely covers ~15ms on modern CPUs, guaranteeing they stay awake and "hot" between frames.
                            // 200,000 spins is the sweet spot. It keeps the core hot for quick jobs, but lets it go to sleep quickly to prevent CPU thermal throttling.
                            // [ 500,000 spins = burns 4-6ms], [200,000 spins = burns 1.5-2ms], [100,000 spins = burns 1-1.5ms]
                            // lower numbers means it stays sleeping longer (low-power), worker threads give up looking for work sooner, and moves the work over to the Futex (Fast User-space mutex that is efficient) sooner which.

                            // __rdtsc(): Speaks to the silicons internal clock, so ~4,500,000 cycles is exactly 1.5 milliseconds on a 3GHz processor, regardless of what generation, brand, or architecture the user is running (i.e., deterministic).
                            // If we STILL don't have a job, enter Adaptive Backoff
                            // Spin for ~4,500,000 cycles (~1.5ms on a 3GHz CPU).
                            // This keeps the core hot for quick jobs, but lets it go to sleep quickly to prevent CPU thermal throttling.
                            uint64_t spinStart = __rdtsc();
                            uint64_t spinTarget = 4500000; 
                            uint32_t spinCount = 0;

                            while (__rdtsc() - spinStart < spinTarget) { // 16ms (60fps) or 8.3ms (120fps), 4-Core CPU (1 Main + 3 Workers)
                                
                                // Want to check for work frequently, but not constantly (i.e., reclaims gigabytes gigabytes of internal memory bus bandwidth and keeping the cores awake and ready to grab work in under a microsecond).
                                // By checking every 64 cycles instead of every cycle, you reduce the atomic traffic on your memory bus by 98.4% during idle periods.
                                // This means the workers will stop choking L3 cache and RAM bandwidth, allowing the active physics calculations to run significantly faster.
                                
                                // Idle threads check for work only once every 64 clock cycles (or spins). 
                                // Bitwise AND (& 63) is infinitely faster than modulo (% 64)
                                // This reclaims gigabytes of internal memory bus bandwidth by reducing atomic traffic.
                                if ((spinCount++ & 63) == 0) {
                                    // uint32_t activeQueues = nextWorkerId.load(std::memory_order_relaxed);

                                    // results in a 2x performance boost (fixing schedule logic will improve GigaFlops throughput, 2 billion extra calculations per second).
                                    // Instead of one core doing 100% of the work in 20 millisconds, we have 4 cores doing 25% of the work in 5 milliseconds.
                                    // Hunt for real victims!
                                    // --- Zero-division, cache-friendly victim hunting! ---
                                    uint32_t randomVal = XorShift32(rngState);
                                    uint32_t victimIndex = MapToRange(randomVal, localActiveQueues);

                                    if (victimIndex != tl_workerIndex) {
                                        // PROPERLY CAPTURE THE JOB!
                                        job = queues[victimIndex]->Steal(); // Performs atomic loads across shared memory.
                                    }

                                    // Quick check for new arrivals
                                    if(job) { 
                                        break; // We found work! Break out of the spin loop.
                                    }
                                }
                                _mm_pause(); // Rest the core (ALU)
                            }
                            
                            // 5. Sleep if spin-lock fails! If we spun for 500,000 cycles and STILL found nothing, go to sleep.
                            if (!job) {
                                // Clean registers before handing control to the Windows/Linux OS Kernel
                                _mm256_zeroupper();

                                uint32_t currentSignal = wakeSignal.load(std::memory_order_acquire);
                                sleepingThreads.fetch_add(1, std::memory_order_relaxed); // I am asleep!
                                wakeSignal.wait(currentSignal, std::memory_order_relaxed);
                                sleepingThreads.fetch_sub(1, std::memory_order_relaxed); // I woke up!
                            } else {

                                // PREFETCH THE COROUTINE FRAME!
                                // Triggers the MESI transfer across the CPU cores in the background 
                                // before the pipeline hits the indirect jump.
                                // _mm_prefetch((const char*)job, _MM_HINT_T0);
                                
                                // If the spin loop successfully grabbed a job, execute it immediately.
                                uint64_t jobStartCycles = __rdtsc();
                                // job.resume();
                                ExecuteTask(job);
                                activeCycles += (__rdtsc() - jobStartCycles);
                                jobsThisFrame++;
                            }
                        }
                    }

                    // =============================================================
                    // EXACT SILICON UTILIZATION RATIO  (MATHEMTICAL LOAD OF A CORE)
                    // =============================================================
                    // Calculate total cycles that passed during this ~100ms window
                    uint64_t totalChunkCycles = __rdtsc() - chunkStartCycles;
                    
                    // Prevent divide-by-zero just in case the OS suspended the thread instantly
                    double exactUtilization = (totalChunkCycles > 0) ? (static_cast<double>(activeCycles) / static_cast<double>(totalChunkCycles)) : 0.0;

                    // Update global telemetry for ImGui! Every 100ms, update the global telemetry for the UI to read
                    threadStats[tl_workerIndex]->utilization.store(exactUtilization, std::memory_order_relaxed);
                    threadStats[tl_workerIndex]->jobsCompleted.fetch_add(jobsThisFrame, std::memory_order_relaxed);
                }
            });
        }
    }

    ~JobSystem() {
        // 1. Set the flag first
        terminate.store(true, std::memory_order_relaxed);

        // 2. Brutally wake up everyone. 
        // We increment the signal and notify all so those in 'wakeSignal.wait()' jump out immediately.
        wakeSignal.fetch_add(1, std::memory_order_release);
        wakeSignal.notify_all(); // Wake them up to die

        // 3. Join the threads
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }

        // --- Safe Fiber Teardown ---
        freeFibers.clear();
        fiberStorage.clear(); // This triggers ~FiberJob() which calls DeleteFiber
    }

    // --- FIBER POOL MANAGEMENT ---
    
    FiberJob* GetFreeFiber() {
        // Spinlock to safely pop from the shared pool
        while (fiberPoolLock.test_and_set(std::memory_order_acquire)) { _mm_pause(); }
        
        // In a true AAA system, if the pool is empty, you'd dynamically allocate more.
        assert(!freeFibers.empty() && "FATAL: Fiber pool exhausted!"); 
        
        FiberJob* job = freeFibers.back();
        freeFibers.pop_back();
        
        fiberPoolLock.clear(std::memory_order_release);
        return job;
    }

    void ReleaseFiber(FiberJob* job) {
        job->payload = nullptr; // Free any captured variables in the std::function lambda
        job->state.store(FiberState::Ready, std::memory_order_relaxed);
        
        while (fiberPoolLock.test_and_set(std::memory_order_acquire)) { _mm_pause(); }
        freeFibers.push_back(job);
        fiberPoolLock.clear(std::memory_order_release);
    }

    // --- SCHEDULING INTERFACES ---

    // Call this from ANY detached thread so it gets its own lock-free ring buffer!
    void RegisterThread() {
        tl_workerIndex = nextWorkerId.fetch_add(1, std::memory_order_relaxed);
        assert(tl_workerIndex < maxQueues && "Too many external threads registered!");
    }

    // Schedule Stackless Coroutine
    void Schedule(std::coroutine_handle<> handle) {
        queues[tl_workerIndex]->Push(EncodeCoroutineTask(handle));
    }

    // Schedule Stackful Fiber
    template <typename Callable>
    void ScheduleFiber(Callable&& callable) {
        FiberJob* fiber = GetFreeFiber();

        // Construct the FixedFunction directly in the FiberJob's memory
        fiber->payload = std::forward<Callable>(callable);
        
        // Pass the FiberJob pointer (Encoded) so ExecuteTask can check its state later
        queues[tl_workerIndex]->Push(EncodeFiberTask(fiber));
    }

    // Yield back to the Job System without destroying the Fiber
    static void YieldFiber(void* nextTaskToRun) {
        if (nextTaskToRun) {
            g_JobSystem.queues[tl_workerIndex]->Push(nextTaskToRun);
        }

        // If we yielded mid-function, ensure the state reflects that so ExecuteTask doesn't recycle us!
        if (tl_currentFiber && tl_currentFiber->state.load(std::memory_order_relaxed) == FiberState::Running) {
            tl_currentFiber->state.store(FiberState::Yielded, std::memory_order_release);
        }

        // Jump OUT of the current fiber, and immediately back into the worker thread's main loop!
        SwitchToFiber(tl_mainWorkerFiber);
    }

    // Pass 'task' BY VALUE. The lambda closure is tiny, copying it into 
    // the coroutine frame prevents the use-after-free dangling reference.
    // Pass a raw pointer to a stack-allocated atomic. Zero heap overhead!
    template <typename F>
    EngineJob CreateDispatchTask(uint32_t start, uint32_t end, F task, std::atomic<int>* counter) {
        task(start, end); 
        counter->fetch_sub(1, std::memory_order_release);
        co_return;
    }

    template <typename F>
    void DispatchAndWait(uint32_t dataCount, uint32_t chunkSize, F task) {
        // alignas(64): forces compiler pad the memory.
        // Stack allocated. Fastest memory access possible.
        // Force this atomic counter to sit on its own hardware-specific chunk cache line (L1 cache line size).
        // This prevents other local stack variables from getting caught in the crossfire of thread contention!
        alignas(CACHE_CHUNK_SIZE) std::atomic<int> counter{0};

        uint32_t chunksDispatched = 0; // Track exactly how much work we made

        // ==========================================
        // Prevent Queue Overflow
        // ==========================================
        // 1. Get the absolute maximum number of jobs this queue can hold.
        // We subtract 128 as a safety buffer just in case there are nested jobs already in the queue.
        uint32_t safeQueueCapacity = std::max(1u, static_cast<uint32_t>(queues[tl_workerIndex]->Capacity()) - 128);

        // 2. Mathematically calculate the smallest possible chunk size that prevents an overflow.
        uint32_t absoluteMinChunkSize = (dataCount / safeQueueCapacity) + 1;

        // 3. Force the chunkSize to respect the hardware queue limits before AVX padding.
        // If the user requested chunks of 512, but dataCount is 5,000,000, this will forcefully 
        // increase the chunk size to ~1,250 to ensure we only generate ~3,900 jobs!
        chunkSize = std::max(chunkSize, absoluteMinChunkSize);

        // 4. Ensure it remains strictly aligned for AVX2 (multiples of 8)
        chunkSize = (chunkSize + 7) & ~7;

        /* [Time Sliced Multitasking]
            - If there are more software threads (e.g., 5) than hardware threads (e.g., 4) then the
            - OS places the software threads (e.g., 1) into a waiting room called the ready queue (0% CPU utilization, sleeping).
            - The OS assigns each thread a tiny strict time slice (or Quantum) usually around 1-15 milliseconds on Windows, it does not let the active worker threads run forever.
            - The OS sets a hardware timer. As soon as that timer goes off the OS steps in and seizes control of the CPU.
        */
        
        // 1. Calculate exactly how many chunks we are about to dispatch
        uint32_t totalChunks = (dataCount + chunkSize - 1) / chunkSize;

        // 2. Do ONE atomic write, completely removing the fetch_add from the loop
        counter.store(totalChunks, std::memory_order_release);

        for (uint32_t i = 0; i < dataCount; i += chunkSize) {
            uint32_t start = i;
            uint32_t end = std::min(i + chunkSize, dataCount);
            
            // counter.fetch_add(1, std::memory_order_acquire);
            
            // Pass the address of the local stack variable
            EngineJob job = CreateDispatchTask(start, end, task, &counter);
            Schedule(job.handle);
            chunksDispatched++;
        }

        // Cache this variable to take a snapshot of the engine's state, preventing race conditions and reducing traffic on the CPU's memory bus.
        int asleep = sleepingThreads.load(std::memory_order_relaxed); // load(): forces the CPU to go out to physical RAM/Cache layer to fetch the value.

        // Ring the bell ONLY if cores are actually sleeping!
        if (asleep > 0) {
            // Ring the bell to wake up any sleeping cores
            wakeSignal.fetch_add(1, std::memory_order_release);

            // ==========================================
            // BREAK THE THUNDERING HERD
            // ==========================================
            // Only wake exactly as many threads as we have chunks of work!
            int wakeCount = std::min(asleep, (int)chunksDispatched);

            if (wakeCount >= asleep) {
                wakeSignal.notify_all(); // We need everyone, sound the alarm!
            } else {
                for (int i = 0; i < wakeCount; ++i) {
                    wakeSignal.notify_one(); // Tap just enough workers on the shoulder
                }
            }

            // Force the main thread to yield its time-slice. This prevents the main thread from immediately popping all the jobs it just 
            // created before the OS has time to wake up the worker threads.
            // prevents the main thread from hoarding all the work.
            // std::this_thread::yield(); // tells the OS scheduler that this thread is giving up its CPU (i.e., may cause stutter or jitter).

            // --- Hardware Airlock: Burn exactly ~10,000 cycles (~3.3 microseconds on 3GHz CPU) ---                
            // Lets the workers wake up WITHOUT giving control back to the Windows OS Scheduler.
            uint64_t airlockStart = __rdtsc();
            while (__rdtsc() - airlockStart < 10000) {
                // Hardware Intrinsic Pause: Spin-lock rests the CPU's ALU to save power and allows hyper-threaded sibling cores to spin up (i.e., won't jitter anymore unlike this_thread::yield()).
                _mm_pause();  
            }
        }
        
        // --- Initialize lightweight PRNG state ---
        uint32_t rngState = (tl_workerIndex + 1) * 2654435761u;

        // Cache it once per dispatch!
        uint32_t localActiveQueues = nextWorkerId.load(std::memory_order_relaxed);

        // --- Local Telemetry Tracking ---
        uint64_t localActiveCycles = 0;
        uint64_t localJobsCompleted = 0;
        uint64_t telemetryStartCycles = __rdtsc();
        
        while (counter.load(std::memory_order_acquire) > 0) {
            void* job = queues[tl_workerIndex]->Pop();

            // ==========================================
            // THE DISPATCH WAIT LOOP
            // ==========================================
            if (!job) {
                // Ask the JobSystem exactly how many threads actually exist right now
                // uint32_t activeQueues = nextWorkerId.load(std::memory_order_relaxed);

                // results in a 2x performance boost (fixing schedule logic will improve GigaFlops throughput, 2 billion extra calculations per second).
                // Instead of one core doing 100% of the work in 20 millisconds, we have 4 cores doing 25% of the work in 5 milliseconds.
                // Sweep all active queues
                // Only steal from active threads!
                if (localActiveQueues > 1) {
                    // --- Fast random victim selection ---
                    uint32_t randomVal = XorShift32(rngState);
                    uint32_t startVictim = MapToRange(randomVal, localActiveQueues);

                    // OS context switches (sleep, wake back up) are brutally expensive (up to 1,500 microseconds).
                    // Sweep sequentially through every single queue before deciding the system is truly empty.
                    // thread must be 100% certain the engine is devoid of work before it goes to sleep (i.e., loop guarantees that no jobs are left hiding).
                    // If the worker reach the end and fails to find a job, it goes to sleep.
                    for (uint32_t attempt = 0; attempt < localActiveQueues; ++attempt) {

                        // --- Hyper-Fast Wrapping ---
                        uint32_t victimIndex = startVictim + attempt;
                        if (victimIndex >= localActiveQueues) {
                            victimIndex -= localActiveQueues; 
                        }
                        
                        if (victimIndex != tl_workerIndex) {
                            job = queues[victimIndex]->Steal();
                            if (job) break;
                        }
                    }
                }
            }

            if (job) {
                // PREFETCH THE COROUTINE FRAME!
                _mm_prefetch((const char*)job, _MM_HINT_T0);

                // 1. START HARDWARE TIMER
                uint64_t jobStartCycles = __rdtsc();
                // job.resume(); 

                ExecuteTask(job);

                // 2. END TIMER & ACCUMULATE
                localActiveCycles += (__rdtsc() - jobStartCycles);
                localJobsCompleted++;
            }
            else {
                _mm_pause(); 
            }
        }

        // --- Flush Telemetry to the UI ---
        // Once the entire dispatch chunk is done, calculate how hard this specific thread worked.
        uint64_t totalCyclesElapsed = __rdtsc() - telemetryStartCycles;
        
        if (totalCyclesElapsed > 0) {
            double exactUtilization = static_cast<double>(localActiveCycles) / static_cast<double>(totalCyclesElapsed);
            
            // Update the global UI stats for THIS thread (e.g., Worker 2)
            threadStats[tl_workerIndex]->utilization.store(exactUtilization, std::memory_order_relaxed);
            threadStats[tl_workerIndex]->jobsCompleted.fetch_add(localJobsCompleted, std::memory_order_relaxed);
        }

        // Clean the calling thread (usually the UI thread) before it returns to ImGui/OpenGL
        _mm256_zeroupper();
    }
};

// Global Instance
inline JobSystem g_JobSystem; // C++17: 'inline' allows global variables in a header without Multiple Definition Linker Errors!

// ==================================================================================
// 4. POST-DECLARATION DEFINITIONS
// ==================================================================================

// Define the Fiber's entry point after the JobSystem is fully defined
inline void WINAPI FiberJob::FiberEntryPoint(void* lpParameter) {
    FiberJob* job = static_cast<FiberJob*>(lpParameter);

    // The OS Fiber context never dies. It just loops, waits for work, and yields.
    while (true) {
        // 1. Execute the user's deep logic
        if (job->payload) { 
            job->payload();
        }

        // 2. Mark this FiberJob as safely completed
        job->state.store(FiberState::Finished, std::memory_order_release);

        // 3. Yield back to the JobSystem.
        // Execution literally freezes on this line until the pool assigns new work!
        JobSystem::YieldFiber(nullptr); 
    }
}

// Define the Awaiter's suspend logic after the JobSystem is fully defined
inline std::coroutine_handle<> YieldToJobSystem::await_suspend(std::coroutine_handle<> handle) const {
    // 1. Put the current job back in line
    g_JobSystem.Schedule(handle); 

    // // 2. Grab the next available job
    // std::coroutine_handle<> nextJob = g_JobSystem.queues[tl_workerIndex]->Pop();
    
    // if (nextJob) {
    //     // The C++ compiler converts this into an indirect tail-call jump.
    //     // Zero stack growth. Zero return-to-loop overhead.
    //     return nextJob; 
    // }

    // 2. Return noop to safely hand control back to the scheduler's main loop.
    // The main loop will immediately cycle, pop the next void* job, and safely 
    // route it through ExecuteTask() regardless of whether it is a Fiber or Coroutine.
    // 3. If the queue is empty, return noop to safely hand control back to the scheduler loop.
    return std::noop_coroutine(); 
}
