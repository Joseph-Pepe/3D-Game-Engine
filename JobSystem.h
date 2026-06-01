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


// The Job System needs the fast PRNG from Math.h for the work-stealing logic!
#include "Math.h"

// ==================================================================================
// COROUTINE GLOBAL JOB SYSTEM QUEUE (Zero-OS Allocation)
// ==================================================================================

// --- GLOBAL JOB SYSTEM QUEUE --- 
// Forward declaration so our tasks can see the global queue
class JobSystem;
extern JobSystem g_JobSystem; // Use 'extern' here to promise the compiler that g_JobSystem exists.

// Used so a worker knows its own index without checking a map we initialize to 0, but will dynamically assign it.
inline thread_local uint32_t tl_workerIndex = 0;

struct EngineJob {
    struct promise_type {

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
    void await_suspend(std::coroutine_handle<> handle) const;

    // Called when the coroutine is picked back up.
    void await_resume() const noexcept {} 
};

// ==================================================================================
// 2. LOCK-FREE SCHEDULING
// ==================================================================================

// --- THE UNIFIED SCHEDULER ---
// A fixed-size, lock-free ring buffer (Must be a power of 2, e.g., 4096 or 8192)
class alignas(64) WorkStealingQueue {
private:
    alignas(64) std::atomic<int64_t> top{0};    // Thieves steal from this cache line
    alignas(64) std::atomic<int64_t> bottom{0}; // The Owner pushes/pops from this cache line
    std::vector<std::coroutine_handle<>> jobs;
    int64_t mask;
    std::vector<std::coroutine_handle<>> localOverflow; // Zero-lock local overflow buffer

public:
    // The work stealing mechanism can pass jobs to any thread, in any order, dynamically, without ever causing a memory overlap.
    WorkStealingQueue(int64_t capacity = 4096) {
        assert(std::has_single_bit(static_cast<uint64_t>(capacity)) && "Capacity MUST be a power of 2!");
        jobs.resize(capacity);
        mask = capacity - 1; // Bitwise mask for hyper-fast wrapping

        // Pre-reserve to prevent mid-frame allocations
        localOverflow.reserve(1024);
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
        int64_t count = (b > t) ? (b - t) : 0;
        
        // Add the local overflow size
        return count + static_cast<int64_t>(localOverflow.size());
    }

    // ONLY the Owner Thread calls Push()
    void Push(std::coroutine_handle<> job) {
        int64_t b = bottom.load(std::memory_order_relaxed);
        int64_t t = top.load(std::memory_order_acquire); // Read where the thieves are

        // mask + 1 is our actual capacity
        if (b - t >= (mask + 1)) {
            // Queue is full! Push to the local overflow buffer safely 
            // instead of risking a recursive stack-overflow via job.resume()
            localOverflow.push_back(job);
            return; 
        }
        
        jobs[b & mask] = job;
        
        // Ensure the job data is written to RAM before we update the bottom pointer
        std::atomic_thread_fence(std::memory_order_release); 
        bottom.store(b + 1, std::memory_order_relaxed);
    }

    // ONLY the Owner Thread calls Pop()
    std::coroutine_handle<> Pop() {
        // Prioritize draining the overflow buffer first!
        if (!localOverflow.empty()) {
            std::coroutine_handle<> job = localOverflow.back();
            localOverflow.pop_back();
            return job;
        }

        int64_t b = bottom.load(std::memory_order_relaxed) - 1;
        bottom.store(b, std::memory_order_relaxed);
        
        // Prevent CPU out-of-order execution from reading 'top' before 'bottom' is saved
        std::atomic_thread_fence(std::memory_order_seq_cst); 
        
        int64_t t = top.load(std::memory_order_relaxed);

        if (t <= b) {
            // Queue is not empty
            std::coroutine_handle<> job = jobs[b & mask];
            
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
    std::coroutine_handle<> Steal() {
        int64_t t = top.load(std::memory_order_acquire);
        
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        int64_t b = bottom.load(std::memory_order_acquire);

        if (t < b) {
            // There is work to steal!
            std::coroutine_handle<> job = jobs[t & mask];
            
            // Attempt to steal it. If another thief grabs it first, CAS will fail.
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
// 3. THE MASTER THREAD POOL
// ==================================================================================

class JobSystem {
private:
    std::vector<std::thread> workers;

    // Read constantly by every thread. Keep it on its own isolated island! Must keep wake and sleep away from terminate.
    alignas(64) std::atomic<bool> terminate{false};

    // C++20: Atomic Futex for lock-free sleeping
    alignas(64) std::atomic<uint32_t> wakeSignal{0};
    
    // ISOLATED: Written frequently as threads sleep/wake.
    // We pad 64 bytes AFTER it (by ensuring the next variable is aligned) 
    // to guarantee it lives completely alone.
    alignas(64) std::atomic<int> sleepingThreads{0};
    
public:
    std::vector<std::unique_ptr<WorkStealingQueue>> queues;
    // ISOLATED: Read heavily by new threads.
    alignas(64) std::atomic<uint32_t> nextWorkerId{0};
    uint32_t maxQueues;

    struct alignas(64) ThreadMetrics {
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

        // FIX 2: Manually allocate the metrics structs
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
                        std::coroutine_handle<> job = queues[tl_workerIndex]->Pop();

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
                                    // Wrap around the ring of queues
                                    // (We still use modulo here for the sequential wrapping offset, 
                                    // but because localActiveQueues is heavily cached and predictable here, it's safer)
                                    uint32_t victimIndex = (startVictim + attempt) % localActiveQueues;

                                    if (victimIndex != tl_workerIndex) {
                                        job = queues[victimIndex]->Steal();
                                        if (job) break; // Found work! Stop sweeping.
                                    }
                                }
                            }
                        }

                        // 3. If we have a job, execute it!
                        if (job) {

                            // PREFETCH THE COROUTINE FRAME!
                            // Triggers the MESI transfer across the CPU cores in the background before the pipeline hits the indirect jump.
                            _mm_prefetch((const char*)job.address(), _MM_HINT_T0);

                            // WE FOUND WORK: Start the hardware cycle counter!
                            uint64_t jobStartCycles = __rdtsc(); // __rdtsc() counts literal CPU clock pulses  exactly when the AVX instructions fire.
                            job.resume(); // Execute the heavy AVX2 math

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
                                _mm_prefetch((const char*)job.address(), _MM_HINT_T0);
                                
                                // If the spin loop successfully grabbed a job, execute it immediately.
                                uint64_t jobStartCycles = __rdtsc();
                                job.resume();
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
    }

    // Call this from ANY detached thread so it gets its own lock-free ring buffer!
    void RegisterThread() {
        tl_workerIndex = nextWorkerId.fetch_add(1, std::memory_order_relaxed);
        assert(tl_workerIndex < maxQueues && "Too many external threads registered!");
    }

    void Schedule(std::coroutine_handle<> handle) {
        queues[tl_workerIndex]->Push(handle);
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
        // Force this atomic counter to sit on its own 64-byte chunk cache line (L1 cache line size).
        // This prevents other local stack variables from getting caught in the crossfire of thread contention!
        alignas(64) std::atomic<int> counter{0};

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
        
        for (uint32_t i = 0; i < dataCount; i += chunkSize) {
            uint32_t start = i;
            uint32_t end = std::min(i + chunkSize, dataCount);
            
            counter.fetch_add(1, std::memory_order_acquire);
            
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
            std::coroutine_handle<> job = queues[tl_workerIndex]->Pop();

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
                        uint32_t victimIndex = (startVictim + attempt) % localActiveQueues;
                        if (victimIndex != tl_workerIndex) {
                            job = queues[victimIndex]->Steal();
                            if (job) break;
                        }
                    }
                }
            }

            if (job) {
                // PREFETCH THE COROUTINE FRAME!
                _mm_prefetch((const char*)job.address(), _MM_HINT_T0);

                // 1. START HARDWARE TIMER
                uint64_t jobStartCycles = __rdtsc();
                job.resume(); 

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

// Define the Awaiter's suspend logic after the JobSystem is fully defined
inline void YieldToJobSystem::await_suspend(std::coroutine_handle<> handle) const {
    g_JobSystem.Schedule(handle); // Put back in line!
}
