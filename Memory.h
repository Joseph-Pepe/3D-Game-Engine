#pragma once

#include "SIMD/SIMDCustomWrapper.h"

#include <vector>
#include <new>
#include <limits>
#include <memory>      // Required for std::allocator in consteval
#include <print>       // Required for std::println
#include <stacktrace>  // Required for std::stacktrace
#include <atomic>      // Required for thread-safe arena

#include <type_traits>
#include <concepts>
#include <span>
#include <cassert>

#include <iostream>    // Required for std::cerr
#include <string_view> // Required for C++26 Reflection string views

// ==================================================================================
// COMPILER SPECIFIC MACROS
// ==================================================================================
#if defined(_MSC_VER)
    #define ENGINE_FORCE_INLINE __forceinline
#elif defined(__clang__) || defined(__GNUC__)
    #define ENGINE_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define ENGINE_FORCE_INLINE inline
#endif

#if __has_include(<meta>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    #include <meta>        // Required for C++26 reflection
    #define ENGINE_HAS_CXX26_META_REFLECTION 1
#else
    #define ENGINE_HAS_CXX26_META_REFLECTION 0
#endif

// ==================================================================================
// CROSS-PLATFORM CACHE CHUNK LINE ALIGNMENT
// ==================================================================================
// Prevents compilation errors on strict compilers while maintaining false-sharing protection.
#if defined(__cpp_lib_hardware_interference_size)
    inline constexpr std::size_t ENGINE_CACHE_CHUNK_SIZE = std::hardware_destructive_interference_size;
#else
    inline constexpr std::size_t ENGINE_CACHE_CHUNK_SIZE = 64; // Standard L1 cache line size
#endif

// =================================================================================================
// COMPILER PROBING (C++26 SIMD DETECTION, NATIVE SIMD ALIGNMENT): Check if C++26 SIMD is available.
// =================================================================================================

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode.
#if __has_include(<simd>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    // C++26 features are unlocked (Optional) Include <simd> if you want the portable vector typedefs
    #include <simd>
    #define ENGINE_HAS_CXX26_SIMD 1

    // C++26: Automatically adapts to AVX2 (8 floats) or AVX-512 (16 floats) based on compiler flags
    using EngineFloatV = std::simd<float>; 

    struct TransformComponent {
        // These natively align themselves and handle their own hardware padding
        EngineFloatV positionX;
        EngineFloatV positionY;
        EngineFloatV positionZ;
    };
#else
    // Fallback for C++23 and older
    #define ENGINE_HAS_CXX26_SIMD 0
#endif

// ==================================================================================
// MEMORY ALLOCATION (HARDWARE ALIGNMENT)
// ==================================================================================
/*
    - std::vector allocators only guarantee 8-byte or 16-byte alignment.
    - AlignedAllocator is a custom allocator for std::vector that forces the OS to give us memory that strictly aligns to 16, 32, 64-byte boundaries.

    - AVX-2: Now every array will start on a 32-byte boundary. Since our loops iterate in multiple of 8 we can use aligned load.
    - Ensures AVX reads will not straddle two different 64-byte cache chunk lines.
*/

// A standard-compliant allocator that guarantees strict memory alignment (cross-platform)
template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
    // Standard STL Allocator Typedefs (MSVC template deduction)
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Modern C++ optimization: Tells vectors they can safely move memory pointers without deep-copying during swaps.
    using is_always_equal = std::true_type;

    // =============================================================
    // EXPLICIT REBIND 
    // =============================================================
    // Explicit rebind is mandatory because 'Alignment' is a non-type parameter.
    // Explicitly tell the STL how to rebind this allocator
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    // constexpr for compile-time generation
    constexpr AlignedAllocator() noexcept = default;

    template <typename U> 
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // Enforce [[nodiscard]] and allow compile-time execution
    [[nodiscard]] constexpr T* allocate(std::size_t n) {
        if (n == 0) return nullptr;

        // --- C++26 COMPILE-TIME EVALUATION ---
        // If the compiler is generating data before the game boots, use standard memory.
        if consteval {
            // C++26 standard way to handle constexpr allocation safely
            std::allocator<T> alloc;
            return std::allocator_traits<std::allocator<T>>::allocate(alloc, n); // Can declare constexpr AlignedVector<float> which generates the math during compilation and embeds the results directly into the executable binary, and load it into the AVX2-aligned arrays at runtime with zero CPU cycles spent on calculation. 
        } else {
            // Runtime Engine Execution: Strict AVX2 Hardware Alignment
            if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
        
            // Native C++17 aligned allocation
            // [::operator new]: when used  with std::align_val_t, the compiler has full visibility into the memory allocation semantics leading to better loop unrolling and aliasing optimizations.
            return static_cast<T*>(
                // std::assume_aligned informs the optimizer, ::operator new handles the actual OS alignment
                std::assume_aligned<Alignment>(::operator new(n * sizeof(T), std::align_val_t{Alignment}))
            );
        }
    }

    constexpr void deallocate(T* ptr, std::size_t n) noexcept {
        if consteval {
            std::allocator<T>{}.deallocate(ptr, n);
        } else {
            // [C++17]: Native aligned deallocation bypasses expensive OS metadata lookups
            ::operator delete(ptr, n * sizeof(T), std::align_val_t{Alignment});
        }
    }
};

// ==================================================================================
// EXPLICIT CAPACITY PADDING (Explicit Control)
// ==================================================================================
/*
    - Guarantees that the total number of entries allocated is a multiple of the SIMD width, so it does not read past the end of the allocation.
    - Mathematical rounding logic used to ensure the OS gives us ghost entries to round to the the nearest multiple of the SIMD width.
    - Prevents multiplying by NaN or uninitialized garbage.
    - AVX2 [__m256]: 8-Width.
*/

ENGINE_FORCE_INLINE size_t GetPaddedCount(size_t count, size_t simdWidth = 8) {
    return (count + simdWidth - 1) & ~(simdWidth - 1);
}

// ==================================================================================
// ARENA ARRAY (Zero-Initialization Container)
// ==================================================================================
/*
    - Replaces std::vector for high-performance engine loops.
    - Operates directly on top of the LinearArena.
    - NEVER initializes memory on resize unless explicitly requested.
    - Automatically propagates SIMD alignment hints to the compiler via std::assume_aligned.
*/

template <typename T, std::size_t Alignment = alignof(T)>
class ArenaArray {
private:
    T* m_data;
    std::size_t m_size;
    std::size_t m_capacity;

public:
    // Delete default constructor to force explicit arena allocation
    ArenaArray() = delete;
    
    // Disable copying to prevent massive memory duplicates (move semantics only)
    ArenaArray(const ArenaArray&) = delete;
    ArenaArray& operator=(const ArenaArray&) = delete;

    // Fast Move Constructor
    ArenaArray(ArenaArray&& other) noexcept 
        : m_data(other.m_data), m_size(other.m_size), m_capacity(other.m_capacity) {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_capacity = 0;
    }

    // ---------------------------------------------------------
    // CONSTRUCTOR: Claim Memory, Bypass Constructors
    // ---------------------------------------------------------
    // Templated to accept either LocalLinearArena or ConcurrentLinearArena
    template <typename ArenaType>
    explicit ArenaArray(ArenaType& arena, std::size_t capacity) 
        : m_size(0), m_capacity(capacity) {
        // We claim the raw memory from the arena. 
        // Zero constructors are called here. It is purely an O(1) pointer bump.
        // 'template' keyword required here because ArenaType is a dependent type
        m_data = arena.template Allocate<T, Alignment>(capacity);
    }

    // ---------------------------------------------------------
    // O(1) ZERO-INITIALIZATION RESIZE
    // ---------------------------------------------------------
    ENGINE_FORCE_INLINE void ResizeUninitialized(std::size_t newSize) {
        // Safety Guard: We strictly enforce trivial types for this operation.
        // If someone tries to skip initialization on a struct containing a std::string or a smart pointer,
        // it will crash the engine when it tries to destruct garbage memory.
        static_assert(std::is_trivially_default_constructible_v<T>, 
            "[ArenaArray] Fatal: T must be trivially default constructible to bypass initialization!");
        
        assert(newSize <= m_capacity && "ArenaArray exceeded reserved arena capacity!");
        m_size = newSize;
    }

    // ---------------------------------------------------------
    // FAST EMPLACE (For explicit initialization)
    // ---------------------------------------------------------
    template <typename... Args>
    ENGINE_FORCE_INLINE T& EmplaceBack(Args&&... args) {
        assert(m_size < m_capacity && "ArenaArray capacity overflow!");
        
        // Placement new: Construct the object directly into our pre-allocated arena memory
        T* ptr = new (&m_data[m_size]) T(std::forward<Args>(args)...);
        m_size++;
        return *ptr;
    }

    // ---------------------------------------------------------
    // FAST PUSH BACK (Uninitialized entry)
    // ---------------------------------------------------------
    ENGINE_FORCE_INLINE T& PushBackUninitialized() {
        static_assert(std::is_trivially_default_constructible_v<T>);
        assert(m_size < m_capacity && "ArenaArray capacity overflow!");
        
        return m_data[m_size++];
    }

    // ---------------------------------------------------------
    // ALIGNED HARDWARE ACCESS
    // ---------------------------------------------------------
    // By using std::assume_aligned here, every time you loop over this array, 
    // the C++ compiler knows it can safely emit AVX/AVX-512 instructions without checking for unaligned bounds.
    [[nodiscard]] ENGINE_FORCE_INLINE T& operator[](std::size_t index) noexcept {
        assert(index < m_size && "ArenaArray Out of Bounds!");
        return *(std::assume_aligned<Alignment>(m_data) + index);
    }

    [[nodiscard]] ENGINE_FORCE_INLINE const T& operator[](std::size_t index) const noexcept {
        assert(index < m_size && "ArenaArray Out of Bounds!");
        return *(std::assume_aligned<Alignment>(m_data) + index);
    }

    // ---------------------------------------------------------
    // C++20/C++26 STANDARD SPAN COMPATIBILITY
    // ---------------------------------------------------------
    // Allows this custom container to be passed directly into standard library algorithms
    // (e.g., std::sort, std::ranges) without needing custom iterators.
    [[nodiscard]] operator std::span<T>() noexcept {
        return std::span<T>(std::assume_aligned<Alignment>(m_data), m_size);
    }
    
    [[nodiscard]] operator std::span<const T>() const noexcept {
        return std::span<const T>(std::assume_aligned<Alignment>(m_data), m_size);
    }

    // ---------------------------------------------------------
    // TELEMETRY & VIEW
    // ---------------------------------------------------------
    [[nodiscard]] ENGINE_FORCE_INLINE std::size_t Size() const noexcept { return m_size; }
    [[nodiscard]] ENGINE_FORCE_INLINE std::size_t Capacity() const noexcept { return m_capacity; }
    
    // Iterable range support
    [[nodiscard]] ENGINE_FORCE_INLINE T* begin() noexcept { return std::assume_aligned<Alignment>(m_data); }
    [[nodiscard]] ENGINE_FORCE_INLINE T* end() noexcept { return std::assume_aligned<Alignment>(m_data) + m_size; }
};

// ==================================================================================
// EXPLICIT HARDWARE ALIGNMENTS (Explicit Control)
// ==================================================================================
/*
    - Typedef for clean architecture

    - SSE     [__m128] demands exactly 16 bytes of data.
    - AVX-2   [__m256] demands exactly 32 bytes of data.
    - AVX-512 [__m512] demands exactly 64 bytes of data.
*/

#if defined(__AVX512F__)
    // --- 16-BYTE ALIGNED ALLOCATOR FOR SSE ALIGNMENT ---
    template <typename T> 
    using AlignedVector16 = std::vector<T, AlignedAllocator<T, 16>>; // SSE     (16-byte aligned vector)

    // --- 32-BYTE ALIGNED ALLOCATOR FOR AVX2 ALIGNMENT ---
    template <typename T>
    using AlignedVector32 = std::vector<T, AlignedAllocator<T, 32>>; // AVX2    (32-byte aligned vector)

    // --- 64-BYTE ALIGNED ALLOCATOR FOR AVX-512 ALIGNMENT ---
    template <typename T>
    using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64>>; // AVX-512 (64-byte aligned vector)
#endif

// ==================================================================================
// C++26 DYNAMIC HARDWARE ALIGNMENT (Portable SIMD)
// ==================================================================================

#if ENGINE_HAS_CXX26_SIMD
    // C++26 standardizes ABI tags for hardware detection
    using NativeFloatSIMD = std::simd<float, std::simd::simd_abi::native<float>>;

    // 1. Detect the hardware's preferred alignment at compile time! Ask the C++26 standard exactly how many bytes the current hardware needs
    constexpr std::size_t NATIVE_SIMD_ALIGN = alignof(NativeFloatSIMD);

    // 2. Define a ArenaArray that automatically aligns to the current machine's architecture
    template <typename T>
    using NativeAlignedArray = ArenaArray<T, NATIVE_SIMD_ALIGN>;
#else
    // C++26 standardizes ABI tags for hardware detection
    using NativeFloatSIMD = Engine::ISAArch::simd<float, Engine::ISAArch::simd_abi::native<float>>; // (or) its implicit equivalent "Engine::ISAArch::WideFloat", both mean [NativeFloatSIMD =  WideFloat]

    // 1. Detect the hardware's preferred alignment at compile time! Because our custom SIMD class wraps native __m256/__m512 types, the compiler natively understands the required byte alignment.
    constexpr std::size_t NATIVE_SIMD_ALIGN = alignof(NativeFloatSIMD);

    // 2. Define an ArenaArray that automatically aligns to the current machine's architecture
    template <typename T>
    using NativeAlignedArray = ArenaArray<T, NATIVE_SIMD_ALIGN>;
#endif


// ==================================================================================
// LOCAL LINEAR ARENA (SINGLE-THREADED, SAFE TO MARK/RESET)
// ==================================================================================
/*
    - Grabs one massive block of memory from the OS at startup.
    - e.g., reserves a flat 1GB of RAM at the exact moment the engine boots up.
    - Allocations are just pointer addition (O(1) time).
    - Deallocations are a single integer reset (O(1) time).
    - Completely eliminates Heap Fragmentation and OS-level memory stalls.
*/

// Created a strictly single-threaded arena for max performance in local scopes.
template <bool ThreadSafe>
class LinearArena {
private:
    uint8_t* m_memory;       // The master pointer to our massive memory block
    size_t   m_capacity;     // Total size of the arena in bytes

    // Dynamically switches between atomic and non-atomic at compile time!
    using OffsetType = std::conditional_t<ThreadSafe, std::atomic<size_t>, size_t>;

    // Aligned to 64 bytes to prevent false sharing across threads
    alignas(ENGINE_CACHE_CHUNK_SIZE) OffsetType m_offset;

    // alignas(ENGINE_CACHE_CHUNK_SIZE) size_t m_offset;                // ThreadSafe (false): The bump pointer (how much we have used) raw O(1) performance
    // alignas(ENGINE_CACHE_CHUNK_SIZE) std::atomic<size_t> m_offset;   // ThreadSafe  (true): The bump pointer (how much we have used), thread-safe! 

public:
    // Ask the OS for a massive chunk of memory upfront, strictly aligned to 64 bytes (AVX-512 ready)
    LinearArena(size_t sizeInBytes) : m_capacity(sizeInBytes), m_offset(0) {
        // We use native aligned new to guarantee the master block starts on a cache line boundary
        m_memory = static_cast<uint8_t*>(::operator new(sizeInBytes, std::align_val_t{64}));

        if (!m_memory) {
            std::println(std::cerr, "[FATAL] OS Refused to allocate {} bytes for Linear Arena!", sizeInBytes);
            std::abort();
        }

        std::println("[MEMORY] Initialized Local Linear Arena: {:.2f} MB", (float)sizeInBytes / (1024.0f * 1024.0f));
    }

    ~LinearArena() { 
        ::operator delete(m_memory, m_capacity, std::align_val_t{64}); 
    }

    // Prevent copying (We don't want two objects thinking they own the same 1GB of RAM)
    LinearArena(const LinearArena&) = delete;
    LinearArena& operator=(const LinearArena&) = delete;

    // --- BARE-METAL BUMP ALLOCATION --- Alignment is now a template parameter to unlock std::assume_aligned optimizations
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate(size_t count) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        #if ENGINE_HAS_CXX26_META_REFLECTION
            // C++26 Reflection: Introspect the type at compile-time to get its string identifier
            [[maybe_unused]] constexpr std::string_view typeName = std::meta::identifier_of(^T);
        #endif

        if constexpr (ThreadSafe) {
            // ==================================================================================
            // CONCURRENT LINEAR ARENA (THREAD-SAFE, GLOBAL)
            // ==================================================================================

            // Execute the Atomic CAS Loop
            size_t oldOffset = m_offset.load(std::memory_order_relaxed);
            size_t padding, totalAllocationSize;

            // Thread-Safe Aligned Allocation Loop
            do {
                // 1. Where are we currently in memory?
                uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_memory + oldOffset);

                // 2. Bitwise Alignment Calculation (No slow modulo arithmetic!)
                // Formula pushes the address forward to the nearest multiple of the requested alignment.
                padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
                totalAllocationSize = padding + (count * sizeof(T));

                // Warn if threads are hammering this for tiny allocations
                if (totalAllocationSize < 65536) [[unlikely]] {
                    // In a real engine, send this to the telemetry system. 
                    // Threads should be asking for at least 64KB chunks to prevent cache-line bouncing.
                    std::println(std::cerr, "[BOUNCE] Thread Allocated Tiny Amount: {} bytes", totalAllocationSize);
                }

                // 3. Out of Memory Guard
                if (oldOffset + totalAllocationSize > m_capacity) [[unlikely]] {
                    std::println(std::cerr, "[FATAL] ConcurrentLinearArena Exhausted! Capacity: {} bytes", m_capacity);
                    std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
                    std::abort();
                }
            // Safely claim the memory block. If another thread beats us, loop and recalculate.
            } while(!m_offset.compare_exchange_weak(
                oldOffset, 
                oldOffset + totalAllocationSize, 
                std::memory_order_acq_rel,   // On Success: Ensure memory visibility across threads
                std::memory_order_relaxed // On Failure: CPU is just looping, no sync needed
            ));

            // Now your telemetry subsystem knows exactly what is being allocated natively
            #if ENGINE_ENABLE_MEMORY_PROFILING && ENGINE_HAS_CXX26_META_REFLECTION
                MemoryProfiler::TrackAllocation(typeName, totalAllocationSize);
            #endif
            
            // 4. Calculate the final aligned pointer
            uintptr_t alignedAddress = reinterpret_cast<uintptr_t>(m_memory + oldOffset) + padding;

            // C++20/26: Prove to the compiler that the memory boundary is safe for AVX
            return std::assume_aligned<Align>(reinterpret_cast<T*>(alignedAddress));
        } 
        // Execute the simple O(1) bump
        else { 
            // ==================================================================================
            // LOCAL LINEAR ARENA (SINGLE-THREADED, SAFE TO MARK/RESET)
            // ==================================================================================

            // 1. Where are we currently in memory?
            uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_memory + m_offset);

            // 2. Bitwise Alignment Calculation pushes the address forward to the nearest multiple of the requested alignment.
            size_t padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
            size_t totalAllocationSize = padding + (count * sizeof(T));

            // 3. Out of Memory Guard
            if (m_offset + totalAllocationSize > m_capacity) [[unlikely]] {
                std::println(std::cerr, "[FATAL] LocalLinearArena Exhausted!");
                std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
                std::abort();
            }

            // 4. Bump the offset forward
            m_offset += totalAllocationSize;

            // 5. Calculate the final aligned pointer
            uintptr_t alignedAddress = currentAddress + padding;

            #if ENGINE_ENABLE_MEMORY_PROFILING && ENGINE_HAS_CXX26_META_REFLECTION
                MemoryProfiler::TrackAllocation(typeName, totalAllocationSize);
            #endif

            // C++20/26: Prove to the compiler that the memory boundary is safe for AVX
            return std::assume_aligned<Align>(reinterpret_cast<T*>(alignedAddress));
        }
    }

    // --- BARE-METAL ALLOCATION WITH SIMD CAPACITY PADDING ---
    // simdWidth: 8 for AVX2 (float), 16 for AVX-512 (float)
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* AllocatePadded(size_t count, size_t simdWidthElements = 8) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        #if ENGINE_HAS_CXX26_META_REFLECTION
            // C++26 Reflection: Introspect the type at compile-time to get its string identifier
            [[maybe_unused]] constexpr std::string_view typeName = std::meta::identifier_of(^T);
        #endif
        
        // 1. CAPACITY PADDING: Round the requested count UP to the nearest multiple of the SIMD width (e.g., 8 for AVX2), so it does not read past the end of the allocation preventing memory corruption.
        // If count is 1021 and simdWidth is 8, paddedCount becomes 1024.
        size_t paddedCount = (count + simdWidthElements - 1) & ~(simdWidthElements - 1);

        if constexpr (ThreadSafe) {
            // ==================================================================================
            // CONCURRENT LINEAR ARENA (THREAD-SAFE, GLOBAL)
            // ==================================================================================

            size_t oldOffset = m_offset.load(std::memory_order_relaxed);
            size_t padding, totalAllocationSize;

            // MUST use CAS loop here as well to guarantee alignment safety
            do {
                // 2. Where are we currently in memory?
                uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_memory + oldOffset);

                // 3. Bitwise Alignment Calculation
                padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
                totalAllocationSize = padding + (paddedCount * sizeof(T)); // Use paddedCount!

                // 4. Out of Memory Guard checks the reserved block
                if (oldOffset + totalAllocationSize > m_capacity) [[unlikely]] {
                    std::println(std::cerr, "[FATAL] LinearArena Exhausted! Capacity: {} bytes", m_capacity);
                    std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
                    std::abort();
                }
            } while (!m_offset.compare_exchange_weak(
                oldOffset, 
                oldOffset + totalAllocationSize, 
                std::memory_order_acq_rel,   // On Success: Ensure memory visibility across threads
                std::memory_order_relaxed    // On Failure: CPU is just looping, no sync needed
            ));

            // Now your telemetry subsystem knows exactly what is being allocated natively
            #if ENGINE_ENABLE_MEMORY_PROFILING && ENGINE_HAS_CXX26_META_REFLECTION
                MemoryProfiler::TrackAllocation(typeName, totalAllocationSize);
            #endif

            // 5. Calculate the final aligned pointer and bump the offset
            uintptr_t alignedAddress = reinterpret_cast<uintptr_t>(m_memory + oldOffset) + padding;
            T* result = std::assume_aligned<Align>(reinterpret_cast<T*>(alignedAddress));

            // 6. ZERO-INITIALIZE THE GHOST ELEMENTS!
            // This is strictly required so that SIMD math on the padded tail doesn't result in NaNs or subnormals.
            for(size_t i = count; i < paddedCount; ++i) {
                new (&result[i]) T(); // Placement new initializes to 0 / default constructor
            }

            return result;
        } 
        else {
            // ==================================================================================
            // LOCAL LINEAR ARENA (SINGLE-THREADED, SAFE TO MARK/RESET)
            // ==================================================================================

            // 2. Where are we currently in memory?
            uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_memory + m_offset);
            
            // 3. Bitwise Alignment Calculation
            size_t padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
            size_t totalAllocationSize = padding + (paddedCount * sizeof(T)); // Use paddedCount!

            // 4. Out of Memory Guard
            if (m_offset + totalAllocationSize > m_capacity) {
                std::println(std::cerr, "[FATAL] LinearArena Exhausted! Capacity: {} bytes", m_capacity);
                std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
                std::abort();
            }

            // 5. Calculate the final aligned pointer and bump the offset
            uintptr_t alignedAddress = currentAddress + padding;
            m_offset += totalAllocationSize;

            #if ENGINE_ENABLE_MEMORY_PROFILING && ENGINE_HAS_CXX26_META_REFLECTION
                MemoryProfiler::TrackAllocation(typeName, totalAllocationSize);
            #endif

            T* result = std::assume_aligned<Align>(reinterpret_cast<T*>(alignedAddress));

            // 6. ZERO-INITIALIZE THE GHOST ELEMENTS!
            // This is strictly required so that SIMD math on the padded tail doesn't result in NaNs or subnormals.
            for(size_t i = count; i < paddedCount; ++i) {
                new (&result[i]) T(); // Placement new initializes to 0 / default constructor
            }

            return result;
        }
    }

    // --- O(1) FREE ---
    ENGINE_FORCE_INLINE void Reset() noexcept { 
        if constexpr (ThreadSafe) {
            m_offset.store(0, std::memory_order_relaxed);
        } else {
            // We don't overwrite the memory (that wastes CPU cycles). We just move the bump pointer back to the start. The next allocation will cleanly overwrite the old data.
            m_offset = 0; 
        }
    }

    // Only allow SetOffset on single-threaded arenas
    ENGINE_FORCE_INLINE void SetOffset(size_t targetOffset) noexcept requires (!ThreadSafe) { 
        m_offset = targetOffset; 
    }
    
    // Telemetry
    ENGINE_FORCE_INLINE size_t GetUsedMemory() const noexcept { 
        if constexpr (ThreadSafe) {
            return m_offset.load(std::memory_order_relaxed);
        } else {
            return m_offset; 
        }
    }

    ENGINE_FORCE_INLINE size_t GetCapacity() const noexcept { return m_capacity; }
};

// --- LOCAL LINEAR ARENA (SINGLE-THREADED, SAFE TO MARK/RESET) ---
using LocalLinearArena = LinearArena<false>;

// --- CONCURRENT LINEAR ARENA (THREAD-SAFE, GLOBAL) ---
using ConcurrentLinearArena = LinearArena<true>;

// ==================================================================================
// FRAME ALLOCATOR (DOUBLE / TRIPLE BUFFERING)
// ==================================================================================
/*
    - Replaces general heap allocation for data that only lives for a single frame.
    - Render commands, UI vertices, physics contact manifolds, temporary string formatting.
    - Required for the GPU to process Vulkan/DirectX12 command buffers.
    - Automatically supports Double Buffering (2) or Triple Buffering (3).
    - Prevents the CPU from overwriting the memory the GPU is trying to read which leads to screen tearing and crashes.
    - ZERO destructors are called. The memory is instantly obliterated at the end of the frame.
*/

// BufferCount: 2 = Console/V-Sync Locked (Double), 3 = Uncapped PC (Triple)
template <size_t BufferCount = 2>
class alignas(ENGINE_CACHE_CHUNK_SIZE) FrameAllocator {
    static_assert(BufferCount >= 2, "Frame Allocator must have at least 2 buffers (Double Buffering).");

private:
    // C++20/26: std::array guarantees contiguous allocation of the arenas
    // std::array<LocalLinearArena, BufferCount> m_arenas;

    // By using an array of uninitialized bytes, we bypass the missing default constructor.
    // alignas() ensures the block starts perfectly on a cache line boundary.
    alignas(LocalLinearArena) std::byte m_arenaStorage[sizeof(LocalLinearArena) * BufferCount];
    
    // Aligned to prevent false sharing if placed near other atomic variables
    alignas(ENGINE_CACHE_CHUNK_SIZE) size_t m_currentFrame;

    ENGINE_FORCE_INLINE LocalLinearArena& GetArena(size_t index) {
        return *reinterpret_cast<LocalLinearArena*>(&m_arenaStorage[index * sizeof(LocalLinearArena)]);
    }

    ENGINE_FORCE_INLINE const LocalLinearArena& GetArena(size_t index) const {
        return *reinterpret_cast<const LocalLinearArena*>(&m_arenaStorage[index * sizeof(LocalLinearArena)]);
    }

public:
    // Initialize all underlying arenas with the exact same capacity
    // Example: 16MB per frame buffer = 32MB total RAM usage for Double Buffering.
    FrameAllocator(size_t bytesPerFrame) 
        : m_currentFrame(0) {

        for (size_t i = 0; i < BufferCount; ++i) {
            // Placement new directly into our contiguous uninitialized byte block.
            new (&GetArena(i)) LocalLinearArena(bytesPerFrame);
        }
    }

    ~FrameAllocator() {
        for (size_t i = 0; i < BufferCount; ++i) {
            // Because m_arenaStorage is just standard bytes, the compiler will NOT 
            // auto-destruct the objects. Manual destruction is now perfectly safe and required!
            GetArena(i).~LocalLinearArena();
        }
    }

    // Prevent copying and moving to lock this memory physically to its instantiation site
    FrameAllocator(const FrameAllocator&) = delete;
    FrameAllocator& operator=(const FrameAllocator&) = delete;

    // --- BARE-METAL FRAME ALLOCATION ---
    // Passes the alignment requirement straight through to the underlying local arena.
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate(size_t count = 1) {
        // O(1) pointer bump, perfectly aligned, zero locks.
        return GetArena(m_currentFrame).template Allocate<T, Align>(count);
    }

    // --- FRAME OBJECT INSTANTIATION ---
    // Allocates memory AND calls the constructor. Useful for temporary frame-bound objects.
    template <typename T, size_t Align = alignof(T), typename... Args>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Emplace(Args&&... args) {
        T* ptr = GetArena(m_currentFrame).template Allocate<T, Align>(1);
        return new (ptr) T(std::forward<Args>(args)...);
    }

    // --- GPU FENCE SYNC & FRAME FLIP ---
    // Call this EXACTLY ONCE per thread at the very beginning of your main loop.
    ENGINE_FORCE_INLINE void FlipFrame() {
        
        // 1. Compile-Time Modulo Optimization
        // Modulo (%) is a slow CPU instruction. If BufferCount is 2 (power of 2), 
        // the compiler can replace it with a blindingly fast bitwise AND (& 1).
        if constexpr ((BufferCount & (BufferCount - 1)) == 0) {
            m_currentFrame = (m_currentFrame + 1) & (BufferCount - 1);
        } else {
            // Fallback for Triple Buffering (3 is not a power of 2)
            m_currentFrame = (m_currentFrame + 1) % BufferCount;
        }

        // =====================================================================
        // CRITICAL ENGINE ARCHITECTURE NOTE: GPU FENCES
        // =====================================================================
        // In a real Vulkan/DX12 engine, before we reset this arena, we MUST check 
        // the hardware GPU Fence to guarantee the GPU has finished rendering 
        // the frame associated with this memory block.
        // 
        // Example:
        // if (!RenderAPI::IsGPUFinished(m_currentFrame)) {
        //     RenderAPI::WaitForGPU(m_currentFrame); // Stall the CPU until GPU is done
        // }
        // =====================================================================

        // 2. Wipe the newly active frame clean in O(1) time.
        // Zero destructors are called. The memory is ready to be instantly overwritten.
        GetArena(m_currentFrame).Reset();
    }

    // Telemetry
    ENGINE_FORCE_INLINE size_t GetCurrentFrameIndex() const noexcept { return m_currentFrame; }
    ENGINE_FORCE_INLINE size_t GetCurrentUsedMemory() const noexcept { return GetArena(m_currentFrame).GetUsedMemory(); }
    ENGINE_FORCE_INLINE size_t GetCapacityPerFrame() const noexcept { return GetArena(m_currentFrame).GetCapacity(); }
};

// ==================================================================================
// GLOBAL FRAME MEMORY DECLARATION
// ==================================================================================
// Define exactly how much temporary memory a single frame is allowed to generate.
constexpr size_t FRAME_MEMORY_SIZE = 16 * 1024 * 1024; // 16 MB per frame

// Single-Threaded Application:
// inline FrameAllocator<2> g_MainFrameAllocator(FRAME_MEMORY_SIZE);

// Multi-Threaded Application (The AAA Way):
// Every worker thread gets its own Double-Buffered 16MB allocator. Zero locks!
// thread_local FrameAllocator<2> t_WorkerFrameAllocator(FRAME_MEMORY_SIZE);

/*
void RenderSystem::Update() {
    // 1. Advance the frame (swaps buffer 0 to 1, waits for GPU, and wipes it)
    t_WorkerFrameAllocator.FlipFrame();

    // 2. Allocate an array of SIMD matrices for 10,000 asteroids
    // This memory is strictly aligned to 64 bytes (AVX-512 ready).
    Matrix4x4_SIMD* modelMatrices = t_WorkerFrameAllocator.Allocate<Matrix4x4_SIMD, 64>(10000);

    for (int i = 0; i < 10000; i++) {
        modelMatrices[i] = CalculateAsteroidTransform(i);
    }

    // 3. Allocate a temporary string for the UI
    // char arrays allocated here will vanish when the frame flips. No std::string heap allocations!
    char* fpsText = t_WorkerFrameAllocator.Allocate<char>(64);
    std::snprintf(fpsText, 64, "FPS: %d", currentFPS);

    // 4. Send the perfectly aligned, contiguous memory to the GPU Command Buffer
    VulkanRenderer::SubmitDrawCall(modelMatrices, 10000);
    UIRenderer::SubmitText(fpsText);
    
    // Frame ends. We don't delete anything!
}
*/

// ==================================================================================
// LOCAL POOL ALLOCATOR (O(1) BUMP-TO-POOL HYBRID)
// ==================================================================================
/*
    - Designed for Game Objects, Projectiles, Audio Voices, and Network Packets.
    - Handles objects that spawn and die at completely random, unpredictable intervals.
    - O(1) Instant Startup: Defers free-list generation until objects actually die (Zero Page Faults).
    - Intrusive Free-List: The 'next' pointer is secretly stored inside the dead object's memory.
    - Pre-allocates a massive array of objects. When an object dies, it writes the memory address of the next free slot into the dead object's memory, creating a linked list of free memory (a free list) with zero overhead.
    - Mandatory for Game Object/Entity system, projectiles, network packets, and audio voices.
    - LocalPoolAllocator is for single threaded object recycling.
*/

template <typename T, bool ThreadSafe, size_t Align = alignof(T)>
class alignas(ENGINE_CACHE_CHUNK_SIZE) PoolAllocator {
    
    // Ensure alignment is a power of 2 and at least large enough to hold a pointer
    static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");
    static_assert(Align >= alignof(void*), "Alignment must accommodate at least a pointer size");

private:
    // --- INTRUSIVE LINKED LIST NODE ---
    // This node secretly lives inside the memory of a DEAD object. 
    // When the object is alive, this data is completely overwritten by the actual T object.
    struct alignas(Align) FreeNode {
        FreeNode* next;
    };

    // --- COMPILE-TIME BLOCK SIZING ---
    // We must guarantee that the block is large enough to hold the object OR the free node,
    // and that every single block perfectly adheres to the hardware alignment (e.g., 64 bytes for AVX-512).
    static constexpr size_t MinBlockSize = (sizeof(T) > sizeof(FreeNode)) ? sizeof(T) : sizeof(FreeNode);
    static constexpr size_t BlockSize = (MinBlockSize + Align - 1) & ~(Align - 1);

    uint8_t* m_memory;  // Master block of perfectly aligned memory
    size_t m_capacity;  // Maximum number of objects this pool can hold

    // Dynamically switch to atomics if ThreadSafe is true (std::atomic for thread safety)
    using NodePtrType = std::conditional_t<ThreadSafe, std::atomic<FreeNode*>, FreeNode*>;
    using IndexType   = std::conditional_t<ThreadSafe, std::atomic<size_t>, size_t>;

    // Align them to prevent false sharing.
    alignas(ENGINE_CACHE_CHUNK_SIZE) NodePtrType m_freeListHead;  // Head of the linked list (only used for dead objects)
    alignas(ENGINE_CACHE_CHUNK_SIZE) IndexType   m_bumpIndex;     // Tracks how many blocks we have allocated initially (The Lazy Bump Pointer)

public:
    // O(1) Allocation from the OS. Zero initialization loops.
    PoolAllocator(size_t capacity) 
        : m_capacity(capacity), m_bumpIndex(0), m_freeListHead(nullptr) {
        
        // Native aligned allocation guarantees the entire block starts on the correct boundary
        m_memory = static_cast<uint8_t*>(::operator new(capacity * BlockSize, std::align_val_t{Align}));

        if (!m_memory) {
            std::println(std::cerr, "[FATAL] OS Refused to allocate Pool Arena!");
            std::abort();
        }

        std::println("[MEMORY] Initialized Hybrid Pool Allocator: {:.2f} MB (Capacity: {})", 
                     (float)(capacity * BlockSize) / (1024.0f * 1024.0f), capacity);
    }

    ~PoolAllocator() {
        ::operator delete(m_memory, m_capacity * BlockSize, std::align_val_t{Align});
    }

    // Prevent copy/move to isolate the memory pool
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;

    // --- BARE-METAL ALLOCATION ---
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate() {
        if constexpr (ThreadSafe) {
            // 1. FREE-LIST PATH (Thread-Safe Pop)
            FreeNode* head = m_freeListHead.load(std::memory_order_acquire);
            while (head != nullptr) {
                // Attempt to move the head to the next node. If another thread beat us, loop and try again.
                if (m_freeListHead.compare_exchange_weak(head, head->next, std::memory_order_release, std::memory_order_relaxed)) {
                    return std::assume_aligned<Align>(reinterpret_cast<T*>(head));
                }
            }

            // 2. BUMP-POINTER PATH (Thread-Safe Increment), fetch_add returns the PREVIOUS value, ensuring each thread gets a unique index.
            size_t index = m_bumpIndex.fetch_add(1, std::memory_order_relaxed);
            
            if (index < m_capacity) {
                return std::assume_aligned<Align>(reinterpret_cast<T*>(m_memory + (index * BlockSize)));
            }

            std::abort();     // Out of memory
            return nullptr;
        } else {
            // 1. FREE-LIST PATH (local pop, prioritize recycling dead objects to keep cache memory "hot")
            if (m_freeListHead != nullptr) {
                void* ptr = m_freeListHead;
                m_freeListHead = m_freeListHead->next;
                // C++20/26: Prove to the compiler the pointer strictly adheres to AVX/SIMD boundaries
                return std::assume_aligned<Align>(static_cast<T*>(ptr));
            } 

            // 2. BUMP-POINTER PATH (local increment)
            if (m_bumpIndex < m_capacity) {
                void* ptr = m_memory + (m_bumpIndex * BlockSize);
                m_bumpIndex++;
                // C++20/26: Prove to the compiler the pointer strictly adheres to AVX/SIMD boundaries
                return std::assume_aligned<Align>(static_cast<T*>(ptr));
            } 

            // 3. OUT OF MEMORY
            std::println(std::cerr, "[FATAL] Pool Allocator Exhausted!");
            std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
            std::abort();
            return nullptr;
        }
    }

    // --- O(1) EMPLACE (ALLOCATE + CONSTRUCT) ---
    template <typename... Args>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Emplace(Args&&... args) {
        T* ptr = Allocate();
        // Placement new constructs the object directly in the recycled memory
        return new (ptr) T(std::forward<Args>(args)...);
    }

    // --- O(1) DEALLOCATION ---
    ENGINE_FORCE_INLINE void Free(T* ptr) noexcept {
        if (!ptr) return;

        // 1. Explicitly call the destructor of the object BEFORE we overwrite its memory
        if constexpr (!std::is_trivially_destructible_v<T>) {
            ptr->~T();
        }

        // 2. Intrusive Linked List: Cast the dead object's memory into a FreeNode
        FreeNode* node = reinterpret_cast<FreeNode*>(ptr);

        if constexpr (ThreadSafe) {
            // 3. FREE-LIST PUSH (Thread-Safe Push)
            FreeNode* head = m_freeListHead.load(std::memory_order_relaxed);
            do {
                node->next = head;
            // Attempt to set the head to our new node. If another thread changed the head, update our 'head' variable and try again.
            } while (!m_freeListHead.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
        } else {
            // 3. Push it to the front of the Free List (LIFO order ensures cache-hot memory is reused first)
            node->next = m_freeListHead;
            m_freeListHead = node;
        }
    }

    // --- TELEMETRY ---
    // Note: Active count requires subtracting the length of the free list (which is an O(N) operation to traverse).
    // In a pure performance environment, we only track the high-water mark via bump index.
    ENGINE_FORCE_INLINE size_t GetHighWaterMark() const noexcept { return m_bumpIndex; }
    ENGINE_FORCE_INLINE size_t GetCapacity() const noexcept { return m_capacity; }
};

// --- LOCAL POOL ALLOCATOR (O(1) BUMP-TO-POOL HYBRID) --- 
// Backwards compatibility aliases!
template <typename T, size_t Align = alignof(T)>
using LocalPoolAllocator = PoolAllocator<T, false, Align>;


// --- CONCURRENT POOL ALLOCATOR (THREAD-SAFE O(1) BUMP-TO-POOL HYBRID) --- 
// Is a thread-safe version of LocalPoolAllocator that is used for global game objects (bullets, network packets) spawned from multiple threads.
template <typename T, size_t Align = alignof(T)>
using ConcurrentPoolAllocator = PoolAllocator<T, true, Align>;

/*
// 1. Create a massive pool of perfectly aligned Game Objects
// For AVX2, align to 32 bytes.
LocalPoolAllocator<GameObject, 32> g_ProjectilePool(100000); // Pool of 100,000 bullets

void FireWeapon() {
    // 2. Instantly grabs cache-hot memory and constructs the bullet
    GameObject* bullet = g_ProjectilePool.Emplace(currentPos, currentDirection);
}

void OnBulletHitWall(GameObject* bullet) {
    // 3. Instantly shreds the bullet, calls its destructor, and adds it back to the free list
    g_ProjectilePool.Free(bullet);
}
*/

// ==================================================================================
// OS VIRTUAL MEMORY PAGING ARENA (ZERO-FRAGMENTATION STREAMING VAULT)
// ==================================================================================
/*
    - Designed for Massive Open-World Streaming, Entity Component Systems, and Asset Vaults.
    - Reserves billions of virtual addresses instantly with exactly 0 bytes of physical RAM overhead.
    - Eliminates kernel page faults by managing explicit physical page commitment.
    - Dual-Kernel Architecture: Native integration for Windows Kernel and POSIX/Linux systems.
    - Interface directly with the OS Virtual Memory Manager, bypasses 'new' entirely.
    - e.g., reserve 10GB of contiguous virtual addresses, but consumes 0 bytes of physical RAM.
*/

#if defined(_WIN32) || defined(_WIN64)
    #define WIN32_LEAN_AND_MEAN
    #define NOMINMAX
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

class alignas(ENGINE_CACHE_CHUNK_SIZE) VirtualMemoryArena {
private:
    uint8_t* m_baseAddress;       // The structural anchor in the 64-bit address space
    size_t   m_reservedSize;      // Total size of the virtual address reservation
    size_t   m_bumpOffset;        // High-water mark of requested engine memory
    size_t   m_committedOffset;   // High-water mark of actual physical RAM committed from the OS
    size_t   m_pageSize;          // Detected hardware MMU page size (typically 4KB)

    // --- KERNEL SPECIFIC DETECTION ---
    size_t QueryOSPageSize() noexcept {
        #if defined(_WIN32) || defined(_WIN64)
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            return static_cast<size_t>(sysInfo.dwPageSize);
        #else
            return static_cast<size_t>(sysconf(_SC_PAGESIZE));
        #endif
    }

    // --- KERNEL SYSTEM CALLS ---
    void OSReserve(size_t size) {
        #if defined(_WIN32) || defined(_WIN64)
            m_baseAddress = static_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS));
        #else
            m_baseAddress = static_cast<uint8_t*>(mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (m_baseAddress == MAP_FAILED) m_baseAddress = nullptr;
        #endif
    }

    void OSCommit(void* ptr, size_t size) noexcept {
        #if defined(_WIN32) || defined(_WIN64)
            VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
        #else
            mprotect(ptr, size, PROT_READ | PROT_WRITE);
        #endif
    }

    void OSDecommit(void* ptr, size_t size) noexcept {
        #if defined(_WIN32) || defined(_WIN64)
            VirtualFree(ptr, size, MEM_DECOMMIT);
        #else
            // Inform the kernel to release physical backing frames instantly
            madvise(ptr, size, MADV_DONTNEED);
            mprotect(ptr, size, PROT_NONE);
        #endif
    }

public:
    explicit VirtualMemoryArena(size_t totalReservationSize)
        : m_baseAddress(nullptr), m_bumpOffset(0), m_committedOffset(0) {
        
        m_pageSize = QueryOSPageSize();
        
        // Enforce that the entire reservation layout maps strictly to OS page blocks
        m_reservedSize = (totalReservationSize + m_pageSize - 1) & ~(m_pageSize - 1);

        OSReserve(m_reservedSize);

        if (!m_baseAddress) [[unlikely]] {
            std::println(std::cerr, "[FATAL] Kernel completely refused virtual address reservation of {} GB!", 
                         (float)m_reservedSize / (1024.0f * 1024.0f * 1024.0f));
            std::abort();
        }

        std::println("[MEMORY] Virtual Arena Configured. Reserved Address Space: {:.2f} GB | Hardware Page Size: {} KB", 
                     (float)m_reservedSize / (1024.0f * 1024.0f * 1024.0f), m_pageSize / 1024);
    }

    ~VirtualMemoryArena() {
        if (m_baseAddress) {
            #if defined(_WIN32) || defined(_WIN64)
                VirtualFree(m_baseAddress, 0, MEM_RELEASE);
            #else
                munmap(m_baseAddress, m_reservedSize);
            #endif
        }
    }

    VirtualMemoryArena(const VirtualMemoryArena&) = delete;
    VirtualMemoryArena& operator=(const VirtualMemoryArena&) = delete;

    // --- HYPER-OPTIMIZED ALLOCATION LOOP ---
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate(size_t count) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_baseAddress + m_bumpOffset);
        size_t padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
        size_t totalAllocationSize = padding + (count * sizeof(T));

        size_t newBumpOffset = m_bumpOffset + totalAllocationSize;

        if (newBumpOffset > m_reservedSize) [[unlikely]] {
            std::println(std::cerr, "[FATAL] Out of Virtual Address Space! Hard Limit reached.");
            std::println(std::cerr, "{}", std::to_string(std::stacktrace::current()));
            std::abort();
        }

        // --- AUTOMATIC LAZY PAGE COMMITMENT ---
        if (newBumpOffset > m_committedOffset) {
            size_t dynamicBytesNeeded = newBumpOffset - m_committedOffset;
            // Mathematical rounding logic to map directly to hardware page widths
            size_t pageAlignedBytes = (dynamicBytesNeeded + m_pageSize - 1) & ~(m_pageSize - 1);

            OSCommit(m_baseAddress + m_committedOffset, pageAlignedBytes);
            m_committedOffset += pageAlignedBytes;
        }

        uintptr_t targetAddress = currentAddress + padding;
        m_bumpOffset = newBumpOffset;

        return std::assume_aligned<Align>(reinterpret_cast<T*>(targetAddress));
    }

    // --- SURGICAL RAM RELEASE (AAA WORLD STREAMING UNLOAD) ---
    /*
        Allows the engine to free megabytes of physical RAM back to the operating system
        whenever a zone is streamed out, without invalidating references or fragmenting memory.
    */
    ENGINE_FORCE_INLINE void FreeToMarker(size_t targetOffsetMarker) noexcept {
        if (targetOffsetMarker >= m_bumpOffset) return;

        // Round up target offset marker to the nearest page boundary to safeguard preceding active assets
        size_t safePageBoundary = (targetOffsetMarker + m_pageSize - 1) & ~(m_pageSize - 1);

        if (safePageBoundary < m_committedOffset) {
            size_t bytesToDecommit = m_committedOffset - safePageBoundary;
            OSDecommit(m_baseAddress + safePageBoundary, bytesToDecommit);
            m_committedOffset = safePageBoundary;
        }

        m_bumpOffset = targetOffsetMarker;
    }

    // --- INSTANT RESET ---
    ENGINE_FORCE_INLINE void Reset() noexcept {
        if (m_committedOffset > 0) {
            OSDecommit(m_baseAddress, m_committedOffset);
            m_committedOffset = 0;
            m_bumpOffset = 0;
        }
    }

    // --- TELEMETRY ---
    ENGINE_FORCE_INLINE size_t GetVirtualAddressUsage() const noexcept { return m_bumpOffset; }
    ENGINE_FORCE_INLINE size_t GetPhysicalRamCommitment() const noexcept { return m_committedOffset; }
    ENGINE_FORCE_INLINE size_t GetTotalReservedCapacity() const noexcept { return m_reservedSize; }
};


// ==================================================================================
// ARENA MARKER (TRANSIENT UNWINDING)
// ==================================================================================
template <bool ThreadSafe>
class ArenaMarker {
    static_assert(!ThreadSafe, "Fatal: You cannot mark and rewind a Concurrent Arena! Only Local Arenas are safe to rewind.");
    
private:
    // Restrict marking entirely to LocalLinearArena. Passing the global concurrent arena will now trigger a compile-time failure.
    LinearArena<ThreadSafe>& m_arena;
    size_t m_savedOffset;

public:
    ArenaMarker(LinearArena<ThreadSafe>& arena) : m_arena(arena), m_savedOffset(arena.GetUsedMemory()) {}
    
    // Automatically unwind the memory when this object goes out of scope
    ~ArenaMarker() {
        m_arena.SetOffset(m_savedOffset); // Add a SetOffset function to your arena
    }
};

// ==================================================================================
// GLOBAL ENGINE INSTANCES
// ==================================================================================

// -- Global Memory Pools -- Allocate a 500MB Master Engine Arena

// 1. Master Engine Arena (Thread-safe, NEVER marked/reset during gameplay)
inline ConcurrentLinearArena g_EnginePersistentArena(500 * 1024 * 1024);

// 2. Thread-Local Arenas (Single-threaded access, SAFE to mark), A dedicated 50MB arena just for the Physics thread or subsystem.
thread_local LocalLinearArena t_PhysicsTransientArena(50 * 1024 * 1024);
