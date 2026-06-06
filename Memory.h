#pragma once

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
// C++26 SIMD DETECTION, NATIVE SIMD ALIGNMENT
// ==================================================================================

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode
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

// A standard-compliant allocator that guarantees strict memory alignment
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
            return std::allocator<T>{}.allocate(n); // Can declare constexpr AlignedVector<float> which generates the math during compilation and embeds the results directly into the executable binary, and load it into the AVX2-aligned arrays at runtime with zero CPU cycles spent on calculation. 
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
// EXPLICIT HARDWARE ALIGNMENTS (Explicit Control)
// ==================================================================================
/*
    - Typedef for clean architecture

    - SSE     [__m128] demands exactly 16 bytes of data.
    - AVX-2   [__m256] demands exactly 32 bytes of data.
    - AVX-512 [__m512] demands exactly 64 bytes of data.
*/

// --- 16-BYTE ALIGNED ALLOCATOR FOR SSE ALIGNMENT ---
template <typename T> 
using AlignedVector16 = std::vector<T, AlignedAllocator<T, 16>>; // SSE     (16-byte aligned vector)

// --- 32-BYTE ALIGNED ALLOCATOR FOR AVX2 ALIGNMENT ---
template <typename T>
using AlignedVector32 = std::vector<T, AlignedAllocator<T, 32>>; // AVX2    (32-byte aligned vector)

// --- 64-BYTE ALIGNED ALLOCATOR FOR AVX-512 ALIGNMENT ---
template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64>>; // AVX-512 (64-byte aligned vector)

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
// C++26 DYNAMIC HARDWARE ALIGNMENT (Portable SIMD)
// ==================================================================================

#if ENGINE_HAS_CXX26_SIMD
    // C++26 standardizes ABI tags for hardware detection
    using NativeFloatSIMD = std::simd<float, std::simd::simd_abi::native<float>>;

    // 1. Detect the hardware's preferred alignment at compile time! Ask the C++26 standard exactly how many bytes the current hardware needs
    constexpr std::size_t NATIVE_SIMD_ALIGN = alignof(NativeFloatSIMD);

    // 2. Define a ArenaArray that automatically aligns to the current machine's architecture
    template <typename T>
    using NativeAlignedArray = ArenaArray<T, NATIVE_SIMD_ALIGN>; // Assuming custom array
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
class LocalLinearArena {
private:
    uint8_t* m_memory;       // The master pointer to our massive memory block
    size_t   m_capacity;     // Total size of the arena in bytes
    size_t   m_offset;       // The bump pointer (how much we have used) raw O(1) performance

public:
    // Ask the OS for a massive chunk of memory upfront, strictly aligned to 64 bytes (AVX-512 ready)
    LocalLinearArena(size_t sizeInBytes) : m_capacity(sizeInBytes), m_offset(0) {
        // We use native aligned new to guarantee the master block starts on a cache line boundary
        m_memory = static_cast<uint8_t*>(::operator new(sizeInBytes, std::align_val_t{64}));

        if (!m_memory) {
            std::println(std::cerr, "[FATAL] OS Refused to allocate {} bytes for Local Arena!", sizeInBytes);
            std::abort();
        }

        std::println("[MEMORY] Initialized Local Linear Arena: {:.2f} MB", (float)sizeInBytes / (1024.0f * 1024.0f));
    }

    ~LocalLinearArena() { 
        ::operator delete(m_memory, m_capacity, std::align_val_t{64}); 
    }

    // Prevent copying (We don't want two objects thinking they own the same 1GB of RAM)
    LocalLinearArena(const LocalLinearArena&) = delete;
    LocalLinearArena& operator=(const LocalLinearArena&) = delete;

    // --- BARE-METAL BUMP ALLOCATION --- Alignment is now a template parameter to unlock std::assume_aligned optimizations
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate(size_t count) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        #if ENGINE_HAS_CXX26_META_REFLECTION
            [[maybe_unused]] constexpr std::string_view typeName = std::meta::identifier_of(^T);
        #endif

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

    // --- BARE-METAL ALLOCATION WITH SIMD CAPACITY PADDING ---
    // simdWidth: 8 for AVX2 (float), 16 for AVX-512 (float)
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* AllocatePadded(size_t count, size_t simdWidthElements = 8) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        #if ENGINE_HAS_CXX26_META_REFLECTION
            [[maybe_unused]] constexpr std::string_view typeName = std::meta::identifier_of(^T);
        #endif
        
        // 1. CAPACITY PADDING: Round the requested count UP to the nearest multiple of the SIMD width (e.g., 8 for AVX2), so it does not read past the end of the allocation preventing memory corruption.
        // If count is 1021 and simdWidth is 8, paddedCount becomes 1024.
        size_t paddedCount = (count + simdWidthElements - 1) & ~(simdWidthElements - 1);

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

    // --- O(1) FREE ---
    ENGINE_FORCE_INLINE void Reset() noexcept { 
        // We don't overwrite the memory (that wastes CPU cycles).
        // We just move the bump pointer back to the start. The next allocation will cleanly overwrite the old data.
        m_offset = 0; 
    }

    ENGINE_FORCE_INLINE void SetOffset(size_t targetOffset) noexcept { m_offset = targetOffset; }
    
    // Telemetry
    ENGINE_FORCE_INLINE size_t GetUsedMemory() const noexcept { return m_offset; }
    ENGINE_FORCE_INLINE size_t GetCapacity() const noexcept { return m_capacity; }
};

// ==================================================================================
// CONCURRENT LINEAR ARENA (THREAD-SAFE, GLOBAL)
// ==================================================================================

class ConcurrentLinearArena {
private:
    uint8_t* m_memory;              // The master pointer to our massive memory block
    size_t   m_capacity;            // Total size of the arena in bytes
    std::atomic<size_t> m_offset;   // The bump pointer (how much we have used), thread-safe! 

public:
    // Ask the OS for a massive chunk of memory upfront, strictly aligned to 64 bytes (AVX-512 ready)
    ConcurrentLinearArena(size_t sizeInBytes) : m_capacity(sizeInBytes), m_offset(0) {
        // We use native aligned new to guarantee the master block starts on a cache line boundary
        m_memory = static_cast<uint8_t*>(::operator new(sizeInBytes, std::align_val_t{64}));
        
        if (!m_memory) {
            std::println(std::cerr, "[FATAL] OS Refused to allocate {} bytes for Arena!", sizeInBytes);
            std::abort();
        }
        
        std::println("[MEMORY] Initialized Concurrent Linear Arena: {:.2f} MB", (float)sizeInBytes / (1024.0f * 1024.0f));
    }

    ~ConcurrentLinearArena() {
        ::operator delete(m_memory, m_capacity, std::align_val_t{64});
    }

    // Prevent copying (We don't want two objects thinking they own the same 1GB of RAM)
    ConcurrentLinearArena(const ConcurrentLinearArena&) = delete;
    ConcurrentLinearArena& operator=(const ConcurrentLinearArena&) = delete;

    // --- BARE-METAL BUMP ALLOCATION --- Alignment is now a template parameter to unlock std::assume_aligned optimizations
    template <typename T, size_t Align = alignof(T)>
    [[nodiscard]] ENGINE_FORCE_INLINE T* Allocate(size_t count) {
        static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of 2");

        #if ENGINE_HAS_CXX26_META_REFLECTION
            // C++26 Reflection: Introspect the type at compile-time to get its string identifier
            [[maybe_unused]] constexpr std::string_view typeName = std::meta::identifier_of(^T);
        #endif 

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

    // Telemetry
    ENGINE_FORCE_INLINE size_t GetUsedMemory() const noexcept { return m_offset; }
    ENGINE_FORCE_INLINE size_t GetCapacity() const noexcept { return m_capacity; }
};

// ==================================================================================
// ARENA MARKER (TRANSIENT UNWINDING)
// ==================================================================================

class ArenaMarker {
private:
    // Restrict marking entirely to LocalLinearArena. Passing the global concurrent arena will now trigger a compile-time failure.
    LocalLinearArena& m_arena;
    size_t m_savedOffset;

public:
    ArenaMarker(LocalLinearArena& arena) : m_arena(arena), m_savedOffset(arena.GetUsedMemory()) {}
    
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
