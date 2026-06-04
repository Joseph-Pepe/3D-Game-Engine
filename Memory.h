#pragma once

#include <vector>
#include <new>
#include <limits>
#include <memory> // Required for std::allocator in consteval
#include <print>      // Required for std::println
#include <stacktrace> // Required for std::stacktrace

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

// ==================================================================================
// C++26 SIMD DETECTION
// ==================================================================================

// Check if the header exists AND if the compiler is running in C++26 (or newer) mode
#if __has_include(<simd>) && (defined(__cplusplus) && __cplusplus > 202302L || defined(_MSVC_LANG) && _MSVC_LANG > 202302L)
    // C++26 features are unlocked (Optional) Include <simd> if you want the portable vector typedefs
    #include <simd>
    #define ENGINE_HAS_CXX26_SIMD 1
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
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

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

    void deallocate(T* ptr, std::size_t n) noexcept {
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
// C++26 DYNAMIC HARDWARE ALIGNMENT (Portable SIMD)
// ==================================================================================

#if ENGINE_HAS_CXX26_SIMD

    // 1. Detect the hardware's preferred alignment at compile time
    // Ask the C++26 standard exactly how many bytes the current hardware needs
    constexpr std::size_t NATIVE_SIMD_ALIGN = std::simd_alignment_v<std::simd<float>>;

    // 2. Define a vector that automatically aligns to the current machine's architecture
    template <typename T>
    using NativeAlignedVector = std::vector<T, AlignedAllocator<T, NATIVE_SIMD_ALIGN>>;

#endif

// ==================================================================================
// LINEAR ARENA ALLOCATOR (ZERO-FRAGMENTATION MEMORY)
// ==================================================================================
/*
    - Grabs one massive block of memory from the OS at startup.
    - e.g., reserves a flat 1GB of RAM at the exact moment the engine boots up.
    - Allocations are just pointer addition (O(1) time).
    - Deallocations are a single integer reset (O(1) time).
    - Completely eliminates Heap Fragmentation and OS-level memory stalls.
*/

class LinearArena {
private:
    uint8_t* m_memory;        // The master pointer to our massive memory block
    size_t   m_capacity;      // Total size of the arena in bytes
    size_t   m_offset;        // The bump pointer (how much we have used)

public:
    // Ask the OS for a massive chunk of memory upfront, strictly aligned to 64 bytes (AVX-512 ready)
    LinearArena(size_t sizeInBytes) : m_capacity(sizeInBytes), m_offset(0) {
        // We use native aligned new to guarantee the master block starts on a cache line boundary
        m_memory = static_cast<uint8_t*>(::operator new(sizeInBytes, std::align_val_t{64}));
        
        if (!m_memory) {
            std::println(stderr, "[FATAL] OS Refused to allocate {} bytes for Arena!", sizeInBytes);
            std::abort();
        }
        
        std::println("[MEMORY] Initialized Linear Arena: {:.2f} MB", (float)sizeInBytes / (1024.0f * 1024.0f));
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
        
        // 1. Where are we currently in memory?
        uintptr_t currentAddress = reinterpret_cast<uintptr_t>(m_memory + m_offset);
        
        // 2. Bitwise Alignment Calculation (No slow modulo arithmetic!)
        // Formula pushes the address forward to the nearest multiple of the requested alignment.
        size_t padding = (Align - (currentAddress & (Align - 1))) & (Align - 1);
        size_t totalAllocationSize = padding + (count * sizeof(T));

        // 3. Out of Memory Guard
        if (m_offset + totalAllocationSize > m_capacity) {
            std::println(stderr, "[FATAL] LinearArena Exhausted! Capacity: {} bytes", m_capacity);
            std::println(stderr, "{}", std::to_string(std::stacktrace::current()));
            std::abort();
        }

        // 4. Calculate the final aligned pointer
        uintptr_t alignedAddress = currentAddress + padding;
        
        // 5. Bump the offset forward
        m_offset += totalAllocationSize;

        // C++20/26: Prove to the compiler that the memory boundary is safe for AVX
        return std::assume_aligned<Align>(reinterpret_cast<T*>(alignedAddress));
    }

    // --- THE MAGIC O(1) FREE ---
    ENGINE_FORCE_INLINE void Reset() {
        // We don't overwrite the memory (that wastes CPU cycles).
        // We just move the bump pointer back to the start. The next allocation will cleanly overwrite the old data.
        m_offset = 0;
    }

    // Telemetry
    ENGINE_FORCE_INLINE size_t GetUsedMemory() const { return m_offset; }
    ENGINE_FORCE_INLINE size_t GetCapacity() const { return m_capacity; }
};

// Global Memory Pools
// Allocate a 500MB Master Engine Arena
inline LinearArena g_EngineArena(500 * 1024 * 1024);
