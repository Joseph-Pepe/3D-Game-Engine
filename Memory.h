#pragma once

#include <vector>
#include <new>
#include <limits>
#include <memory> // Required for std::allocator in consteval

// ==================================================================================
// MEMORY ALLOCATION (AVX2 ALIGNMENT)
// ==================================================================================
// --- 32-BYTE ALIGNED ALLOCATOR FOR AVX2 ---
/*
    - __m256 demands exactly 32 bytes of data.
    - std::vector allocators don't guarantee this, they only guarantee 8-byte or 16-byte alignment.
    - AlignedAllocator is a custom allocator for std::vector that forces the OS to give us memory that strictly aligns to 32-byte boundaries.
    - Ensures AVX reads will not straddle two different 64-byte cache chunk lines.
    - Now every array will start on a 32-byte boundary. Since our loops iterate in multiple of 8 we can use aligned load.
*/
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
            return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
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

// Typedef for clean architecture
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 32>>;
