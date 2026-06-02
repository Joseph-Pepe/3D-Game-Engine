#pragma once

// ==================================================================================
// MEMORY ALLOCATION (AVX2 ALIGNMENT)
// ==================================================================================

// --- 32-BYTE ALIGNED ALLOCATOR FOR AVX2 ---
/*
    - __m256 demands exactly 32 bytes of data.
    - std::vector allocators don't guarantee this, they only guarantee 8-byte or 16-byte alignment.
    - AlignedAllocator is a custom allocator for std::vector that forces the OS to give us memory that strictly aligns to 32-byte boundaries.
    - Ensures AVX reads will not straddle two different 64-byte cache chunk lines.
    - Now every array will start on a 32-byte boundary.
    - Since our loops iterate in multiple of 8 we can use aligned load.
*/

template <typename T, std::size_t Alignment = 32>
struct AlignedAllocator {
    using value_type = T;

    // =============================================================
    // Explicitly tell the STL how to rebind this allocator
    // =============================================================
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    // Modern C++ optimization: Tells vectors they can safely move 
    // memory pointers without deep-copying during swaps.
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // [::operator new]: when used  with std::align_val_t, the compiler has full visibility into the memory allocation semantics leading to better loop unrolling and aliasing optimizations.
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
        
        // Native C++17 aligned allocation (No OS macros needed!)
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* ptr, std::size_t n) noexcept {
        // Native C++17 aligned deallocation (Requires passing the size and alignment!), this bypasses an expensive metadata lookup
        ::operator delete(ptr, n * sizeof(T), std::align_val_t{Alignment});
    }
};

// Typedef for clean architecture
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 32>>;
