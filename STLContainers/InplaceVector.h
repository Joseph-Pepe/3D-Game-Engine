#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <iterator>
#include <initializer_list>
#include <algorithm>
#include <cstring>


// --- ENGINE ASSERTION MACRO ---
// In a real engine, this routes to your custom crash handler.
// By default, it traps the debugger instantly on console hardware.
// Hook this into your engine's custom assertion/crash handler. On Windows/Xbox, this usually maps to __debugbreak(). On PS5, SCE_BREAK().
#ifndef ENGINE_ASSERT
    #if defined(_MSC_VER)
        #define ENGINE_ASSERT(condition) do { if (!(condition)) __debugbreak(); } while(false)
    #elif defined(__clang__) || defined(__GNUC__)
        #define ENGINE_ASSERT(condition) do { if (!(condition)) __builtin_trap(); } while(false)
    #else
        #include <cassert>
        #define ENGINE_ASSERT(condition) assert((condition))
    #endif
#endif

// --- COMPILER HINT: ASSUME ---
// Forces the compiler to strip bounds-checking branches in release builds.
#if defined(_MSC_VER)
    #define ENGINE_ASSUME(condition) __assume(condition)
#elif defined(__clang__)
    #define ENGINE_ASSUME(condition) __builtin_assume(condition)
#else
    #define ENGINE_ASSUME(condition) do {} while(false)
#endif

namespace Engine::STLContainer {

    template <typename T, std::size_t Capacity>
    class inplace_vector {
        static_assert(Capacity > 0, "inplace_vector capacity must be strictly greater than 0.");

    public:
        // --- Standard Library Typedefs (For algorithm/iterator compatibility) ---
        using value_type      = T;
        using size_type       = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference       = T&;
        using const_reference = const T&;
        using pointer         = T*;
        using const_pointer   = const T*;
        using iterator        = T*;
        using const_iterator  = const T*;

    private:
        // =====================================================================
        // C++26 CONSTEXPR STORAGE UNION
        // =====================================================================
        // Prior to C++20, developers used alignas(T) std::byte array[...].
        // However, reinterpret_cast is forbidden in constexpr contexts. By using a union, we can maintain completely uninitialized memory 
        // that is fully C++26 constexpr compliant!
        union Storage {
            char m_buffer; // Active member initially (empty state)
            T m_data[Capacity];

            constexpr Storage() : m_buffer{} {}
            constexpr ~Storage() {} // Destruction handled manually by inplace_vector
        };

        Storage m_storage;
        size_type m_size = 0;

        // C++17 compliant aligned uninitialized storage.
        // alignas(T) std::byte m_buffer[Capacity * sizeof(T)];
        // std::size_t m_size = 0;

        // // cleanly cast bytes to our type
        // [[nodiscard]] constexpr T* ptr(std::size_t index) noexcept {
        //     return reinterpret_cast<T*>(m_buffer + (index * sizeof(T)));
        // }
        // [[nodiscard]] constexpr const T* ptr(std::size_t index) const noexcept {
        //     return reinterpret_cast<const T*>(m_buffer + (index * sizeof(T)));
        // }

    public:
        // --- CONSTRUCTORS ---
        constexpr inplace_vector() noexcept = default;

        constexpr inplace_vector(std::initializer_list<T> ilist) {
            ENGINE_ASSERT(ilist.size() <= Capacity && "Initializer list exceeds capacity!");
            for (const auto& item : ilist) {
                unchecked_push_back(item);
            }
        }

        // --- COPY & MOVE SEMANTICS (Memory Optimized) ---
        constexpr inplace_vector(const inplace_vector& other) noexcept(std::is_nothrow_copy_constructible_v<T>) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                if consteval {
                    // For compile-time, we must loop and construct
                    for (std::size_t i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                    }
                } else {
                    // For runtime, raw memory copy is the absolute fastest path
                    std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                }
                m_size = other.m_size; // Safe to batch-assign size for trivial types
            } else {
                for (std::size_t i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                    ++m_size; // Increment safely one-by-one
                }
            }
        }

        constexpr inplace_vector(inplace_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                if consteval {
                    for (std::size_t i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                    }
                } else {
                    std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                }
                m_size = other.m_size;
            } else {
                for (std::size_t i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                    ++m_size; 
                }
            }
            other.m_size = 0; // Clear the moved-from container safely
        }

        // --- DESTRUCTOR ---
        constexpr ~inplace_vector() noexcept {
            // CRITICAL: If T is a plain struct (like Vector3D), we skip the destruction loop entirely.
            if constexpr (!std::is_trivially_destructible_v<T>) {
                clear();
            }
        }

        // --- COPY/MOVE ASSIGNMENT ---
        constexpr inplace_vector& operator=(const inplace_vector& other) {
            if (this != &other) {
                clear(); // Sets m_size to 0
                if constexpr (std::is_trivially_copyable_v<T>) {
                    if consteval {
                        for (size_type i = 0; i < other.m_size; ++i) {
                            std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                        }
                    } else {
                        std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                    }
                    m_size = other.m_size;
                } else {
                    for (size_type i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                        ++m_size;
                    }
                }
            }
            return *this;
        }

        constexpr inplace_vector& operator=(inplace_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
            if (this != &other) {
                clear();
                if constexpr (std::is_trivially_copyable_v<T>) {
                    if consteval {
                        for (size_type i = 0; i < other.m_size; ++i) {
                            std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                        }
                    } else {
                        std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                    }
                    m_size = other.m_size;
                } else {
                    for (size_type i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                        ++m_size;
                    }
                }
                other.m_size = 0;
            }
            return *this;
        }

        // --- ITERATORS (Enables C++ ranges and algorithms) ---
        [[nodiscard]] constexpr iterator begin() noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr iterator end() noexcept { return m_storage.m_data + m_size; }
        [[nodiscard]] constexpr const_iterator end() const noexcept { return m_storage.m_data + m_size; }

        // --- CAPACITY & SIZE ---
        [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
        [[nodiscard]] constexpr static size_type capacity() noexcept { return Capacity; }
        [[nodiscard]] constexpr bool empty() const noexcept { return m_size == 0; }
        [[nodiscard]] constexpr bool full() const noexcept { return m_size == Capacity; }

        // --- DATA ACCESS ---
        [[nodiscard]] constexpr pointer data() noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr const_pointer data() const noexcept { return m_storage.m_data; }

        [[nodiscard]] constexpr reference front() noexcept { 
            ENGINE_ASSUME(m_size > 0); 
            return operator[](0); 
        }
        [[nodiscard]] constexpr const_reference front() const noexcept { 
            ENGINE_ASSUME(m_size > 0); 
            return operator[](0); 
        }

        [[nodiscard]] constexpr reference back() noexcept { 
            ENGINE_ASSUME(m_size > 0); 
            return operator[](m_size - 1); 
        }
        [[nodiscard]] constexpr const_reference back() const noexcept { 
            ENGINE_ASSUME(m_size > 0); 
            return operator[](m_size - 1); 
        }

        // --- DATA INDEX ACCESS ---
        [[nodiscard]] constexpr reference operator[](size_type index) noexcept {
            ENGINE_ASSERT(index < m_size && "Index out of bounds!"); // For the debugger
            ENGINE_ASSUME(index < m_size);                           // For the optimizer
            return m_storage.m_data[index];
        }

        [[nodiscard]] constexpr const_reference operator[](size_type index) const noexcept {
            ENGINE_ASSERT(index < m_size && "Index out of bounds!"); // For the debugger
            ENGINE_ASSUME(index < m_size);                           // For the optimizer
            return m_storage.m_data[index];
        }

        // --- MODIFIERS ---
        // Safe push_back (Traps in debug if full)
        constexpr void push_back(const T& value) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            // std::construct_at is a C++20 requirement for constexpr dynamic object creation
            std::construct_at(m_storage.m_data + m_size, value);
            ++m_size;
        }

        constexpr void push_back(T&& value) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            std::construct_at(m_storage.m_data + m_size, std::move(value));
            ++m_size;
        }

        template <typename... Args>
        constexpr reference emplace_back(Args&&... args) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            pointer ptr = m_storage.m_data + m_size;
            std::construct_at(ptr, std::forward<Args>(args)...);
            ++m_size;
            return *ptr;
        }

        // =====================================================================
        // UNCHECKED ACCESS
        // =====================================================================
        // When your physics loop guarantees it will only ever process N elements,
        // bounds checking is wasted CPU cycles. These bypass the assert completely and force the compiler to drop all branching logic.
        constexpr void unchecked_push_back(const T& value) {
            // The auto-vectorizer will completely strip bounds-checking branches from this loop!
            ENGINE_ASSUME(m_size < Capacity);
            std::construct_at(m_storage.m_data + m_size, value);
            ++m_size;
        }

        constexpr void unchecked_push_back(T&& value) {
            ENGINE_ASSUME(m_size < Capacity);
            std::construct_at(m_storage.m_data + m_size, std::move(value));
            ++m_size;
        }

        constexpr void pop_back() noexcept {
            ENGINE_ASSERT(m_size > 0 && "inplace_vector underflow, cannot pop_back from an empty container!");
            --m_size;
            // Only call the destructor if the type actually needs it!
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_at(m_storage.m_data + m_size);
            }
        }

        // OPTIMIZATION: If the object is purely data (like a float or a Matrix), we skip the destruction loop entirely and just set size to 0. 
        constexpr void clear() noexcept {
            // This turns an O(N) operation into an O(1) operation.
            if constexpr (!std::is_trivially_destructible_v<T>) {
                // All arrays and containers must destroy its entries in strict reverse order of construction! Prevents bugs.
                for (size_type i = m_size; i > 0; --i) {
                    std::destroy_at(&m_storage.m_data[i - 1]);
                }
            }
            m_size = 0;
        }
    };
} 
