#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <iterator>
#include <initializer_list>
#include <algorithm>
#include <ranges>
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
#elif defined(__GNUC__)
    #define ENGINE_ASSUME(condition) do { if (!(condition)) __builtin_unreachable(); } while(false)
#else
    #define ENGINE_ASSUME(condition) do {} while(false)
#endif

// Detect Trivial Relocatability (Clang/GCC extension, fallback to trivial_copyable for MSVC)
#if defined(__clang__) || defined(__GNUC__)
    #define ENGINE_IS_RELOCATABLE(T) __is_trivially_relocatable(T)
#else
    // Fallback for MSVC until P1144 is fully merged into the compiler frontend
    #define ENGINE_IS_RELOCATABLE(T) std::is_trivially_copyable_v<T>
#endif

namespace Engine::STLContainer {

    template <typename T, std::size_t Capacity>
    class inplace_vector {
        static_assert(Capacity > 0, "inplace_vector capacity must be strictly greater than 0.");

    public:
        // --- Standard Library Interface (For algorithm/iterator compatibility) ---
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
        // CONSTEXPR STORAGE UNION (C++20/26 Compliant)
        // =====================================================================
        // Prior to C++20, developers used alignas(T) std::byte array[...].
        // By using a union, we can maintain completely uninitialized memory that is fully C++26 constexpr compliant!
        // Union allows uninitialized memory to be valid in constexpr contexts.
        // We manage the exact lifetime of 'm_data' elements manually.
        union Storage {
            struct EmptyState {} m_empty; // Active member initially (empty state)
            T m_data[Capacity];

            constexpr Storage() : m_empty{} {}
            constexpr ~Storage() {} // Destruction handled manually by inplace_vector
        };

        Storage m_storage;
        size_type m_size = 0;

        // reinterpret_cast is forbidden in constexpr contexts. 

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
                    for (size_type i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                    }
                } else {
                    // For runtime, raw memory copy is the absolute fastest path
                    std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                }
            } else {
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                }
            }
            m_size = other.m_size; // Safe to batch-assign size for trivial types
        }

        constexpr inplace_vector(inplace_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
            if constexpr (ENGINE_IS_RELOCATABLE(T)) {
                if consteval {
                    for (size_type i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                    }
                } else {
                    // Memcpy is mathematically safe for trivially relocatable types, bypassing move constructors entirely.
                    std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                }
            } else {
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                }
            }
            m_size = other.m_size;
            other.clear(); // Ensure moved-from container obeys lifetime rules
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
                    if !consteval {
                        std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                        m_size = other.m_size;
                        return *this;
                    } 
                } 
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], other.m_storage.m_data[i]);
                }
                m_size = other.m_size;
            }
            return *this;
        }

        constexpr inplace_vector& operator=(inplace_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
            if (this != &other) {
                clear();
                if constexpr (ENGINE_IS_RELOCATABLE(T)) {
                    if !consteval {
                        std::memcpy(m_storage.m_data, other.m_storage.m_data, other.m_size * sizeof(T));
                        m_size = other.m_size;
                        other.m_size = 0;
                        return *this;
                    }
                }
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_storage.m_data[i], std::move(other.m_storage.m_data[i]));
                }
                m_size = other.m_size;
                other.m_size = 0;
            }
            return *this;
        }

        // --- CAPACITY & SIZE ---
        [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
        [[nodiscard]] constexpr static size_type capacity() noexcept { return Capacity; }
        [[nodiscard]] constexpr bool empty() const noexcept { return m_size == 0; }
        [[nodiscard]] constexpr bool full() const noexcept { return m_size == Capacity; }

        // OPTIMIZATION: If the object is purely data (like a float or a Matrix), we skip the destruction loop entirely and just set size to 0. 
        constexpr void clear() noexcept {
            // This turns an O(N) operation into an O(1) operation.
            if constexpr (!std::is_trivially_destructible_v<T>) {
                // Destruction of containers entries must occur in reverse order of construction! Prevents bugs.
                for (size_type i = m_size; i > 0; --i) {
                    std::destroy_at(&m_storage.m_data[i - 1]);
                }
            }
            m_size = 0;
        }

        // --- DATA ACCESS ---
        [[nodiscard]] constexpr pointer data() noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr const_pointer data() const noexcept { return m_storage.m_data; }

        // --- ITERATORS (Enables C++ ranges and algorithms) ---
        [[nodiscard]] constexpr iterator begin() noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return m_storage.m_data; }
        [[nodiscard]] constexpr iterator end() noexcept { return m_storage.m_data + m_size; }
        [[nodiscard]] constexpr const_iterator end() const noexcept { return m_storage.m_data + m_size; }

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

        // =====================================================================
        // INSERTION / MODIFIERS
        // =====================================================================

        // Standard safe push
        constexpr reference push_back(const T& value) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            // std::construct_at is a C++20 requirement for constexpr dynamic object creation
            std::construct_at(&m_storage.m_data[m_size], value);
            reference ref = m_storage.m_data[m_size]; // Cache reference
            ++m_size;                                 // Increment safely
            return ref;
        }

        constexpr reference push_back(T&& value) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            std::construct_at(&m_storage.m_data[m_size], std::move(value));
            reference ref = m_storage.m_data[m_size]; // Cache reference
            ++m_size;                                 // Increment safely
            return ref;
        }

        // C++26 standard: try_push_back (Safe boundary testing without assertions)
        [[nodiscard]] constexpr pointer try_push_back(const T& value) noexcept {
            if (m_size >= Capacity) [[unlikely]] return nullptr;
            std::construct_at(&m_storage.m_data[m_size], value);
            pointer ptr = &m_storage.m_data[m_size]; // Cache pointer
            ++m_size;                                // Increment safely
            return ptr;
        }

        [[nodiscard]] constexpr pointer try_push_back(T&& value) noexcept {
            if (m_size >= Capacity) [[unlikely]] return nullptr;
            std::construct_at(&m_storage.m_data[m_size], std::move(value));
            pointer ptr = &m_storage.m_data[m_size]; // Cache pointer
            ++m_size;                                // Increment safely
            return ptr;
        }

        // =====================================================================
        // UNCHECKED ACCESS
        // =====================================================================
        // When your physics loop guarantees it will only ever process N elements, bounds checking is wasted CPU cycles. These bypass the assert completely and force the compiler to drop all branching logic.
        // High-performance bypass: Guarantees compiler auto-vectorization
        constexpr reference unchecked_push_back(const T& value) {
            // The auto-vectorizer will completely strip bounds-checking branches from this loop!
            ENGINE_ASSUME(m_size < Capacity);
            std::construct_at(&m_storage.m_data[m_size], value);
            reference ref = m_storage.m_data[m_size]; // Cache reference
            ++m_size;                                 // Increment safely
            return ref;
        }

        constexpr reference unchecked_push_back(T&& value) {
            ENGINE_ASSUME(m_size < Capacity);
            std::construct_at(&m_storage.m_data[m_size], std::move(value));
            reference ref = m_storage.m_data[m_size]; // Cache reference
            ++m_size;                                 // Increment safely
            return ref;
        }

        template <typename... Args>
        constexpr reference emplace_back(Args&&... args) {
            ENGINE_ASSERT(m_size < Capacity && "inplace_vector overflow, capacity exceeded!");
            std::construct_at(&m_storage.m_data[m_size], std::forward<Args>(args)...);
            reference ref = m_storage.m_data[m_size]; // Cache reference
            ++m_size;                                 // Increment safely
            return ref;
        }

        constexpr void pop_back() noexcept {
            ENGINE_ASSERT(m_size > 0 && "inplace_vector underflow, cannot pop_back from an empty container!");
            --m_size;
            // Only call the destructor if the type actually needs it!
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_at(&m_storage.m_data[m_size]);
            }
        }

        // =====================================================================
        // C++26 RANGES API (append_range & try_append_range)
        // =====================================================================

        // Standard C++26 append_range
        template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, T>
        constexpr void append_range(R&& rg) {
            if constexpr (std::ranges::sized_range<R> || std::ranges::forward_range<R>) {
                size_type dist = static_cast<size_type>(std::ranges::distance(rg));
                ENGINE_ASSERT(m_size + dist <= Capacity && "append_range overflow!");

                // O(1) Block Memory Transfer for trivially copyable contiguous ranges
                if constexpr (std::ranges::contiguous_range<R> && std::is_trivially_copyable_v<T>) {
                    if !consteval {
                        if (dist > 0) {
                            std::memcpy(m_storage.m_data + m_size, std::ranges::data(rg), dist * sizeof(T));
                            m_size += dist;
                        }
                        return;
                    }
                }
                
                // Fallback for non-trivial or constexpr evaluation
                for (auto&& e : rg) {
                    std::construct_at(&m_storage.m_data[m_size++], std::forward<decltype(e)>(e));
                }
            } else {
                // Input iterators cannot be measured upfront, bounds check on every iteration
                for (auto&& e : rg) {
                    ENGINE_ASSERT(m_size < Capacity && "append_range overflow!");
                    std::construct_at(&m_storage.m_data[m_size++], std::forward<decltype(e)>(e));
                }
            }
        }

        // Standard C++26 try_append_range (Safely fills to capacity, returns iterator of uninserted elements)
        template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, T>
        [[nodiscard]] constexpr std::ranges::borrowed_iterator_t<R> try_append_range(R&& rg) noexcept {
            auto it = std::ranges::begin(rg);
            auto end = std::ranges::end(rg);
            
            if constexpr (std::ranges::contiguous_range<R> && std::is_trivially_copyable_v<T>) {
                if !consteval {
                    size_type remaining_capacity = Capacity - m_size;
                    size_type dist = static_cast<size_type>(std::ranges::distance(rg));
                    size_type to_copy = std::min(dist, remaining_capacity);
                    
                    if (to_copy > 0) {
                        std::memcpy(m_storage.m_data + m_size, std::ranges::data(rg), to_copy * sizeof(T));
                        m_size += to_copy;
                        std::ranges::advance(it, to_copy);
                    }
                    return it; // Returns iterator pointing to the first rejected element
                }
            }
            
            while (it != end && m_size < Capacity) {
                std::construct_at(&m_storage.m_data[m_size++], *it);
                ++it;
            }
            return it;
        }

        // Engine Specific: Unchecked bulk append
        template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, T>
        constexpr void unchecked_append_range(R&& rg) noexcept {
            if constexpr (std::ranges::sized_range<R> || std::ranges::forward_range<R>) {
                size_type dist = static_cast<size_type>(std::ranges::distance(rg));
                ENGINE_ASSUME(m_size + dist <= Capacity);

                if constexpr (std::ranges::contiguous_range<R> && std::is_trivially_copyable_v<T>) {
                    if !consteval {
                        if (dist > 0) {
                            std::memcpy(m_storage.m_data + m_size, std::ranges::data(rg), dist * sizeof(T));
                            m_size += dist;
                        }
                        return;
                    }
                }
                for (auto&& e : rg) {
                    std::construct_at(&m_storage.m_data[m_size++], std::forward<decltype(e)>(e));
                }
            } else {
                for (auto&& e : rg) {
                    ENGINE_ASSUME(m_size < Capacity);
                    std::construct_at(&m_storage.m_data[m_size++], std::forward<decltype(e)>(e));
                }
            }
        }

        // =====================================================================
        // ENGINE SPECIFIC OPERATIONS
        // =====================================================================

        // --- FAST ENGINE ERASURE (O(1)) DATA-ORIENTED PIPELINES ---
        // Overwrites the target element with the last element in the array, then pops the back. Destroys the original ordering of the array.
        // Game Engine: We don't really care about maintaining the order of an array when killing a particle or destroying an entity.
        constexpr void erase_unsorted(const_iterator pos) {
            iterator it = const_cast<iterator>(pos);
            ENGINE_ASSERT(it >= begin() && it < end() && "erase_unsorted iterator out of bounds!");
            
            // If it's not the very last element, move the last element into this spot
            if (it != end() - 1) {
                *it = std::move(back()); 
            }
            
            pop_back(); // Handles the destruction and size decrement safely
        }
        
        constexpr void erase_unsorted_by_index(size_type index) {
            ENGINE_ASSERT(index < m_size && "erase_unsorted index out of bounds!");
            if (index != m_size - 1) {
                m_storage.m_data[index] = std::move(back());
            }
            pop_back();
        }

        // --- BULK SIZING ---
        
        // Bypasses all constructors. Extremely dangerous but incredibly fast.
        // Use ONLY when passing the buffer to a C-API (like Vulkan, DirectX, or File I/O) that will immediately overwrite the memory.
        constexpr void resize_uninitialized(size_type new_size) noexcept {
            static_assert(std::is_trivially_constructible_v<T>, "resize_uninitialized is only safe for trivial types (POD)!");
            ENGINE_ASSERT(new_size <= Capacity && "resize_uninitialized exceeds capacity!");
            // ENGINE_ASSUME helps the optimizer know m_size is now exactly new_size
            m_size = new_size; 
            ENGINE_ASSUME(m_size == new_size);
        }

        // Standard resize (safe)
        constexpr void resize(size_type new_size) {
            ENGINE_ASSERT(new_size <= Capacity && "resize exceeds capacity!");
            if (new_size > m_size) {
                // Grow and default construct
                for (size_type i = m_size; i < new_size; ++i) {
                    std::construct_at(&m_storage.m_data[i]);
                }
            } else if (new_size < m_size) {
                // Shrink and destroy in reverse order
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    for (size_type i = m_size; i > new_size; --i) {
                        std::destroy_at(&m_storage.m_data[i - 1]);
                    }
                }
            }
            m_size = new_size;
        }

        // --- STANDARD ORDERED MODIFIERS (O(N)) ---
        
        constexpr iterator erase(const_iterator pos) {
            iterator it = const_cast<iterator>(pos);
            ENGINE_ASSERT(it >= begin() && it < end() && "erase iterator out of bounds!");
            
            // Shift everything left by 1
            if (it + 1 < end()) {
                std::move(it + 1, end(), it);
            }
            
            pop_back();
            return it;
        }

        constexpr iterator insert(const_iterator pos, const T& value) {
            iterator it = const_cast<iterator>(pos);
            ENGINE_ASSERT(it >= begin() && it <= end() && "insert iterator out of bounds!");
            ENGINE_ASSERT(m_size < Capacity && "insert overflow, capacity exceeded!");

            if (it == end()) {
                push_back(value);
                return end() - 1;
            }

            // Construct a dummy element at the end to extend the active lifetime area safely
            std::construct_at(&m_storage.m_data[m_size], std::move(back()));
            ++m_size;

            // Shift everything right by 1
            std::move_backward(it, end() - 2, end() - 1);
            
            // Insert the new value
            *it = value;
            return it;
        }
    };
} 
