namespace Engine::STLContainer {

    // Defaults to std::allocator, but perfectly accepts your custom AlignedAllocator
    template <typename T, std::size_t InlineCapacity, typename Allocator = std::allocator<T>>
    class small_vector {
        static_assert(InlineCapacity > 0, "small_vector inline capacity must be strictly greater than 0.");

    public:
        // --- Standard Library Interface ---
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
        // SVO STATE & STORAGE
        // =====================================================================
        pointer m_data;       // Points to either m_inline_storage or a heap allocation
        size_type m_size;
        size_type m_capacity;

        // C++26 union trick for safe constexpr uninitialized memory
        union Storage {
            struct EmptyState {} m_empty;
            T m_data[InlineCapacity];

            constexpr Storage() : m_empty{} {}
            constexpr ~Storage() {} // Handled explicitly by the vector
        };

        Storage m_inline_storage;
        
        // C++20 standard: ensures zero-byte overhead if the allocator is empty
        [[no_unique_address]] Allocator m_allocator;

        [[nodiscard]] constexpr bool is_inline() const noexcept {
            return m_data == m_inline_storage.m_data;
        }

        constexpr void grow(size_type min_required_capacity) {
            // Standard geometric growth factor (2x)
            size_type new_capacity = m_capacity * 2;
            if (new_capacity < min_required_capacity) {
                new_capacity = min_required_capacity;
            }
            reserve(new_capacity);
        }

    public:
        // --- CONSTRUCTORS ---
        constexpr small_vector() noexcept 
            : m_data(m_inline_storage.m_data), m_size(0), m_capacity(InlineCapacity) {}

        constexpr small_vector(std::initializer_list<T> ilist) 
            : small_vector() {
            reserve(ilist.size());
            for (const auto& item : ilist) {
                unchecked_push_back(item);
            }
        }

        // --- COPY SEMANTICS ---
        constexpr small_vector(const small_vector& other) 
            : m_data(m_inline_storage.m_data), m_size(0), m_capacity(InlineCapacity) {
            
            reserve(other.m_size);
            
            if constexpr (std::is_trivially_copyable_v<T>) {
                if !consteval {
                    std::memcpy(m_data, other.m_data, other.m_size * sizeof(T));
                    m_size = other.m_size;
                    return;
                }
            }
            for (size_type i = 0; i < other.m_size; ++i) {
                std::construct_at(&m_data[i], other.m_data[i]);
            }
            m_size = other.m_size;
        }

        // --- MOVE SEMANTICS (The Magic of SVO) ---
        constexpr small_vector(small_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) 
            : m_data(m_inline_storage.m_data), m_size(0), m_capacity(InlineCapacity) {
            
            if (!other.is_inline()) {
                // If the other vector is on the heap, we execute a blazing fast Pointer Steal O(1)
                m_data = other.m_data;
                m_size = other.m_size;
                m_capacity = other.m_capacity;
                
                // Reset the victim back to its stack memory
                other.m_data = other.m_inline_storage.m_data;
                other.m_size = 0;
                other.m_capacity = InlineCapacity;
            } else {
                // If the other vector is on the stack, we are forced to physically move the elements O(N)
                if constexpr (ENGINE_IS_RELOCATABLE(T)) {
                    if !consteval {
                        std::memcpy(m_data, other.m_data, other.m_size * sizeof(T));
                        m_size = other.m_size;
                        other.m_size = 0;
                        return;
                    }
                }
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_data[i], std::move(other.m_data[i]));
                }
                m_size = other.m_size;
                other.clear();
            }
        }

        // --- DESTRUCTOR ---
        constexpr ~small_vector() noexcept {
            clear();
            if (!is_inline()) {
                std::allocator_traits<Allocator>::deallocate(m_allocator, m_data, m_capacity);
            }
        }

        // --- ASSIGNMENT OPERATORS ---
        constexpr small_vector& operator=(const small_vector& other) {
            if (this != &other) {
                clear();
                reserve(other.m_size);
                
                if constexpr (std::is_trivially_copyable_v<T>) {
                    if !consteval {
                        std::memcpy(m_data, other.m_data, other.m_size * sizeof(T));
                        m_size = other.m_size;
                        return *this;
                    }
                }
                for (size_type i = 0; i < other.m_size; ++i) {
                    std::construct_at(&m_data[i], other.m_data[i]);
                }
                m_size = other.m_size;
            }
            return *this;
        }

        constexpr small_vector& operator=(small_vector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
            if (this != &other) {
                clear();
                
                // If other is on the heap, steal the pointer
                if (!other.is_inline()) {
                    if (!is_inline()) {
                        // Free our own heap memory first
                        std::allocator_traits<Allocator>::deallocate(m_allocator, m_data, m_capacity);
                    }
                    m_data = other.m_data;
                    m_capacity = other.m_capacity;
                    m_size = other.m_size;

                    other.m_data = other.m_inline_storage.m_data;
                    other.m_size = 0;
                    other.m_capacity = InlineCapacity;
                } else {
                    // Other is inline, we must move elements
                    if constexpr (ENGINE_IS_RELOCATABLE(T)) {
                        if !consteval {
                            std::memcpy(m_data, other.m_data, other.m_size * sizeof(T));
                            m_size = other.m_size;
                            other.m_size = 0;
                            return *this;
                        }
                    }
                    for (size_type i = 0; i < other.m_size; ++i) {
                        std::construct_at(&m_data[i], std::move(other.m_data[i]));
                    }
                    m_size = other.m_size;
                    other.clear();
                }
            }
            return *this;
        }

        // --- CAPACITY & LIFETIME ---
        [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }
        [[nodiscard]] constexpr size_type capacity() const noexcept { return m_capacity; }
        [[nodiscard]] constexpr bool empty() const noexcept { return m_size == 0; }

        constexpr void clear() noexcept {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_type i = m_size; i > 0; --i) {
                    std::destroy_at(&m_data[i - 1]);
                }
            }
            m_size = 0;
        }

        // =====================================================================
        // HEAP FALLBACK ALLOCATION LOGIC
        // =====================================================================
        constexpr void reserve(size_type new_capacity) {
            if (new_capacity <= m_capacity) return;

            // 1. Allocate new heap memory
            T* new_data = std::allocator_traits<Allocator>::allocate(m_allocator, new_capacity);

            // 2. Relocate elements
            if constexpr (ENGINE_IS_RELOCATABLE(T)) {
                if !consteval {
                    std::memcpy(new_data, m_data, m_size * sizeof(T));
                } else {
                    for (size_type i = 0; i < m_size; ++i) {
                        std::construct_at(&new_data[i], std::move(m_data[i]));
                    }
                }
            } else {
                for (size_type i = 0; i < m_size; ++i) {
                    std::construct_at(&new_data[i], std::move(m_data[i]));
                    std::destroy_at(&m_data[i]); // Manually call destructor for non-trivial types
                }
            }

            // 3. Free old heap memory (if we weren't inline)
            if (!is_inline()) {
                std::allocator_traits<Allocator>::deallocate(m_allocator, m_data, m_capacity);
            }

            // 4. Update state
            m_data = new_data;
            m_capacity = new_capacity;
        }

        // --- DATA ACCESS ---
        [[nodiscard]] constexpr pointer data() noexcept { return m_data; }
        [[nodiscard]] constexpr const_pointer data() const noexcept { return m_data; }

        [[nodiscard]] constexpr iterator begin() noexcept { return m_data; }
        [[nodiscard]] constexpr iterator end() noexcept { return m_data + m_size; }
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return m_data; }
        [[nodiscard]] constexpr const_iterator end() const noexcept { return m_data + m_size; }

        [[nodiscard]] constexpr reference operator[](size_type index) noexcept {
            ENGINE_ASSERT(index < m_size && "Index out of bounds!");
            ENGINE_ASSUME(index < m_size);
            return m_data[index];
        }

        [[nodiscard]] constexpr const_reference operator[](size_type index) const noexcept {
            ENGINE_ASSERT(index < m_size && "Index out of bounds!");
            ENGINE_ASSUME(index < m_size);
            return m_data[index];
        }

        [[nodiscard]] constexpr reference front() noexcept { ENGINE_ASSUME(m_size > 0); return m_data[0]; }
        [[nodiscard]] constexpr const_reference front() const noexcept { ENGINE_ASSUME(m_size > 0); return m_data[0]; }
        [[nodiscard]] constexpr reference back() noexcept { ENGINE_ASSUME(m_size > 0); return m_data[m_size - 1]; }
        [[nodiscard]] constexpr const_reference back() const noexcept { ENGINE_ASSUME(m_size > 0); return m_data[m_size - 1]; }

        // =====================================================================
        // INSERTION / MODIFIERS
        // =====================================================================
        
        constexpr reference push_back(const T& value) {
            if (m_size == m_capacity) [[unlikely]] { grow(m_size + 1); }
            std::construct_at(&m_data[m_size], value);
            reference ref = m_data[m_size];
            ++m_size;
            return ref;
        }

        constexpr reference push_back(T&& value) {
            if (m_size == m_capacity) [[unlikely]] { grow(m_size + 1); }
            std::construct_at(&m_data[m_size], std::move(value));
            reference ref = m_data[m_size];
            ++m_size;
            return ref;
        }

        constexpr reference unchecked_push_back(const T& value) noexcept {
            ENGINE_ASSUME(m_size < m_capacity);
            std::construct_at(&m_data[m_size], value);
            reference ref = m_data[m_size];
            ++m_size;
            return ref;
        }

        constexpr reference unchecked_push_back(T&& value) noexcept {
            ENGINE_ASSUME(m_size < m_capacity);
            std::construct_at(&m_data[m_size], std::move(value));
            reference ref = m_data[m_size];
            ++m_size;
            return ref;
        }

        template <typename... Args>
        constexpr reference emplace_back(Args&&... args) {
            if (m_size == m_capacity) [[unlikely]] { grow(m_size + 1); }
            std::construct_at(&m_data[m_size], std::forward<Args>(args)...);
            reference ref = m_data[m_size];
            ++m_size;
            return ref;
        }

        constexpr void pop_back() noexcept {
            ENGINE_ASSERT(m_size > 0 && "small_vector underflow!");
            --m_size;
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_at(&m_data[m_size]);
            }
        }

        // =====================================================================
        // C++26 RANGES API
        // =====================================================================
        template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>, T>
        constexpr void append_range(R&& rg) {
            if constexpr (std::ranges::sized_range<R> || std::ranges::forward_range<R>) {
                size_type dist = static_cast<size_type>(std::ranges::distance(rg));
                reserve(m_size + dist); // Safely scale memory up

                if constexpr (std::ranges::contiguous_range<R> && std::is_trivially_copyable_v<T>) {
                    if !consteval {
                        if (dist > 0) {
                            std::memcpy(m_data + m_size, std::ranges::data(rg), dist * sizeof(T));
                            m_size += dist;
                        }
                        return;
                    }
                }
                for (auto&& e : rg) {
                    std::construct_at(&m_data[m_size++], std::forward<decltype(e)>(e));
                }
            } else {
                for (auto&& e : rg) {
                    push_back(std::forward<decltype(e)>(e));
                }
            }
        }

        // =========================================
        // RESIZE UNINITIALIZED (NO DEFAULT VALUES)
        // =========================================
        /*
            - Expands a vector's active size by allocating more heap memory without filling it with default values (initialize) or zeroing out the new memory.
            - Reserved memory is left as uninitialized garbage until its explicitly overwritten.
            - Allocate the space and set the size before a for loop runs to improve performance by 50% for insertions.
            - Used for performing Bulk Writes where you guarantee that every single newly allocated byte will be overwritten before its ever read.
            - Never use on objects, its dangerous because it will try to delete memory it does not own (segfault/access violation).
            - e.g., reading asset files, textures, binary save data directly from the disk into the memory buffer.
            - e.g., preparing a staging buffer where the GPU will dump data directly into RAM.
        */
        
        /// @brief Resizes the vector without initializing elements. 
        /// @note ONLY use this for Trivially Copyable types (like physics transforms/vectors). 
        constexpr void resize_uninitialized(size_type new_size) {
            // std::is_trivially_copyable_v<T> ensures that it can only be used on pure data (floats, matrices, basic structs) that does not care if the memory starts out as garbage.
            static_assert(std::is_trivially_copyable_v<T>, 
                "resize_uninitialized can only be called on Trivially Copyable types.");
            
            if (new_size > m_capacity) [[unlikely]] {
                grow(new_size);
            }
            m_size = new_size;
        }

        // --- FAST ENGINE ERASURE (O(1)) ---
        constexpr void erase_unsorted(const_iterator pos) {
            iterator it = const_cast<iterator>(pos);
            ENGINE_ASSERT(it >= begin() && it < end() && "erase_unsorted iterator out of bounds!");
            if (it != end() - 1) *it = std::move(back()); 
            pop_back(); 
        }
        
        constexpr void erase_unsorted_by_index(size_type index) {
            ENGINE_ASSERT(index < m_size && "erase_unsorted index out of bounds!");
            if (index != m_size - 1) m_data[index] = std::move(back());
            pop_back();
        }
    };
}
