#pragma once

#include <utility>
#include <type_traits>
#include <new>
#include <cstddef>
#include <cassert>

// Default capacity of 64 bytes is the size of one cache line.
// This is usually enough to capture ~8 pointers or a couple of matrices.
template <typename Signature, std::size_t Capacity = 64>
class FixedFunction;

template <typename Ret, typename... Args, std::size_t Capacity>
class FixedFunction<Ret(Args...), Capacity> {
private:
    // 1. The VTable Interface (Type Erasure)
    // We don't know what type of lambda we are holding, so we store function 
    // pointers to do the heavy lifting (invoke, move, destroy).
    struct VTable {
        Ret (*invoke)(void*, Args&&...);
        void (*move)(void*, void*);
        void (*destroy)(void*);
    };

    // 2. The Internal Storage
    // alignas(std::max_align_t) guarantees the memory is properly aligned for 
    // ANY type you could possibly capture (including AVX __m256 types if aligned manually).
    alignas(std::max_align_t) std::byte storage_[Capacity];
    const VTable* vtable_ = nullptr;

    // 3. The Implementation Generator
    // The compiler generates a unique version of this struct for every specific lambda type.
    template <typename Callable>
    struct VTableImpl {
        static Ret invoke(void* storage, Args&&... args) {
            // Re-cast the raw memory back into the exact lambda type and call it
            Callable* callable = static_cast<Callable*>(storage);
            return (*callable)(std::forward<Args>(args)...);
        }

        static void move(void* dest, void* src) {
            Callable* srcCallable = static_cast<Callable*>(src);
            // Placement new: Move-construct the lambda into the new destination memory
            new (dest) Callable(std::move(*srcCallable));
        }

        static void destroy(void* storage) {
            // Explicitly call the destructor of the captured variables
            static_cast<Callable*>(storage)->~Callable();
        }

        // A static constexpr instance of the VTable for this specific lambda
        static constexpr VTable table = { &invoke, &move, &destroy };
    };

public:
    // --- Constructors & Destructors ---

    FixedFunction() noexcept = default;
    FixedFunction(std::nullptr_t) noexcept {}

    // The Magic Constructor: Accepts any lambda or callable object
    template <typename Callable, 
              typename Decayed = std::decay_t<Callable>,
              // SFINAE: Don't let this constructor hijack the copy/move constructors
              typename = std::enable_if_t<!std::is_same_v<Decayed, FixedFunction>>>
    FixedFunction(Callable&& callable) {
        
        // STRICT GUARANTEE: If the lambda is too big, fail compilation!
        // No silent heap allocations allowed.
        static_assert(sizeof(Decayed) <= Capacity, 
            "FATAL: Lambda closure is too large for FixedFunction! Increase capacity or capture by reference/pointer.");
        
        // STRICT GUARANTEE: Must respect hardware alignment.
        static_assert(alignof(Decayed) <= alignof(std::max_align_t), 
            "FATAL: Lambda capture alignment exceeds maximum safe alignment.");

        // Construct the lambda directly inside our pre-allocated byte array
        new (storage_) Decayed(std::forward<Callable>(callable));
        
        // Point the vtable to the specific implementation for this lambda type
        vtable_ = &VTableImpl<Decayed>::table;
    }

    ~FixedFunction() {
        if (vtable_) {
            vtable_->destroy(storage_);
        }
    }

    // --- Move Semantics (No Copying allowed for absolute performance) ---

    FixedFunction(FixedFunction&& other) noexcept {
        if (other.vtable_) {
            other.vtable_->move(storage_, other.storage_);
            vtable_ = other.vtable_;
            
            other.vtable_->destroy(other.storage_);
            other.vtable_ = nullptr;
        }
    }

    FixedFunction& operator=(FixedFunction&& other) noexcept {
        if (this != &other) {
            // Destroy our current lambda
            if (vtable_) {
                vtable_->destroy(storage_);
                vtable_ = nullptr;
            }

            // Steal the other lambda
            if (other.vtable_) {
                other.vtable_->move(storage_, other.storage_);
                vtable_ = other.vtable_;
                
                other.vtable_->destroy(other.storage_);
                other.vtable_ = nullptr;
            }
        }
        return *this;
    }

    // Disable Copying
    FixedFunction(const FixedFunction&) = delete;
    FixedFunction& operator=(const FixedFunction&) = delete;

    // --- Execution ---

    explicit operator bool() const noexcept {
        return vtable_ != nullptr;
    }

    Ret operator()(Args... args) {
        assert(vtable_ != nullptr && "Attempted to invoke an empty FixedFunction!");
        return vtable_->invoke(storage_, std::forward<Args>(args)...);
    }
};
