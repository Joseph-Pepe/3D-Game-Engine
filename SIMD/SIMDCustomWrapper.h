#pragma once
#include <concepts>
#include <cstdint>
#include <type_traits>

// ==================================================
// INSTRUCTION SET ARCHITECTURES (ISA)
// ==================================================
/*
    - x86_64 (AMD Zen 2): Xbox Series X/S & PS5 | AVX2 (256-bit registers)
    - ARM64 (ARM): Nintendo Switch 2 (Nvidia Tegra) & Apple Silicon (M1/M2/M3) | ARM NEON (128-bit registers)
    - Legacy: Baseline Legacy PC | SSE4.1 (128 bit registers)
*/

// --- HARDWARE DETECTION & INCLUDES ---
#if defined(__AVX2__)
    #include <immintrin.h>
    #define ENGINE_ARCH_AVX2 1
#elif defined(__aarch64__)
    #include <arm_neon.h>
    #define ENGINE_ARCH_NEON 1
#else
    #error "Unsupported architecture. Fallback SSE4.1 required."
#endif

namespace Engine::ISAArch {
    // ==========================================
    // ABI NAMESPACE (Hardware Tags)
    // ==========================================
    namespace simd_abi {
        struct scalar {};
        struct avx2 {};
        struct neon {};

        // C++26 'native' alias: Automatically deduces the best hardware vector length at compile time based on your compiler flags (e.g., /arch:AVX2).
        template <typename T>
        #if ENGINE_ARCH_AVX2
            using native = avx2;     // Xbox/PS5/PC
        #elif ENGINE_ARCH_NEON
            using native = neon;     // Nintendo Switch 2 / Apple Silicon
        #else
            using native = scalar;   // Legacy PC
        #endif
    }

    // ==========================================
    // BACKEND INTRINSICS (Traits/Storage)
    // ==========================================
    namespace detail {
        // Primary template (Undefined)
        template <typename T, typename Abi> struct simd_traits;

        // --- AVX2 BACKEND (Xbox Series X, PS5, PC) ---
        template <> struct simd_traits<float, simd_abi::avx2> {
            using register_type = __m256;
            using mask_type     = __m256; // AVX2 masks are bit patterns in identical registers
            static constexpr int size = 8;
            
            static inline register_type broadcast(float v) { return _mm256_set1_ps(v); }
            static inline register_type load(const float* mem) { return _mm256_loadu_ps(mem); }
            static inline void store(float* mem, register_type v) { _mm256_storeu_ps(mem, v); }
            
            static inline register_type add(register_type a, register_type b) { return _mm256_add_ps(a, b); }
            static inline register_type mul(register_type a, register_type b) { return _mm256_mul_ps(a, b); }
            static inline register_type sub(register_type a, register_type b) { return _mm256_sub_ps(a, b); }
            static inline register_type div(register_type a, register_type b) { return _mm256_div_ps(a, b); }

            static inline register_type min(register_type a, register_type b) { return _mm256_min_ps(a, b); }
            static inline register_type max(register_type a, register_type b) { return _mm256_max_ps(a, b); }

            static inline register_type rsqrt(register_type a) { return _mm256_rsqrt_ps(a); } // 1 / sqrt(x)
            static inline register_type rcp(register_type a) { return _mm256_rcp_ps(a); }     // 1 / x

            static inline register_type abs(register_type a) { 
                // Clear the sign bit using bitwise AND
                return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF))); 
            }
            static inline register_type floor(register_type a) { return _mm256_floor_ps(a); }
            static inline register_type ceil(register_type a) { return _mm256_ceil_ps(a); }
            
            // Relational Intrinsic
            static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
            static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
            static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }

            // Mask Logic
            static inline mask_type mask_not(mask_type a) { 
                // XOR with all 1s (0xFFFFFFFF) flips every bit
                return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1))); 
            }
            static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_ps(a, b); }
            static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_ps(a, b); }
            
            // Branchless Conditional Blending
            static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                return _mm256_blendv_ps(false_v, true_v, mask);
            }

            // Hardware Math
            static inline register_type fmadd(register_type a, register_type b, register_type c) {
                return _mm256_fmadd_ps(a, b, c); // (a * b) + c
            }
            static inline register_type sqrt(register_type a) {
                return _mm256_sqrt_ps(a);
            }

            // SIMD Casting AVX2 doesn't have a direct float-to-unsigned-int instruction, it only has float-to-signed-int. 
            template <typename Target>
            static inline __m256i cast_to(register_type a) {
                static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t casting is currently implemented.");
                return _mm256_cvttps_epi32(a); 
            }

            // Horizontal Reduction
            // AVX2 doesn't have a single-instruction horizontal add across 256 bits, so we fold it in half repeatedly.
            static inline float reduce_add(register_type a) {
                // Step 1: Fold 256-bit into 128-bit
                __m128 lo = _mm256_castps256_ps128(a);
                __m128 hi = _mm256_extractf128_ps(a, 1);
                lo = _mm_add_ps(lo, hi);
                
                // Step 2: Fold 128-bit down to 64-bit, then down to 32-bit scalar
                __m128 shuf = _mm_movehdup_ps(lo);
                __m128 sums = _mm_add_ps(lo, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                
                return _mm_cvtss_f32(sums);
            }

            static inline bool mask_any(mask_type a) { 
                return _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0; 
            }
            static inline bool mask_all(mask_type a) { 
                return _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF; 
            }
        };

        // --- AVX2 UINT32 TRAITS ---
        template <> struct simd_traits<uint32_t, simd_abi::avx2> {
            using register_type = __m256i;
            using mask_type     = __m256i; 
            static constexpr int size = 8;
            
            // Memory
            static inline register_type broadcast(uint32_t v) { return _mm256_set1_epi32(v); }
            static inline register_type load(const uint32_t* mem) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem)); }
            static inline void store(uint32_t* mem, register_type v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v); }
            
            static inline register_type add(register_type a, register_type b) { return _mm256_add_epi32(a, b); }
            static inline register_type sub(register_type a, register_type b) { return _mm256_sub_epi32(a, b); }
            static inline register_type mul(register_type a, register_type b) { return _mm256_mullo_epi32(a, b); }

            static inline register_type min(register_type a, register_type b) { return _mm256_min_epu32(a, b); }
            static inline register_type max(register_type a, register_type b) { return _mm256_max_epu32(a, b); }

            // Bitwise Math (Required for your expandBits_SIMD function)
            static inline register_type bit_or(register_type a, register_type b) { return _mm256_or_si256(a, b); }
            static inline register_type bit_and(register_type a, register_type b) { return _mm256_and_si256(a, b); }
            static inline register_type shift_l(register_type a, int imm) { return _mm256_slli_epi32(a, imm); }
            
            static inline mask_type cmp_gt(register_type a, register_type b) { return _mm256_cmpgt_epi32(a, b); }
            static inline mask_type cmp_lt(register_type a, register_type b) { return _mm256_cmpgt_epi32(b, a); } // Flipped operands for Less-Than
            static inline mask_type cmp_eq(register_type a, register_type b) { return _mm256_cmpeq_epi32(a, b); }

            static inline mask_type mask_not(mask_type a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
            static inline mask_type mask_and(mask_type a, mask_type b) { return _mm256_and_si256(a, b); }
            static inline mask_type mask_or(mask_type a, mask_type b) { return _mm256_or_si256(a, b); }

            static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                // Integer blending inherently looks at the highest bit of each byte
                return _mm256_blendv_epi8(false_v, true_v, mask);
            }

            // Horizontal Integer Reduction
            static inline uint32_t reduce_add(register_type a) {
                __m128i lo = _mm256_castsi256_si128(a);
                __m128i hi = _mm256_extracti128_si256(a, 1);
                lo = _mm_add_epi32(lo, hi);
                
                // Shuffle and fold 128-bit down to scalar
                __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
                __m128i sums = _mm_add_epi32(lo, shuf);
                shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
                sums = _mm_add_epi32(sums, shuf);
                
                return _mm_cvtsi128_si32(sums);
            }

            static inline bool mask_any(mask_type a) { 
                return _mm256_movemask_ps(_mm256_castsi256_ps(a)) != 0; 
            }
            static inline bool mask_all(mask_type a) { 
                return _mm256_movemask_ps(_mm256_castsi256_ps(a)) == 0xFF; 
            }
        };

        // --- NEON BACKEND (Nintendo Switch 2, Apple Silicon) ---
        template <> struct simd_traits<float, simd_abi::neon> {
            using register_type = float32x4_t;
            using mask_type     = uint32x4_t; // NEON strictly separates math and mask register types
            static constexpr int size = 4;
            
            static inline register_type broadcast(float v) { return vdupq_n_f32(v); }
            static inline register_type load(const float* mem) { return vld1q_f32(mem); }
            static inline void store(float* mem, register_type v) { vst1q_f32(mem, v); }
            
            static inline register_type add(register_type a, register_type b) { return vaddq_f32(a, b); }
            static inline register_type mul(register_type a, register_type b) { return vmulq_f32(a, b); }
            static inline register_type sub(register_type a, register_type b) { return vsubq_f32(a, b); }
            static inline register_type div(register_type a, register_type b) { return vdivq_f32(a, b); }

            static inline register_type min(register_type a, register_type b) { return vminq_f32(a, b); }
            static inline register_type max(register_type a, register_type b) { return vmaxq_f32(a, b); }

            static inline register_type rsqrt(register_type a) { return vrsqrteq_f32(a); }
            static inline register_type rcp(register_type a) { return vrecpeq_f32(a); }

            static inline register_type abs(register_type a) { return vabsq_f32(a); }
            static inline register_type floor(register_type a) { return vrndmq_f32(a); } // Round towards Minus infinity
            static inline register_type ceil(register_type a) { return vrndpq_f32(a); }  // Round towards Plus infinity
            
            static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_f32(a, b); }
            static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_f32(a, b); }
            static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_f32(a, b); }

            static inline mask_type mask_not(mask_type a) { return vmvnq_u32(a); } // NEON Bitwise NOT
            static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u32(a, b); }
            static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u32(a, b); }
            
            static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                return vbslq_f32(mask, true_v, false_v);
            }

            // Hardware Math
            static inline register_type fmadd(register_type a, register_type b, register_type c) {
                return vfmaq_f32(c, a, b); // accumulates (a * b) into c
            }
            static inline register_type sqrt(register_type a) {
                return vsqrtq_f32(a);
            }

            // SIMD Casting: NEON explicitly supports float-to-unsigned-int conversion natively.
            template <typename Target>
            static inline uint32x4_t cast_to(register_type a) {
                static_assert(std::is_same_v<Target, uint32_t>, "Only float-to-uint32_t casting is currently implemented.");
                return vcvtq_u32_f32(a);
            }

            // Horizontal Reduction: Because the Switch 2 and Apple Silicon use AArch64, we don't have to do the folding dance like on AVX2. 
            // We get to use a gorgeous, single-cycle hardware reduction instruction!
            static inline float reduce_add(register_type a) {
                return vaddvq_f32(a); 
            }

            static inline bool mask_any(mask_type a) { 
                // If the maximum value across the vector is > 0, at least one lane is true
                return vmaxvq_u32(a) > 0; 
            }
            static inline bool mask_all(mask_type a) { 
                // If the minimum value across the vector is > 0, all lanes are true
                return vminvq_u32(a) > 0; 
            }
        };

        // --- NEON UINT32 TRAITS ---
        template <> struct simd_traits<uint32_t, simd_abi::neon> {
            using register_type = uint32x4_t;
            using mask_type     = uint32x4_t;
            static constexpr int size = 4;
            
            // Memory
            static inline register_type broadcast(uint32_t v) { return vdupq_n_u32(v); }
            static inline register_type load(const uint32_t* mem) { return vld1q_u32(mem); }
            static inline void store(uint32_t* mem, register_type v) { vst1q_u32(mem, v); }

            static inline register_type add(register_type a, register_type b) { return vaddq_u32(a, b); }
            static inline register_type sub(register_type a, register_type b) { return vsubq_u32(a, b); }
            static inline register_type mul(register_type a, register_type b) { return vmulq_u32(a, b); }

            static inline register_type min(register_type a, register_type b) { return vminq_f32(a, b); }
            static inline register_type max(register_type a, register_type b) { return vmaxq_f32(a, b); }
            
            // Bitwise Math
            static inline register_type bit_or(register_type a, register_type b) { return vorrq_u32(a, b); }
            static inline register_type bit_and(register_type a, register_type b) { return vandq_u32(a, b); }
            static inline register_type shift_l(register_type a, int imm) { 
                // Broadcast the scalar 'imm' into a signed 32x4 vector, then shift
                return vshlq_u32(a, vdupq_n_s32(imm)); 
            }
            
            static inline mask_type cmp_gt(register_type a, register_type b) { return vcgtq_u32(a, b); }
            static inline mask_type cmp_lt(register_type a, register_type b) { return vcltq_u32(a, b); }
            static inline mask_type cmp_eq(register_type a, register_type b) { return vceqq_u32(a, b); }

            static inline mask_type mask_not(mask_type a) { return vmvnq_u32(a); }
            static inline mask_type mask_and(mask_type a, mask_type b) { return vandq_u32(a, b); }
            static inline mask_type mask_or(mask_type a, mask_type b) { return vorrq_u32(a, b); }

            static inline register_type blend(mask_type mask, register_type true_v, register_type false_v) {
                return vbslq_u32(mask, true_v, false_v);
            }

            // Horizontal Integer Reduction
            static inline uint32_t reduce_add(register_type a) {
                return vaddvq_u32(a); 
            }

            static inline bool mask_any(mask_type a) { 
                // If the maximum value across the vector is > 0, at least one lane is true
                return vmaxvq_u32(a) > 0; 
            }
            static inline bool mask_all(mask_type a) { 
                // If the minimum value across the vector is > 0, all lanes are true
                return vminvq_u32(a) > 0; 
            }
        };
    }

    // ==========================================
    // C++26 MASK FRONTEND
    // ==========================================
    template <typename T, typename Abi = simd_abi::native<T>> class simd;

    // C++26 strictly requires a separate mask type to safely handle SIMD conditionals
    template <typename T, typename Abi = simd_abi::native<T>>
    class simd_mask {
    private:
        using Traits = detail::simd_traits<T, Abi>;
        typename Traits::mask_type m_mask;

        // Allow masks of different types to access each other's private data for Mixed-Type casting
        template <typename U, typename UAbi> friend class simd_mask;

        // Only intrinsics can construct a mask directly
        friend class simd<T, Abi>;
        explicit inline simd_mask(typename Traits::mask_type mask) : m_mask(mask) {}

    public:
        simd_mask() = default;

        // --- MIXED-TYPE MASK CASTING ---
        template <typename U>
        inline simd_mask<U, Abi> cast_to() const {
            if constexpr (std::is_same_v<T, U>) {
                return *this;
            } else if constexpr (std::is_same_v<T, uint32_t> && std::is_same_v<U, float>) {
                #ifdef ENGINE_ARCH_AVX2
                    return simd_mask<U, Abi>(_mm256_castsi256_ps(m_mask));
                #elif defined(ENGINE_ARCH_NEON)
                    return simd_mask<U, Abi>(m_mask);
                #endif
            } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, uint32_t>) {
                #ifdef ENGINE_ARCH_AVX2
                    return simd_mask<U, Abi>(_mm256_castps_si256(m_mask));
                #elif defined(ENGINE_ARCH_NEON)
                    return simd_mask<U, Abi>(m_mask);
                #endif
            } else {
                static_assert(sizeof(U) == 0, "Unsupported mask cast.");
            }
        }

        // --- LOGICAL OPERATORS ---
        friend inline simd_mask operator!(const simd_mask& a) {
            return simd_mask(Traits::mask_not(a.m_mask));
        }

        template <typename U>
        inline simd_mask operator&&(const simd_mask<U, Abi>& b) const {
            return simd_mask(Traits::mask_and(m_mask, b.template cast_to<T>().m_mask));
        }

        template <typename U>
        inline simd_mask operator||(const simd_mask<U, Abi>& b) const {
            return simd_mask(Traits::mask_or(m_mask, b.template cast_to<T>().m_mask));
        }

        friend inline bool any_of(const simd_mask& m) { return Traits::mask_any(m.m_mask); }
        friend inline bool all_of(const simd_mask& m) { return Traits::mask_all(m.m_mask); }
        friend inline bool none_of(const simd_mask& m) { return !Traits::mask_any(m.m_mask); }
    };

    // ==========================================
    // C++26 SIMD FRONTEND
    // ==========================================
    template <typename T, typename Abi>
    class simd {
    private:
        using Traits = detail::simd_traits<T, Abi>;
        typename Traits::register_type m_data;

        // Internal constructor for operations to bypass memory
        explicit inline simd(typename Traits::register_type data) : m_data(data) {}

    public:
        // C++26 Type Definitions
        using value_type = T;
        using abi_type   = Abi;
        using mask_type  = simd_mask<T, Abi>;

        static constexpr int size() { return Traits::size; }

        // --- HARDWARE INTEROP ---
        typename Traits::register_type native_handle() const { return m_data; }
        static simd from_native(typename Traits::register_type raw) { return simd(raw); }

        // --- CONSTRUCTORS ---
        // C++26 guarantees default simds are uninitialized, just like raw floats!
        simd() = default; 
        
        // Broadcast Constructor
        simd(T value) : m_data(Traits::broadcast(value)) {} 
        
        // C++26 Memory Load (P1928 allows implicit load from memory)
        explicit simd(const T* mem) : m_data(Traits::load(mem)) {}

        // --- MEMORY STORE ---
        void copy_to(T* mem) const { Traits::store(mem, m_data); }

        friend inline simd min(const simd& a, const simd& b) { return simd(Traits::min(a.m_data, b.m_data)); }
        friend inline simd max(const simd& a, const simd& b) { return simd(Traits::max(a.m_data, b.m_data)); }
        friend inline simd clamp(const simd& v, const simd& lo, const simd& hi) { return min(max(v, lo), hi); }

        friend inline simd rsqrt(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::rsqrt(a.m_data)); }
        friend inline simd rcp(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::rcp(a.m_data)); }

        friend inline simd abs(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::abs(a.m_data)); }
        friend inline simd floor(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::floor(a.m_data)); }
        friend inline simd ceil(const simd& a) requires std::is_floating_point_v<T> { return simd(Traits::ceil(a.m_data)); }

        // --- C++26 OPERATORS (HIDDEN FRIENDS) ---
        // Using friends prevents ambiguous overload resolution and ensures identical inline compilation

        // --- ARITHMETIC OPERATORS ---
        friend inline simd operator+(const simd& a, const simd& b) {
            return simd(Traits::add(a.m_data, b.m_data));
        }

        friend inline simd operator-(const simd& a, const simd& b) {
            return simd(Traits::sub(a.m_data, b.m_data));
        }

        friend inline simd operator*(const simd& a, const simd& b) {
            return simd(Traits::mul(a.m_data, b.m_data));
        }

        // Division explicitly restricted to floating point traits
        friend inline simd operator/(const simd& a, const simd& b) requires std::is_floating_point_v<T> {
            return simd(Traits::div(a.m_data, b.m_data));
        }

        inline simd& operator+=(const simd& rhs) { m_data = Traits::add(m_data, rhs.m_data); return *this; }
        inline simd& operator-=(const simd& rhs) { m_data = Traits::sub(m_data, rhs.m_data); return *this; }
        inline simd& operator*=(const simd& rhs) { m_data = Traits::mul(m_data, rhs.m_data); return *this; }
        inline simd& operator/=(const simd& rhs) requires std::is_floating_point_v<T> { m_data = Traits::div(m_data, rhs.m_data); return *this; }

        // --- RELATIONAL & CONDITIONALS ---
        // In C++26, comparison returns a mask, not a boolean
        friend inline mask_type operator>(const simd& a, const simd& b) {
            return mask_type(Traits::cmp_gt(a.m_data, b.m_data));
        }
        friend inline mask_type operator<(const simd& a, const simd& b) { return mask_type(Traits::cmp_lt(a.m_data, b.m_data)); }
        friend inline mask_type operator==(const simd& a, const simd& b) { return mask_type(Traits::cmp_eq(a.m_data, b.m_data)); }

        // Google Highway's "IfThenElse" equivalent
        // The C++26 proposal suggests letting `mask ? a : b` work natively, but 
        // functionally it requires a blend operation.
        static inline simd choose(const mask_type& mask, const simd& true_val, const simd& false_val) {
            return simd(Traits::blend(mask.m_mask, true_val.m_data, false_val.m_data));
        }

        // --- MATH FUNCTIONS ---

        // Replicates: std::fma(a, b, c)
        friend inline simd fma(const simd& a, const simd& b, const simd& c) {
            return simd(Traits::fmadd(a.m_data, b.m_data, c.m_data));
        }

        // Replicates: std::sqrt(a)
        friend inline simd sqrt(const simd& a) {
            return simd(Traits::sqrt(a.m_data));
        }

        // C++26 Portable Horizontal Reduction (Hidden Friend)
        // By declaring this as a friend inside the class, the compiler knows exactly 
        // which 'reduce' to call when you pass it a simd<float>.
        friend inline T reduce(const simd& a) {
            return Traits::reduce_add(a.m_data);
        }

        // --- BITWISE MATH (Restricted to Integers) ---

        // C++20 (Concpets): Add 'requires std::is_integral_v<T>' to the bitwise hidden friends to restrict bitwise math exclusively to integer types.
        friend inline simd operator|(const simd& a, const simd& b) requires std::is_integral_v<T> {
            return simd(Traits::bit_or(a.m_data, b.m_data));
        }

        friend inline simd operator&(const simd& a, const simd& b) requires std::is_integral_v<T> {
            return simd(Traits::bit_and(a.m_data, b.m_data));
        }

        friend inline simd operator<<(const simd& a, int shift) requires std::is_integral_v<T> {
            return simd(Traits::shift_l(a.m_data, shift));
        }
    };

    // =========================================================
    // PROXY OBJECT & SIMD CASTING 
    // =========================================================

    // A lightweight proxy object that executes masked operations, overloads operators to perform a branchless hardware blend.
    template <typename T, typename Abi>
    struct WhereExpression {
        const simd_mask<T, Abi>& mask;
        simd<T, Abi>& target; // Reference to the batch we are modifying

        // SIMD batch assignments

        // Replicates: std::simd::where(mask, x) = 0.0f;
        inline void operator=(T scalar) {
            target = simd<T, Abi>::choose(mask, simd<T, Abi>(scalar), target);
        }
        inline void operator+=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target + rhs, target);
        }
        inline void operator-=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target - rhs, target);
        }

        // Replicates: std::simd::where(mask, x) *= invLen;
        inline void operator*=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, target * rhs, target);
        }

        inline void operator/=(const simd<T, Abi>& rhs) requires std::is_floating_point_v<T> { 
            target = simd<T, Abi>::choose(mask, target / rhs, target); 
        }

        // Assign another SIMD Batch
        inline void operator=(const simd<T, Abi>& rhs) {
            target = simd<T, Abi>::choose(mask, rhs, target);
        }
    };

    // The free function that mirrors std::simd::where
    template <typename T, typename Abi>
    inline WhereExpression<T, Abi> where(const simd_mask<T, Abi>& mask, simd<T, Abi>& target) {
        return WhereExpression<T, Abi>{mask, target};
    }

    // The C++26 SIMD Casting.
    template <typename ToType, typename FromType, typename Abi>
    inline simd<ToType, Abi> simd_cast(const simd<FromType, Abi>& from) {
        // Fetch the raw hardware cast from the traits, and explicitly construct the new SIMD type.
        // (Assuming from_data is accessed via a getter or friend declaration).
        auto raw_cast = detail::simd_traits<FromType, Abi>::template cast_to<ToType>(from.native_handle());
        return simd<ToType, Abi>::from_native(raw_cast);
    }
}

/*
// Use the custom C++26-compliant wrapper
using NativeFloatSIMDBatch = Engine::ISAArch::simd<float>;
using NativeUIntBatch      = Engine::ISAArch::simd<uint32_t>;

constexpr std::size_t NATIVE_BATCH_SIZE = NativeFloatSIMDBatch::size();
constexpr std::size_t NATIVE_SIMD_BATCH_ALIGN = alignof(NativeFloatSIMDBatch);

struct alignas(NATIVE_SIMD_BATCH_ALIGN) SIMDVector3D {
    NativeFloatSIMDBatch x, y, z;

    // Dot Product with explicit hardware FMA mapping
    FORCE_INLINE NativeFloatSIMDBatch dot_fma(const NativeFloatSIMDBatch& bx, const NativeFloatSIMDBatch& by, const NativeFloatSIMDBatch& bz) const {
        NativeFloatSIMDBatch res = x * bx;
        res = Engine::ISAArch::fma(y, by, res);
        res = Engine::ISAArch::fma(z, bz, res);
        return res;
    }

    FORCE_INLINE void normalize() {
        NativeFloatSIMDBatch sqLen = length_sq();
        NativeFloatSIMDBatch epsilon = 1e-8f;
        
        auto validMask = sqLen > epsilon;

        // Custom where-proxy seamlessly executing branchless assignment
        NativeFloatSIMDBatch safeSqLen = sqLen;
        Engine::ISAArch::where(!validMask, safeSqLen) = 1.0f; 

        // Emits hardware square root seamlessly
        NativeFloatSIMDBatch invLen = 1.0f / Engine::ISAArch::sqrt(safeSqLen);

        // Masked multiplications execute cleanly via the proxy overload
        Engine::ISAArch::where(validMask, x) *= invLen;
        Engine::ISAArch::where(validMask, y) *= invLen;
        Engine::ISAArch::where(validMask, z) *= invLen;

        Engine::ISAArch::where(!validMask, x) = 0.0f;
        Engine::ISAArch::where(!validMask, y) = 0.0f;
        Engine::ISAArch::where(!validMask, z) = 0.0f;
    }
};
*/
