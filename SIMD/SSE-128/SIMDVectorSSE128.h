#pragma once

#include <vector>
#include <immintrin.h>

#if defined(__x86_64__) || defined(_M_X64)
    // A "Manager" class for a large collection of vectors (processes 4 vectors simultaneously).
    class VectorManagerSOA {
    public:
        // Aligned vectors for X, Y, and Z components
        std::vector<float> xs, ys, zs;

        VectorManagerSOA(size_t count) {
            // Ensure count is a multiple of 4
            size_t paddedCount = (count + 3) & ~3;
            xs.resize(paddedCount, 1.0f);
            ys.resize(paddedCount, 2.0f);
            zs.resize(paddedCount, 3.0f);
        }

        FORCE_INLINE void processBatch(float stepX, float stepY, float stepZ) {
            __m128 sX = _mm_set1_ps(stepX);
            __m128 sY = _mm_set1_ps(stepY);
            __m128 sZ = _mm_set1_ps(stepZ);

            for (size_t i = 0; i < xs.size(); i += 4) {
                // 1. Load 4 vectors at once
                __m128 vX = _mm_load_ps(&xs[i]);
                __m128 vY = _mm_load_ps(&ys[i]);
                __m128 vZ = _mm_load_ps(&zs[i]);

                // 2. Addition (4 vectors at once!)
                vX = _mm_add_ps(vX, sX);
                vY = _mm_add_ps(vY, sY);
                vZ = _mm_add_ps(vZ, sZ);

                // 3. Dot Product (SOA dot product is just 3 muls and 2 adds)
                // d = (x1*x2) + (y1*y2) + (z1*z2)
                __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vX, sX), 
                                                _mm_mul_ps(vY, sY)), 
                                        _mm_mul_ps(vZ, sZ));

                // 4. Update X (A.x += d * small)
                vX = _mm_add_ps(vX, _mm_mul_ps(dot, _mm_set1_ps(0.00001f)));

                // 5. Cross Product (The SOA "Magic")
                // Res.x = (y * b.z) - (z * b.y)
                __m128 resX = _mm_sub_ps(_mm_mul_ps(vY, sZ), _mm_mul_ps(vZ, sY));
                __m128 resY = _mm_sub_ps(_mm_mul_ps(vZ, sX), _mm_mul_ps(vX, sZ));
                __m128 resZ = _mm_sub_ps(_mm_mul_ps(vX, sY), _mm_mul_ps(vY, sX));

                // 6. Store 4 results back
                _mm_store_ps(&xs[i], resX);
                _mm_store_ps(&ys[i], resY);
                _mm_store_ps(&zs[i], resZ);
            }
        }
    };
#endif
