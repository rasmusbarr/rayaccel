//
//  Renderer.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-26.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_Renderer_h
#define Renderer_Renderer_h

#include <RayAccelerator.h>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#define ALIGNED(n) __declspec(align(n))
#define FORCEINLINE __forceinline
#else
#define ALIGNED(n) __attribute__((aligned(n)))
#define FORCEINLINE __attribute__((always_inline))
#endif

#define _MM256_TRANSPOSE4_PS(row0, row1, row2, row3) \
do {\
__m256 tmp3, tmp2, tmp1, tmp0;\
tmp0 = _mm256_unpacklo_ps((row0), (row1));\
tmp2 = _mm256_unpacklo_ps((row2), (row3));\
tmp1 = _mm256_unpackhi_ps((row0), (row1));\
tmp3 = _mm256_unpackhi_ps((row2), (row3));\
(row0) = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(1,0,1,0));\
(row1) = _mm256_shuffle_ps(tmp0, tmp2, _MM_SHUFFLE(3,2,3,2));\
(row2) = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(1,0,1,0));\
(row3) = _mm256_shuffle_ps(tmp1, tmp3, _MM_SHUFFLE(3,2,3,2));\
} while (0)

#define _MM256_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
do {\
__m256 t0, t1, t2, t3, t4, t5, t6, t7;\
__m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;\
\
t0 = _mm256_unpacklo_ps(row0, row1);\
t1 = _mm256_unpackhi_ps(row0, row1);\
t2 = _mm256_unpacklo_ps(row2, row3);\
t3 = _mm256_unpackhi_ps(row2, row3);\
t4 = _mm256_unpacklo_ps(row4, row5);\
t5 = _mm256_unpackhi_ps(row4, row5);\
t6 = _mm256_unpacklo_ps(row6, row7);\
t7 = _mm256_unpackhi_ps(row6, row7);\
\
tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));\
tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));\
tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));\
tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));\
tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));\
tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));\
tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));\
tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));\
\
row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);\
row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);\
row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);\
row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);\
row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);\
row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);\
row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);\
row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);\
} while (0)

static inline __m256 combine(__m128 a, __m128 b) {
	return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
}

static inline __m256 combine(__m128 a) {
	return _mm256_insertf128_ps(_mm256_castps128_ps256(a), a, 1);
}

#ifdef _WIN32
inline unsigned ctz(unsigned x) {
	DWORD r = 0;
	_BitScanForward(&r, x);
	return r;
}
#else
inline unsigned ctz(unsigned x) {
	return __builtin_ctz(x);
}
#endif

#endif
