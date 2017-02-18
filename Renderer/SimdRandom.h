//
//  SimdRandom.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-24.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_SimdRandom_h
#define Renderer_SimdRandom_h

#include "Renderer.h"

class ALIGNED(32) SimdRandom8 {
private:
	__m256i w;
	__m256i z;
	
public:
	SimdRandom8(uint32_t s) {
		ALIGNED(32) uint32_t array[8];
		{
			uint32_t w = s + 1;
			uint32_t z = s*s + s + 2;
			
			for (unsigned i = 0; i < 8; ++i) {
				z = 36969 * (z & 0xffff) + (z >> 16);
				w = 18000 * (w & 0xffff) + (w >> 16);
				array[i] = (z << 16) + w;
			}
		}
		
		__m256i seed = _mm256_load_si256(reinterpret_cast<const __m256i*>(array));
		
		w = _mm256_add_epi32(seed, _mm256_set1_epi32(1));
		z = _mm256_add_epi32(_mm256_mullo_epi32(seed, seed), _mm256_add_epi32(seed, _mm256_set1_epi32(2)));
	}
	
	__m256i nextInt() {
		__m256i mask = _mm256_set1_epi32(0xffff);
		
		z = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(36969), _mm256_and_si256(z, mask)), _mm256_srli_epi32(z, 16));
		w = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(18000), _mm256_and_si256(w, mask)), _mm256_srli_epi32(w, 16));
		
		return _mm256_add_epi32(_mm256_slli_epi32(z, 16), w);
	}
	
	__m256 nextFloat() {
		__m256i mask = _mm256_set1_epi32(0xffff);
		
		__m256i x = _mm256_and_si256(nextInt(), mask);
		__m256 fx = _mm256_cvtepi32_ps(x);
		
		return _mm256_mul_ps(fx, _mm256_set1_ps(1.0f/0xffff));
	}
};

#endif
