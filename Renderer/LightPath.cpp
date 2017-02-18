//
//  LightPath.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2014-10-08.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "LightPath.h"

void generateTileLightPaths(LightPath* lightPaths, unsigned viewportStride, unsigned tileX, unsigned tileY, unsigned tileSize) {
	for (unsigned y = 0; y < tileSize; ++y) {
		unsigned pixelIndex = (y+tileY)*viewportStride + tileX;
		__m256i pi = _mm256_add_epi32(_mm256_set1_epi32(pixelIndex), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
		
		for (unsigned x = 0; x < tileSize; x += 8) {
			__m256 l0, l1, l2, l3;
			__m256 weight = _mm256_set1_ps(1.0f);
			
			l1 = _mm256_unpacklo_ps(weight, _mm256_castsi256_ps(pi));
			l3 = _mm256_unpackhi_ps(weight, _mm256_castsi256_ps(pi));
			
			l0 = _mm256_shuffle_ps(weight, l1, _MM_SHUFFLE(1,0,1,0));
			l1 = _mm256_shuffle_ps(weight, l1, _MM_SHUFFLE(3,2,3,2));
			l2 = _mm256_shuffle_ps(weight, l3, _MM_SHUFFLE(1,0,1,0));
			l3 = _mm256_shuffle_ps(weight, l3, _MM_SHUFFLE(3,2,3,2));
			
			_mm256_storeu_ps(reinterpret_cast<float*>(lightPaths) + 0, l0);
			_mm256_storeu_ps(reinterpret_cast<float*>(lightPaths) + 8, l1);
			_mm256_storeu_ps(reinterpret_cast<float*>(lightPaths) + 16, l2);
			_mm256_storeu_ps(reinterpret_cast<float*>(lightPaths) + 24, l3);
			
			lightPaths += 8;
			
			pi = _mm256_add_epi32(pi, _mm256_set1_epi32(8));
			pixelIndex += 8;
		}
	}
}
