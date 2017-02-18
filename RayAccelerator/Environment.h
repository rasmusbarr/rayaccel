//
//  Environment.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-03-13.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_Environment_h
#define RayAccelerator_Environment_h

#include "RayAccelerator.h"
#include <math.h>

namespace racc {
	struct RACC_ALIGNED(64) Environment {
		float RACC_ALIGNED(16) dimensions[4];
		int32_t RACC_ALIGNED(16) bounds[4];
		uint32_t width;
		uint32_t height;
		cl_mem gpuImage;
		float RACC_ALIGNED(64) pixels[16];
	};
}

namespace racc_internal {
	inline __m128 sample(racc::Environment* environment, __m128 d) {
		__m128 dimensions = _mm_load_ps(environment->dimensions);
		__m128i bounds = _mm_load_si128(reinterpret_cast<const __m128i*>(environment->bounds));
		
		__m128i integerOffset = _mm_setr_epi32(0, 0, 1, 1);
		
		__m128 dzy = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1,2,1,2));
		__m128 dzy2 = _mm_mul_ps(dzy, dzy);
		
		dzy2 = _mm_add_ss(dzy2, _mm_shuffle_ps(dzy2, dzy2, _MM_SHUFFLE(0,1,0,1)));
		
		float len = _mm_cvtss_f32(_mm_rsqrt_ss(dzy2));
		
		float r = acosf(-_mm_cvtss_f32(d))*((1.0f/(2.0f*3.14159265f))*len);
		
		if (!isfinite(r))
			r = 0.0f;
		
		__m128 rm = _mm_set_ss(r);
		__m128 uv = _mm_sub_ps(_mm_set1_ps(0.5f), _mm_mul_ps(dzy, _mm_shuffle_ps(rm, rm, _MM_SHUFFLE(0,0,0,0))));
		uv = _mm_mul_ps(uv, dimensions);
		uv = _mm_sub_ps(uv, _mm_set1_ps(0.5f));
		
		__m128 uvf = _mm_floor_ps(uv);
		__m128 uvt = _mm_sub_ps(uv, uvf);
		
		__m128i xy = _mm_cvttps_epi32(uv);
		xy = _mm_add_epi32(xy, integerOffset);
		
		xy = _mm_max_epi32(_mm_min_epi32(bounds, xy), _mm_setzero_si128());
		
		RACC_ALIGNED(16) uint32_t xyArray[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(xyArray), xy);
		
		unsigned width = environment->width;
		const float* pixels = environment->pixels;
		
		__m128 t00 = _mm_load_ps(pixels + (xyArray[1]*width + xyArray[0])*4);
		__m128 t10 = _mm_load_ps(pixels + (xyArray[1]*width + xyArray[2])*4);
		__m128 t01 = _mm_load_ps(pixels + (xyArray[3]*width + xyArray[0])*4);
		__m128 t11 = _mm_load_ps(pixels + (xyArray[3]*width + xyArray[2])*4);
		
		__m128 uvtn = _mm_sub_ps(_mm_set1_ps(1.0f), uvt);
		
		__m128 utn = _mm_shuffle_ps(uvtn, uvtn, _MM_SHUFFLE(0,0,0,0));
		__m128 ut = _mm_shuffle_ps(uvt, uvt, _MM_SHUFFLE(0,0,0,0));
		
		__m128 t0 = _mm_add_ps(_mm_mul_ps(t00, utn),
							   _mm_mul_ps(t10, ut));
		
		__m128 t1 = _mm_add_ps(_mm_mul_ps(t01, utn),
							   _mm_mul_ps(t11, ut));
		
		return _mm_add_ps(_mm_mul_ps(t0, _mm_shuffle_ps(uvtn, uvtn, _MM_SHUFFLE(1,1,1,1))),
						  _mm_mul_ps(t1, _mm_shuffle_ps(uvt, uvt, _MM_SHUFFLE(1,1,1,1))));
	}
}

#endif
