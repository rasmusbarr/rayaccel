//
//  Materials.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2014-03-15.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Materials.h"

static inline __m256 sinApprox(__m256 x) {
	__m256 y = _mm256_fmadd_ps(_mm256_set1_ps(-16.0f), x, _mm256_set1_ps(8.0f));
	__m256 gt = _mm256_cmp_ps(x, _mm256_set1_ps(0.5), _CMP_GE_OS);
	__m256 xy = _mm256_mul_ps(x, y);
	
	__m256 sign = _mm256_and_ps(gt, _mm256_set1_ps(-0.0f));
	__m256 z = _mm256_and_ps(y, gt);
	
	xy = _mm256_xor_ps(xy, sign);
	
	return _mm256_add_ps(xy, z);
}

static inline __m256 cosApprox(__m256 x) {
	__m256 y = _mm256_sub_ps(x, _mm256_set1_ps(0.75f));
	x = _mm256_blendv_ps(y, _mm256_add_ps(x, _mm256_set1_ps(0.25f)), y);
	return sinApprox(x);
}

Material::~Material() {}

ReflectiveDiffuseMaterial::ReflectiveDiffuseMaterial(float3 k, float eta) {
	ke[0] = k.x;
	ke[1] = k.y;
	ke[2] = k.z;
	ke[3] = eta;
}

void ReflectiveDiffuseMaterial::sample8(const float3_8* rnd, const float3_8* normal, const float3_8* uvt, const float3_8* wo, float3_8* wi, float3_8* color, unsigned* transmitted) {
	__m256 woX = _mm256_load_ps(wo->x.x);
	__m256 woY = _mm256_load_ps(wo->y.x);
	__m256 woZ = _mm256_load_ps(wo->z.x);
	
	__m256 normalX = _mm256_load_ps(normal->x.x);
	__m256 normalY = _mm256_load_ps(normal->y.x);
	__m256 normalZ = _mm256_load_ps(normal->z.x);
	
	__m256 randX = _mm256_load_ps(rnd->x.x);
	__m256 randY = _mm256_load_ps(rnd->y.x);
	__m256 randZ = _mm256_load_ps(rnd->z.x);
	
	__m256 ke = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(this->ke));
	
	*transmitted = 0;
	
	// Calculate reflection vector and fresnel term.
	__m256 cosi = _mm256_fmadd_ps(normalZ, woZ, _mm256_fmadd_ps(normalY, woY, _mm256_mul_ps(normalX, woX)));
	cosi = _mm256_max_ps(cosi, _mm256_setzero_ps());
	
	__m256 reflectionDirX = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), cosi), normalX, woX);
	__m256 reflectionDirY = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), cosi), normalY, woY);
	__m256 reflectionDirZ = _mm256_fmsub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), cosi), normalZ, woZ);
	
	__m256 eta = _mm256_shuffle_ps(ke, ke, _MM_SHUFFLE(3,3,3,3));
	
	__m256 one = _mm256_set1_ps(1.0f);
	__m256 cosi2_minus_one = _mm256_fmsub_ps(cosi, cosi, one);
	__m256 eta2 = _mm256_mul_ps(eta, eta);
	__m256 k = _mm256_fmadd_ps(eta2, cosi2_minus_one, one);
	__m256 cost = _mm256_sqrt_ps(k);

	__m256 signMask = _mm256_set1_ps(-0.0f);
	
	__m256 Rper = _mm256_mul_ps(_mm256_fmsub_ps(eta, cosi, cost), _mm256_rcp_ps(_mm256_fmadd_ps(eta, cosi, cost)));
	__m256 Rpar = _mm256_xor_ps(signMask, _mm256_mul_ps(_mm256_fmsub_ps(eta, cost, cosi), _mm256_rcp_ps(_mm256_fmadd_ps(eta, cost, cosi))));
	
	__m256 fresnel = _mm256_fmadd_ps(Rpar, Rpar, _mm256_mul_ps(Rper, Rper));
	fresnel = _mm256_mul_ps(_mm256_set1_ps(0.5f), fresnel);
	fresnel = _mm256_blendv_ps(fresnel, one, k);
	
	// Calculate diffuse vector and color.
	__m256 baseMask = _mm256_andnot_ps(signMask, normalX);
	baseMask = _mm256_cmp_ps(baseMask, _mm256_set1_ps(0.1f), _CMP_NLE_US);
	
	__m256 negNormalZ = _mm256_xor_ps(signMask, normalZ);
	__m256 baseUx = _mm256_blendv_ps(_mm256_setzero_ps(), negNormalZ, baseMask);
	__m256 baseUy = _mm256_blendv_ps(negNormalZ, _mm256_setzero_ps(), baseMask);
	__m256 baseUz = _mm256_blendv_ps(normalY, normalX, baseMask);
	
	__m256 fb = _mm256_rsqrt_ps(_mm256_fmadd_ps(baseUz, baseUz, _mm256_fmadd_ps(baseUy, baseUy, _mm256_mul_ps(baseUx, baseUx))));
	
	baseUx = _mm256_mul_ps(baseUx, fb);
	baseUy = _mm256_mul_ps(baseUy, fb);
	baseUz = _mm256_mul_ps(baseUz, fb);
	
	__m256 baseVx = _mm256_fmsub_ps(normalY, baseUz, _mm256_mul_ps(normalZ, baseUy));
	__m256 baseVy = _mm256_fmsub_ps(normalZ, baseUx, _mm256_mul_ps(normalX, baseUz));
	__m256 baseVz = _mm256_fmsub_ps(normalX, baseUy, _mm256_mul_ps(normalY, baseUx));
	
	__m256 sinX = sinApprox(randX);
	__m256 cosX = cosApprox(randX);
	
	__m256 r2 = randY;
	__m256 r2s = _mm256_sqrt_ps(r2);
	
	__m256 sq_one_minus_r2 = _mm256_sqrt_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), r2));
	
	__m256 diffuseDirX = _mm256_fmadd_ps(normalX, sq_one_minus_r2, _mm256_mul_ps(_mm256_fmadd_ps(baseUx, cosX, _mm256_mul_ps(baseVx, sinX)), r2s));
	__m256 diffuseDirY = _mm256_fmadd_ps(normalY, sq_one_minus_r2, _mm256_mul_ps(_mm256_fmadd_ps(baseUy, cosX, _mm256_mul_ps(baseVy, sinX)), r2s));
	__m256 diffuseDirZ = _mm256_fmadd_ps(normalZ, sq_one_minus_r2, _mm256_mul_ps(_mm256_fmadd_ps(baseUz, cosX, _mm256_mul_ps(baseVz, sinX)), r2s));
	
	__m256 fd = _mm256_rsqrt_ps(_mm256_fmadd_ps(diffuseDirZ, diffuseDirZ, _mm256_fmadd_ps(diffuseDirY, diffuseDirY, _mm256_mul_ps(diffuseDirX, diffuseDirX))));
	
	diffuseDirX = _mm256_mul_ps(diffuseDirX, fd);
	diffuseDirY = _mm256_mul_ps(diffuseDirY, fd);
	diffuseDirZ = _mm256_mul_ps(diffuseDirZ, fd);
	
	__m256 r = _mm256_shuffle_ps(ke, ke, _MM_SHUFFLE(0,0,0,0));
	__m256 g = _mm256_shuffle_ps(ke, ke, _MM_SHUFFLE(1,1,1,1));
	__m256 b = _mm256_shuffle_ps(ke, ke, _MM_SHUFFLE(2,2,2,2));
	
	// Choose reflection or diffuse.
	__m256 s0 = _mm256_mul_ps(fresnel, _mm256_set1_ps(3.0f));
	__m256 s1 = _mm256_add_ps(b, _mm256_add_ps(r, g));
	__m256 sum = _mm256_add_ps(s0, s1);
	
	__m256 uniform = _mm256_mul_ps(randZ, sum);
	__m256 mask = _mm256_cmp_ps(uniform, s0, _CMP_GE_OQ);
	
	__m256 dirX = _mm256_blendv_ps(reflectionDirX, diffuseDirX, mask);
	__m256 dirY = _mm256_blendv_ps(reflectionDirY, diffuseDirY, mask);
	__m256 dirZ = _mm256_blendv_ps(reflectionDirZ, diffuseDirZ, mask);
	
	r = _mm256_blendv_ps(fresnel, r, mask);
	g = _mm256_blendv_ps(fresnel, g, mask);
	b = _mm256_blendv_ps(fresnel, b, mask);
	
	__m256 scale = _mm256_mul_ps(sum, _mm256_rcp_ps(_mm256_add_ps(b, _mm256_add_ps(r, g))));
	
	r = _mm256_mul_ps(r, scale);
	g = _mm256_mul_ps(g, scale);
	b = _mm256_mul_ps(b, scale);
	
	_mm256_store_ps(wi->x.x, dirX);
	_mm256_store_ps(wi->y.x, dirY);
	_mm256_store_ps(wi->z.x, dirZ);
	
	_mm256_store_ps(color->x.x, r);
	_mm256_store_ps(color->y.x, g);
	_mm256_store_ps(color->z.x, b);
}
