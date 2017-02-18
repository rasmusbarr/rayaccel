//
//  PathTracingRenderer.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2015-09-16.
//  Copyright (c) 2015 Rasmus Barringer. All rights reserved.
//

#include "PathTracingRenderer.h"
#include "SimdRandom.h"
#include "Materials.h"
#include "LightPath.h"
#include "SceneData.h"
#include "Camera.h"

static inline uint32_t* radixSortHigh16(uint32_t* keys, uint32_t* temp, unsigned count) {
	unsigned lowBuckets[257] = {};
	unsigned highBuckets[257] = {};
	
	uint32_t* lowHistogram = lowBuckets+1;
	uint32_t* highHistogram = highBuckets+1;
	
	for (unsigned i = 0; i < count; ++i) {
		unsigned k = keys[i];
		++lowHistogram[(k >> (2 << 3)) & 0xff];
		++highHistogram[(k >> (3 << 3)) & 0xff];
	}
	
	unsigned previousLow = lowBuckets[0];
	unsigned previousHigh = highBuckets[0];
	
	for (unsigned i = 1; i < 256; ++i) {
		previousLow = (lowBuckets[i] += previousLow);
		previousHigh = (highBuckets[i] += previousHigh);
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t k = keys[i];
		temp[lowBuckets[(k >> (2 << 3)) & 0xff]++] = k;
	}
	
	if (highHistogram[0] == count)
		return temp;
	
	for (unsigned i = 0; i < count; ++i) {
		uint32_t k = temp[i];
		keys[highBuckets[(k >> (3 << 3)) & 0xff]++] = k;
	}
	
	return keys;
}

PathTracingRenderer::PathTracingRenderer(racc::Context* context, Camera& camera, SceneData& scene) : TiledRenderer(context, scene.viewportWidth, scene.viewportHeight), camera(camera), scene(scene) {
	racc::ContextInfo info = racc::info(context);
	rayStreamStride = (info.rayStreamSize + 15) & ~15;
	
	unsigned rayStreamCount = info.rayStreamCount;
	
	payloads = static_cast<LightPath*>(_mm_malloc(sizeof(LightPath)*rayStreamCount*rayStreamStride, 64));
}

void PathTracingRenderer::spawnPrimary(unsigned thread, unsigned tileX, unsigned tileY, unsigned viewportWidth, unsigned viewportHeight, racc::RayStream* output) {
	racc::Ray* rays = static_cast<racc::Ray*>(output->rays) + output->count;
	LightPath* lightPaths = this->payloads + output->index*rayStreamStride + output->count;
	
	generateTileRays(rays, camera, tileX, tileY, tileSize);
	generateTileLightPaths(lightPaths, viewportWidth, tileX, tileY, tileSize);
	
	output->count += tileSize*tileSize;
}

void PathTracingRenderer::shade(unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output) {
	Arena arena = threadArenas[thread];
	
	float4* frameBuffer = this->frameBuffer;
	
	const LightPath* lightPaths = this->payloads + input->index*rayStreamStride + start;
	const racc::Ray* rays = input->rays + start;
	const racc::Result* hitData = input->results + start;
	
	unsigned rayCount = end - start;
	
	LightPath* outputLightPaths = this->payloads + output->index*rayStreamStride;
	
	const uint32_t* indices = scene.indices;
	
	const float4* normals = scene.normals;
	const float2* textureCoords = scene.texcoords;
	
	const uint16_t* perTriangleMaterial = scene.triangleMaterials;
	const float4* perTriangleNormal = scene.triangleNormals;
	
	Material** allMaterials = scene.materials;
	
	racc::Ray* outputRays = output->rays;
	
	unsigned triangleCount = scene.triangleCount;
	unsigned writtenRays = output->count;
	
	unsigned maxDepth = scene.maxDepth;
	
	SimdRandom8 rng(rand());
	
	uint32_t* active = allocateArray<uint32_t>(arena, rayCount + 16);
	uint32_t* contributing = allocateArray<uint32_t>(arena, rayCount + 16);
	
	unsigned activeCount = 0;
	unsigned contributingCount = 0;
	
	for (unsigned i = 0; i < rayCount; ++i) {
		__m128 hit = _mm_load_ps(reinterpret_cast<const float*>(hitData + i));
		
		int triangleIndex = _mm_cvtsi128_si32(_mm_castps_si128(hit));
		unsigned pixel = lightPaths[i].pixel;
		unsigned depth = pixel >> 24;
		
		contributing[contributingCount] = i;
		contributingCount += triangleIndex == -1;
		
		if ((unsigned)triangleIndex < triangleCount && depth < maxDepth)
			active[activeCount++] = i | (perTriangleMaterial[triangleIndex] << 16);
	}
	
	active = radixSortHigh16(active, allocateArray<uint32_t>(arena, activeCount), activeCount);
	
	if (activeCount) {
		unsigned last = activeCount-1;
		
		for (unsigned i = 0; i < 8; ++i)
			active[activeCount + i] = active[last];
	}
	
	for (unsigned n = 0; n < activeCount; ) {
		unsigned i0 = active[n+0] & 0xffff;
		unsigned i1 = active[n+1] & 0xffff;
		unsigned i2 = active[n+2] & 0xffff;
		unsigned i3 = active[n+3] & 0xffff;
		unsigned i4 = active[n+4] & 0xffff;
		unsigned i5 = active[n+5] & 0xffff;
		unsigned i6 = active[n+6] & 0xffff;
		unsigned i7 = active[n+7] & 0xffff;
		
		// Load hit data.
		__m256 triangleIndex = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i0)));
		__m256 t = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i1)));
		__m256 u = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i2)));
		__m256 v = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i3)));
		
		triangleIndex = _mm256_insertf128_ps(triangleIndex, _mm_load_ps(reinterpret_cast<const float*>(hitData + i4)), 1);
		t = _mm256_insertf128_ps(t, _mm_load_ps(reinterpret_cast<const float*>(hitData + i5)), 1);
		u = _mm256_insertf128_ps(u, _mm_load_ps(reinterpret_cast<const float*>(hitData + i6)), 1);
		v = _mm256_insertf128_ps(v, _mm_load_ps(reinterpret_cast<const float*>(hitData + i7)), 1);
		
		_MM256_TRANSPOSE4_PS(triangleIndex, t, u, v);
		
		// Load and interpolate texture coordinates and normal.
		ALIGNED(32) unsigned triangleIndices[8];
		_mm256_store_ps(reinterpret_cast<float*>(triangleIndices), triangleIndex);
		
		const uint32_t* indices0 = indices + triangleIndices[0]*3;
		const uint32_t* indices1 = indices + triangleIndices[1]*3;
		const uint32_t* indices2 = indices + triangleIndices[2]*3;
		const uint32_t* indices3 = indices + triangleIndices[3]*3;
		
		__m128 tu01v01_0 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices0[0]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices1[0]]))));
		__m128 tu23v23_0 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices2[0]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices3[0]]))));
		
		__m128 tu01v01_1 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices0[1]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices1[1]]))));
		__m128 tu23v23_1 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices2[1]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices3[1]]))));
		
		__m128 tu01v01_2 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices0[2]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices1[2]]))));
		__m128 tu23v23_2 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices2[2]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices3[2]]))));
		
		__m256 tu0 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu01v01_0), _mm_castps_pd(tu23v23_0))));
		__m256 tv0 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu01v01_0), _mm_castps_pd(tu23v23_0))));
		
		__m256 tu1 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu01v01_1), _mm_castps_pd(tu23v23_1))));
		__m256 tv1 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu01v01_1), _mm_castps_pd(tu23v23_1))));
		
		__m256 tu2 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu01v01_2), _mm_castps_pd(tu23v23_2))));
		__m256 tv2 = _mm256_castps128_ps256(_mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu01v01_2), _mm_castps_pd(tu23v23_2))));
		
		const uint32_t* indices4 = indices + triangleIndices[4]*3;
		const uint32_t* indices5 = indices + triangleIndices[5]*3;
		const uint32_t* indices6 = indices + triangleIndices[6]*3;
		const uint32_t* indices7 = indices + triangleIndices[7]*3;
		
		__m128 tu45v45_0 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices4[0]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices5[0]]))));
		__m128 tu67v67_0 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices6[0]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices7[0]]))));
		
		__m128 tu45v45_1 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices4[1]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices5[1]]))));
		__m128 tu67v67_1 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices6[1]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices7[1]]))));
		
		__m128 tu45v45_2 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices4[2]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices5[2]]))));
		__m128 tu67v67_2 = _mm_unpacklo_ps(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices6[2]]))),
										   _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(&textureCoords[indices7[2]]))));
		
		tu0 = _mm256_insertf128_ps(tu0, _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu45v45_0), _mm_castps_pd(tu67v67_0))), 1);
		tv0 = _mm256_insertf128_ps(tv0, _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu45v45_0), _mm_castps_pd(tu67v67_0))), 1);
		
		tu1 = _mm256_insertf128_ps(tu1, _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu45v45_1), _mm_castps_pd(tu67v67_1))), 1);
		tv1 = _mm256_insertf128_ps(tv1, _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu45v45_1), _mm_castps_pd(tu67v67_1))), 1);
		
		tu2 = _mm256_insertf128_ps(tu2, _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(tu45v45_2), _mm_castps_pd(tu67v67_2))), 1);
		tv2 = _mm256_insertf128_ps(tv2, _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(tu45v45_2), _mm_castps_pd(tu67v67_2))), 1);
		
		__m256 w = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(u, v));
		
		__m256 tu = _mm256_mul_ps(tu0, w);
		__m256 tv = _mm256_mul_ps(tv0, w);
		
		tu = _mm256_fmadd_ps(tu1, u, tu);
		tv = _mm256_fmadd_ps(tv1, u, tv);
		
		tu = _mm256_fmadd_ps(tu2, v, tu);
		tv = _mm256_fmadd_ps(tv2, v, tv);
		
		__m256 normal04_0 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices0[0]].x));
		__m256 normal04_1 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices0[1]].x));
		__m256 normal04_2 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices0[2]].x));
		
		normal04_0 = _mm256_insertf128_ps(normal04_0, _mm_load_ps(&normals[indices4[0]].x), 1);
		normal04_1 = _mm256_insertf128_ps(normal04_1, _mm_load_ps(&normals[indices4[1]].x), 1);
		normal04_2 = _mm256_insertf128_ps(normal04_2, _mm_load_ps(&normals[indices4[2]].x), 1);
		
		__m256 normal15_0 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices1[0]].x));
		__m256 normal15_1 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices1[1]].x));
		__m256 normal15_2 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices1[2]].x));
		
		normal15_0 = _mm256_insertf128_ps(normal15_0, _mm_load_ps(&normals[indices5[0]].x), 1);
		normal15_1 = _mm256_insertf128_ps(normal15_1, _mm_load_ps(&normals[indices5[1]].x), 1);
		normal15_2 = _mm256_insertf128_ps(normal15_2, _mm_load_ps(&normals[indices5[2]].x), 1);
		
		__m256 normal26_0 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices2[0]].x));
		__m256 normal26_1 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices2[1]].x));
		__m256 normal26_2 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices2[2]].x));
		
		normal26_0 = _mm256_insertf128_ps(normal26_0, _mm_load_ps(&normals[indices6[0]].x), 1);
		normal26_1 = _mm256_insertf128_ps(normal26_1, _mm_load_ps(&normals[indices6[1]].x), 1);
		normal26_2 = _mm256_insertf128_ps(normal26_2, _mm_load_ps(&normals[indices6[2]].x), 1);
		
		__m256 normal37_0 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices3[0]].x));
		__m256 normal37_1 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices3[1]].x));
		__m256 normal37_2 = _mm256_castps128_ps256(_mm_load_ps(&normals[indices3[2]].x));
		
		normal37_0 = _mm256_insertf128_ps(normal37_0, _mm_load_ps(&normals[indices7[0]].x), 1);
		normal37_1 = _mm256_insertf128_ps(normal37_1, _mm_load_ps(&normals[indices7[1]].x), 1);
		normal37_2 = _mm256_insertf128_ps(normal37_2, _mm_load_ps(&normals[indices7[2]].x), 1);
		
		__m256 normal04 = _mm256_mul_ps(normal04_0, _mm256_shuffle_ps(w, w, _MM_SHUFFLE(0,0,0,0)));
		__m256 normal15 = _mm256_mul_ps(normal15_0, _mm256_shuffle_ps(w, w, _MM_SHUFFLE(1,1,1,1)));
		__m256 normal26 = _mm256_mul_ps(normal26_0, _mm256_shuffle_ps(w, w, _MM_SHUFFLE(2,2,2,2)));
		__m256 normal37 = _mm256_mul_ps(normal37_0, _mm256_shuffle_ps(w, w, _MM_SHUFFLE(3,3,3,3)));
		
		normal04 = _mm256_fmadd_ps(normal04_1, _mm256_shuffle_ps(u, u, _MM_SHUFFLE(0,0,0,0)), normal04);
		normal15 = _mm256_fmadd_ps(normal15_1, _mm256_shuffle_ps(u, u, _MM_SHUFFLE(1,1,1,1)), normal15);
		normal26 = _mm256_fmadd_ps(normal26_1, _mm256_shuffle_ps(u, u, _MM_SHUFFLE(2,2,2,2)), normal26);
		normal37 = _mm256_fmadd_ps(normal37_1, _mm256_shuffle_ps(u, u, _MM_SHUFFLE(3,3,3,3)), normal37);
		
		normal04 = _mm256_fmadd_ps(normal04_2, _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0)), normal04);
		normal15 = _mm256_fmadd_ps(normal15_2, _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1)), normal15);
		normal26 = _mm256_fmadd_ps(normal26_2, _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2)), normal26);
		normal37 = _mm256_fmadd_ps(normal37_2, _mm256_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3)), normal37);
		
		_MM256_TRANSPOSE4_PS(normal04, normal15, normal26, normal37);
		
		__m256 normalX = normal04;
		__m256 normalY = normal15;
		__m256 normalZ = normal26;
		
		__m256 fn = _mm256_rsqrt_ps(_mm256_fmadd_ps(normalZ, normalZ, _mm256_fmadd_ps(normalY, normalY, _mm256_mul_ps(normalX, normalX))));
		
		normalX = _mm256_mul_ps(normalX, fn);
		normalY = _mm256_mul_ps(normalY, fn);
		normalZ = _mm256_mul_ps(normalZ, fn);
		
		// Load geometry normal.
		__m256 geometryNormal04 = _mm256_castps128_ps256(_mm_load_ps(&perTriangleNormal[triangleIndices[0]].x));
		__m256 geometryNormal15 = _mm256_castps128_ps256(_mm_load_ps(&perTriangleNormal[triangleIndices[1]].x));
		__m256 geometryNormal26 = _mm256_castps128_ps256(_mm_load_ps(&perTriangleNormal[triangleIndices[2]].x));
		__m256 geometryNormal37 = _mm256_castps128_ps256(_mm_load_ps(&perTriangleNormal[triangleIndices[3]].x));
		
		geometryNormal04 = _mm256_insertf128_ps(geometryNormal04, _mm_load_ps(&perTriangleNormal[triangleIndices[4]].x), 1);
		geometryNormal15 = _mm256_insertf128_ps(geometryNormal15, _mm_load_ps(&perTriangleNormal[triangleIndices[5]].x), 1);
		geometryNormal26 = _mm256_insertf128_ps(geometryNormal26, _mm_load_ps(&perTriangleNormal[triangleIndices[6]].x), 1);
		geometryNormal37 = _mm256_insertf128_ps(geometryNormal37, _mm_load_ps(&perTriangleNormal[triangleIndices[7]].x), 1);
		
		_MM256_TRANSPOSE4_PS(geometryNormal04, geometryNormal15, geometryNormal26, geometryNormal37);
		
		__m256 geometryNormalX = geometryNormal04;
		__m256 geometryNormalY = geometryNormal15;
		__m256 geometryNormalZ = geometryNormal26;
		
		// Load ray data.
		__m256 rayOx = _mm256_load_ps(rays[i0].origin);
		__m256 rayOy = _mm256_load_ps(rays[i1].origin);
		__m256 rayOz = _mm256_load_ps(rays[i2].origin);
		__m256 minT = _mm256_load_ps(rays[i3].origin);
		__m256 rayDx = _mm256_load_ps(rays[i4].origin);
		__m256 rayDy = _mm256_load_ps(rays[i5].origin);
		__m256 rayDz = _mm256_load_ps(rays[i6].origin);
		__m256 maxT = _mm256_load_ps(rays[i7].origin);
		
		_MM256_TRANSPOSE8_PS(rayOx, rayOy, rayOz, minT, rayDx, rayDy, rayDz, maxT);
		
		// Load light path data.
		__m256 weightR = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i0].weight));
		__m256 weightG = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i1].weight));
		__m256 weightB = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i2].weight));
		__m256 pixel = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i3].weight));
		
		weightR = _mm256_insertf128_ps(weightR, _mm_load_ps(lightPaths[i4].weight), 1);
		weightG = _mm256_insertf128_ps(weightG, _mm_load_ps(lightPaths[i5].weight), 1);
		weightB = _mm256_insertf128_ps(weightB, _mm_load_ps(lightPaths[i6].weight), 1);
		pixel = _mm256_insertf128_ps(pixel, _mm_load_ps(lightPaths[i7].weight), 1);
		
		_MM256_TRANSPOSE4_PS(weightR, weightG, weightB, pixel);
		
		// Load materials.
		__m256i materials = _mm256_loadu_si256(reinterpret_cast<__m256i*>(active + n));
		materials = _mm256_srli_epi32(materials, 16);
		
		__m256 first = _mm256_castsi256_ps(_mm256_permute2f128_si256(materials, materials, (0) | ((2) << 4)));
		first = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(first), _MM_SHUFFLE(0,0,0,0)));
		
		unsigned matches = ctz(~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(materials, _mm256_castps_si256(first)))));
		
		Material* material = allMaterials[_mm_cvtsi128_si32(_mm256_castsi256_si128(materials))];
		
		// Shade.
		__m256 rdDotGn = _mm256_mul_ps(rayDx, geometryNormalX);
		rdDotGn = _mm256_fmadd_ps(rayDy, geometryNormalY, rdDotGn);
		rdDotGn = _mm256_fmadd_ps(rayDz, geometryNormalZ, rdDotGn);
		__m256 sgn0 = _mm256_and_ps(rdDotGn, _mm256_set1_ps(-0.0f));
		
		normalX = _mm256_xor_ps(sgn0, normalX);
		normalY = _mm256_xor_ps(sgn0, normalY);
		normalZ = _mm256_xor_ps(sgn0, normalZ);
		
		__m256 randX = rng.nextFloat();
		__m256 randY = rng.nextFloat();
		__m256 randZ = rng.nextFloat();
		
		__m256 posX = _mm256_fmadd_ps(rayDx, t, rayOx);
		__m256 posY = _mm256_fmadd_ps(rayDy, t, rayOy);
		__m256 posZ = _mm256_fmadd_ps(rayDz, t, rayOz);
		
		float3_8 rnd, normal, uvt, wo, wi, color;
		unsigned transmitted = 0;

		__m256 signMask = _mm256_set1_ps(-0.0f);
		
		_mm256_store_ps(rnd.x.x, randX);
		_mm256_store_ps(rnd.y.x, randY);
		_mm256_store_ps(rnd.z.x, randZ);
		_mm256_store_ps(normal.x.x, normalX);
		_mm256_store_ps(normal.y.x, normalY);
		_mm256_store_ps(normal.z.x, normalZ);
		_mm256_store_ps(uvt.x.x, tu);
		_mm256_store_ps(uvt.y.x, tv);
		_mm256_store_ps(uvt.z.x, _mm256_xor_ps(t, sgn0));
		_mm256_store_ps(wo.x.x, _mm256_xor_ps(signMask, rayDx));
		_mm256_store_ps(wo.y.x, _mm256_xor_ps(signMask, rayDy));
		_mm256_store_ps(wo.z.x, _mm256_xor_ps(signMask, rayDz));
		_mm256_store_ps(wi.x.x, posX);
		_mm256_store_ps(wi.y.x, posY);
		_mm256_store_ps(wi.z.x, posZ);
		
		material->sample8(&rnd, &normal, &uvt, &wo, &wi, &color, &transmitted);
		
		__m256 colorR = _mm256_load_ps(color.x.x);
		__m256 colorG = _mm256_load_ps(color.y.x);
		__m256 colorB = _mm256_load_ps(color.z.x);
		
		__m256 dirX = _mm256_load_ps(wi.x.x);
		__m256 dirY = _mm256_load_ps(wi.y.x);
		__m256 dirZ = _mm256_load_ps(wi.z.x);
		
		weightR = _mm256_mul_ps(weightR, colorR);
		weightG = _mm256_mul_ps(weightG, colorG);
		weightB = _mm256_mul_ps(weightB, colorB);
		
		unsigned mask0 = _mm256_movemask_ps(_mm256_cmp_ps(weightR, _mm256_set1_ps(0.01f), _CMP_GT_OQ));
		unsigned mask1 = _mm256_movemask_ps(_mm256_cmp_ps(weightG, _mm256_set1_ps(0.01f), _CMP_GT_OQ));
		unsigned mask2 = _mm256_movemask_ps(_mm256_cmp_ps(weightB, _mm256_set1_ps(0.01f), _CMP_GT_OQ));
		
		unsigned mask = mask0 | mask1 | mask2;
		
		__m256 sgn1 = _mm256_fmadd_ps(dirZ, geometryNormalZ, _mm256_fmadd_ps(dirY, geometryNormalY, _mm256_mul_ps(dirX, geometryNormalX)));

		mask &= _mm256_movemask_ps(_mm256_xor_ps(sgn0, sgn1)) ^ transmitted;
		
		sgn1 = _mm256_and_ps(sgn1, _mm256_set1_ps(-0.0f));
		
		geometryNormalX = _mm256_xor_ps(sgn1, geometryNormalX);
		geometryNormalY = _mm256_xor_ps(sgn1, geometryNormalY);
		geometryNormalZ = _mm256_xor_ps(sgn1, geometryNormalZ);
		
		posX = _mm256_fmadd_ps(geometryNormalX, _mm256_set1_ps(1e-4f), posX);
		posY = _mm256_fmadd_ps(geometryNormalY, _mm256_set1_ps(1e-4f), posY);
		posZ = _mm256_fmadd_ps(geometryNormalZ, _mm256_set1_ps(1e-4f), posZ);
		
		pixel = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(pixel), _mm256_set1_epi32(0x1000000)));
		
		// NAN check.
		mask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(posX, posX, _CMP_EQ_OQ), _mm256_cmp_ps(dirX, dirX, _CMP_EQ_OQ)));
		mask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(posY, posY, _CMP_EQ_OQ), _mm256_cmp_ps(dirY, dirY, _CMP_EQ_OQ)));
		mask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(posZ, posZ, _CMP_EQ_OQ), _mm256_cmp_ps(dirZ, dirZ, _CMP_EQ_OQ)));
		
		minT = _mm256_set1_ps(1e-3f);
		maxT = _mm256_set1_ps(1e+6f);
		
		_MM256_TRANSPOSE8_PS(posX, posY, posZ, minT, dirX, dirY, dirZ, maxT);
		_MM256_TRANSPOSE4_PS(weightR, weightG, weightB, pixel);
		
		ALIGNED(32) float rays[8*8];
		ALIGNED(32) float paths[4*8];
		
		_mm256_store_ps(rays + 0, posX);
		_mm256_store_ps(rays + 8, posY);
		_mm256_store_ps(rays + 16, posZ);
		_mm256_store_ps(rays + 24, minT);
		_mm256_store_ps(rays + 32, dirX);
		_mm256_store_ps(rays + 40, dirY);
		_mm256_store_ps(rays + 48, dirZ);
		_mm256_store_ps(rays + 56, maxT);
		
		_mm256_store_ps(paths + 0, weightR);
		_mm256_store_ps(paths + 8, weightG);
		_mm256_store_ps(paths + 16, weightB);
		_mm256_store_ps(paths + 24, pixel);
		
		mask &= (1 << matches)-1;
		
		unsigned bits = mask;
		
		while (bits) {
			unsigned index = ctz(bits);
			bits &= bits-1;
			
			__m256 ray = _mm256_load_ps(rays + 8*index);
			__m128 path = _mm_load_ps(paths + 4*(((index << 1) & 7) + (index >> 2)));
			
			_mm256_store_ps(outputRays[writtenRays].origin, ray);
			_mm_store_ps(outputLightPaths[writtenRays].weight, path);
			
			if (n+index < activeCount)
				++writtenRays;
		}
		
		n += matches;
	}
	
	unsigned n = 0;
	
	for (; n+7 < contributingCount; n += 8) {
		unsigned i0 = contributing[n+0];
		unsigned i1 = contributing[n+1];
		unsigned i2 = contributing[n+2];
		unsigned i3 = contributing[n+3];
		
		__m256 hit04 = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i0)));
		__m256 hit15 = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i1)));
		__m256 hit26 = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i2)));
		__m256 hit37 = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<const float*>(hitData + i3)));
		
		__m256 weight04 = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i0].weight));
		__m256 weight15 = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i1].weight));
		__m256 weight26 = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i2].weight));
		__m256 weight37 = _mm256_castps128_ps256(_mm_load_ps(lightPaths[i3].weight));
		
		unsigned i4 = contributing[n+4];
		unsigned i5 = contributing[n+5];
		unsigned i6 = contributing[n+6];
		unsigned i7 = contributing[n+7];
		
		hit04 = _mm256_insertf128_ps(hit04, _mm_load_ps(reinterpret_cast<const float*>(hitData + i4)), 1);
		hit15 = _mm256_insertf128_ps(hit15, _mm_load_ps(reinterpret_cast<const float*>(hitData + i5)), 1);
		hit26 = _mm256_insertf128_ps(hit26, _mm_load_ps(reinterpret_cast<const float*>(hitData + i6)), 1);
		hit37 = _mm256_insertf128_ps(hit37, _mm_load_ps(reinterpret_cast<const float*>(hitData + i7)), 1);
		
		weight04 = _mm256_insertf128_ps(weight04, _mm_load_ps(lightPaths[i4].weight), 1);
		weight15 = _mm256_insertf128_ps(weight15, _mm_load_ps(lightPaths[i5].weight), 1);
		weight26 = _mm256_insertf128_ps(weight26, _mm_load_ps(lightPaths[i6].weight), 1);
		weight37 = _mm256_insertf128_ps(weight37, _mm_load_ps(lightPaths[i7].weight), 1);
		
		__m256 weightR = weight04;
		__m256 weightG = weight15;
		__m256 weightB = weight26;
		__m256 pixel = weight37;
		
		_MM256_TRANSPOSE4_PS(weightR, weightG, weightB, pixel);
		
		pixel = _mm256_and_ps(pixel, _mm256_castsi256_ps(_mm256_set1_epi32(0xffffff)));
		
		ALIGNED(32) unsigned pixels[8];
		_mm256_store_ps(reinterpret_cast<float*>(pixels), pixel);
		
		float* p0 = &frameBuffer[pixels[0]].x;
		float* p1 = &frameBuffer[pixels[1]].x;
		float* p2 = &frameBuffer[pixels[2]].x;
		float* p3 = &frameBuffer[pixels[3]].x;
		
		__m256 pixelColor04 = _mm256_castps128_ps256(_mm_load_ps(p0));
		__m256 pixelColor15 = _mm256_castps128_ps256(_mm_load_ps(p1));
		__m256 pixelColor26 = _mm256_castps128_ps256(_mm_load_ps(p2));
		__m256 pixelColor37 = _mm256_castps128_ps256(_mm_load_ps(p3));
		
		float* p4 = &frameBuffer[pixels[4]].x;
		float* p5 = &frameBuffer[pixels[5]].x;
		float* p6 = &frameBuffer[pixels[6]].x;
		float* p7 = &frameBuffer[pixels[7]].x;
		
		pixelColor04 = _mm256_insertf128_ps(pixelColor04, _mm_load_ps(p4), 1);
		pixelColor15 = _mm256_insertf128_ps(pixelColor15, _mm_load_ps(p5), 1);
		pixelColor26 = _mm256_insertf128_ps(pixelColor26, _mm_load_ps(p6), 1);
		pixelColor37 = _mm256_insertf128_ps(pixelColor37, _mm_load_ps(p7), 1);
		
		__m256 environment04 = _mm256_shuffle_ps(hit04, hit04, _MM_SHUFFLE(1,3,2,1));
		__m256 environment15 = _mm256_shuffle_ps(hit15, hit15, _MM_SHUFFLE(1,3,2,1));
		__m256 environment26 = _mm256_shuffle_ps(hit26, hit26, _MM_SHUFFLE(1,3,2,1));
		__m256 environment37 = _mm256_shuffle_ps(hit37, hit37, _MM_SHUFFLE(1,3,2,1));
		
		pixelColor04 = _mm256_fmadd_ps(environment04, weight04, pixelColor04);
		pixelColor15 = _mm256_fmadd_ps(environment15, weight15, pixelColor15);
		pixelColor26 = _mm256_fmadd_ps(environment26, weight26, pixelColor26);
		pixelColor37 = _mm256_fmadd_ps(environment37, weight37, pixelColor37);
		
		_mm_store_ps(p0, _mm256_castps256_ps128(pixelColor04));
		_mm_store_ps(p1, _mm256_castps256_ps128(pixelColor15));
		_mm_store_ps(p2, _mm256_castps256_ps128(pixelColor26));
		_mm_store_ps(p3, _mm256_castps256_ps128(pixelColor37));
		_mm_store_ps(p4, _mm256_extractf128_ps(pixelColor04, 1));
		_mm_store_ps(p5, _mm256_extractf128_ps(pixelColor15, 1));
		_mm_store_ps(p6, _mm256_extractf128_ps(pixelColor26, 1));
		_mm_store_ps(p7, _mm256_extractf128_ps(pixelColor37, 1));
	}
	
	for (; n < contributingCount; ++n) {
		unsigned i = contributing[n];
		
		__m128 hit = _mm_load_ps(reinterpret_cast<const float*>(hitData + i));
		__m128 weight = _mm_load_ps(lightPaths[i].weight);
		
		unsigned pixel = lightPaths[i].pixel;
		
		float4* pixelColor = &frameBuffer[pixel & 0xffffff];
		
		__m128 environment = _mm_shuffle_ps(hit, hit, _MM_SHUFFLE(1,3,2,1));
		
		_mm_store_ps(&pixelColor->x, _mm_fmadd_ps(environment, weight, _mm_load_ps(&pixelColor->x)));
	}
	
	output->count = writtenRays;
}

PathTracingRenderer::~PathTracingRenderer() {
	_mm_free(payloads);
}
