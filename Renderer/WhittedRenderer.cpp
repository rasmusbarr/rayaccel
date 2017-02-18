//
//  WhittedRenderer.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2015-09-19.
//  Copyright (c) 2015 Rasmus Barringer. All rights reserved.
//

#include "WhittedRenderer.h"
#include "LightPath.h"
#include "SceneData.h"
#include "Camera.h"

struct ALIGNED(64) LoopData {
	racc::Ray ray;
	LightPath lightPath;
	uint32_t link;
};

static inline void popPush(std::mutex* mutex, uint32_t* freeList, unsigned& freeCount, uint32_t* indices, unsigned popCount, unsigned pushCount) {
	if (pushCount > popCount) {
		unsigned count = pushCount - popCount;
		
		mutex->lock();
		memcpy(indices + popCount, freeList + (freeCount -= count), sizeof(uint32_t)*count);
		mutex->unlock();
	}
	else if (pushCount < popCount) {
		unsigned count = popCount - pushCount;
		
		mutex->lock();
		memcpy(freeList + freeCount, indices + pushCount, sizeof(uint32_t)*count);
		freeCount += count;
		mutex->unlock();
	}
}

WhittedRenderer::WhittedRenderer(racc::Context* context, Camera& camera, SceneData& scene) : TiledRenderer(context, scene.viewportWidth, scene.viewportHeight), camera(camera), scene(scene) {
	racc::ContextInfo info = racc::info(context);
	rayStreamStride = (info.rayStreamSize + 15) & ~15;
	
	unsigned rayStreamCount = info.rayStreamCount;
	
	payloads = static_cast<LightPath*>(_mm_malloc(sizeof(LightPath)*rayStreamCount*rayStreamStride, 64));
	loopHeads = static_cast<uint32_t*>(_mm_malloc(sizeof(uint32_t)*rayStreamCount*rayStreamStride, 64));
	
	unsigned maxShadingDepth = 8;
	maxShadingItems = maxShadingDepth * info.maxRaysInFlight;
	
	loopData = static_cast<LoopData*>(_mm_malloc(sizeof(LoopData)*(maxShadingItems+1), 64));
	freeList = static_cast<uint32_t*>(_mm_malloc(sizeof(uint32_t)*maxShadingItems, 64));
	
	for (unsigned j = 0; j < maxShadingItems; ++j)
		freeList[j] = j+1;
	
	freeCount = maxShadingItems;
	
	loopData[0].link = 0;
}

void WhittedRenderer::endFrame() {
	assert(freeCount == maxShadingItems);
	
	for (unsigned j = 0; j < maxShadingItems; ++j)
		freeList[j] = j+1;
	
	TiledRenderer::endFrame();
}

void WhittedRenderer::spawnPrimary(unsigned thread, unsigned tileX, unsigned tileY, unsigned viewportWidth, unsigned viewportHeight, racc::RayStream* output) {
	racc::Ray* rays = output->rays + output->count;
	LightPath* lightPaths = this->payloads + output->index*rayStreamStride + output->count;
	uint32_t* loopHeads = this->loopHeads + output->index*rayStreamStride + output->count;
	
	generateTileRays(rays, camera, tileX, tileY, tileSize);
	generateTileLightPaths(lightPaths, viewportWidth, tileX, tileY, tileSize);
	memset(loopHeads, 0, tileSize*tileSize*sizeof(uint32_t));
	
	output->count += tileSize*tileSize;
}

static unsigned loopHandling(LoopData* loopData, unsigned writtenRays,
							 racc::Ray* recursiveRays, LightPath* recursiveLightPaths, uint32_t* recursiveIndices, unsigned recursiveRayCount,
							 uint32_t* terminatedHeads, unsigned terminatedCount,
							 LightPath* outputLightPaths, uint32_t* outputRayHeads, racc::Ray* outputRays,
							 std::mutex& mutex, uint32_t* freeList, unsigned& freeCount, Arena arena) {
	uint32_t* indices = allocateArray<uint32_t>(arena, (terminatedCount > recursiveRayCount ? terminatedCount : recursiveRayCount) + 8);
	
	unsigned count = 0;
	
	if (terminatedCount) {
		unsigned lastHead = terminatedHeads[terminatedCount-1];
	
		for (unsigned i = 0; i < 32; ++i)
			terminatedHeads[i + terminatedCount] = lastHead;
		
		for (unsigned i = 0; i < terminatedCount; ++i) {
			unsigned index = terminatedHeads[i];
			unsigned lookahead = terminatedHeads[i + 32];
			
			__m256 d0 = _mm256_load_ps(reinterpret_cast<float*>(loopData + index));
			__m256 d1 = _mm256_load_ps(reinterpret_cast<float*>(loopData + index) + 8);
			
			_mm_prefetch(reinterpret_cast<const char*>(loopData + lookahead), _MM_HINT_NTA);
			
			indices[count] = index;
			
			_mm256_store_ps(outputRays[writtenRays].origin, d0);
			_mm_store_ps(outputLightPaths[writtenRays].weight, _mm256_castps256_ps128(d1));
			_mm_store_ss(reinterpret_cast<float*>(outputRayHeads + writtenRays), _mm256_extractf128_ps(d1, 1));
			
			++writtenRays;
			++count;
		}
	}
	
	popPush(&mutex, freeList, freeCount, indices, count, recursiveRayCount);
	
	for (unsigned i = 0; i < recursiveRayCount; ++i)  {
		unsigned index = indices[i];
		unsigned headIndex = recursiveIndices[i];
		
		__m256 d0 = _mm256_load_ps(reinterpret_cast<float*>(recursiveRays + i));
		__m256 d1 = _mm256_castps128_ps256(_mm_load_ps(reinterpret_cast<float*>(recursiveLightPaths + i)));
		
		unsigned head = outputRayHeads[headIndex];
		d1 = _mm256_insertf128_ps(d1, _mm_castsi128_ps(_mm_cvtsi32_si128(head)), 1);
		outputRayHeads[headIndex] = index;
		
		_mm256_stream_ps(reinterpret_cast<float*>(loopData + index), d0);
		_mm256_stream_ps(reinterpret_cast<float*>(loopData + index) + 8, d1);
	}
	
	return writtenRays;
}

void WhittedRenderer::shade(unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output) {
	Arena arena = threadArenas[thread];
	
	float4* frameBuffer = this->frameBuffer;
	
	const LightPath* lightPaths = this->payloads + input->index*rayStreamStride + start;
	const racc::Ray* rays = input->rays + start;
	const racc::Result* hitData = input->results + start;
	const uint32_t* rayHeads = this->loopHeads + input->index*rayStreamStride + start;
	
	unsigned rayCount = end - start;
	
	LightPath* outputLightPaths = this->payloads + output->index*rayStreamStride;
	uint32_t* outputRayHeads = this->loopHeads + output->index*rayStreamStride;
	
	const uint32_t* indices = scene.indices;
	const float4* normals = scene.normals;
	const float4* perTriangleNormal = scene.triangleNormals;
	
	racc::Ray* outputRays = output->rays;
	
	unsigned triangleCount = scene.triangleCount;
	unsigned writtenRays = output->count;
	
	unsigned maxDepth = scene.maxDepth;
	
	uint32_t* active = static_cast<uint32_t*>(allocate(arena, sizeof(uint32_t)*(rayCount + 16)));
	uint32_t* contributing = static_cast<uint32_t*>(allocate(arena, sizeof(uint32_t)*(rayCount + 16)));
	uint32_t* terminatedHeads = static_cast<uint32_t*>(allocate(arena, sizeof(uint32_t)*(rayCount + 64)));
	
	unsigned activeCount = 0;
	unsigned contributingCount = 0;
	unsigned terminatedCount = 0;
	
	for (unsigned i = 0; i < rayCount; ++i) {
		__m128 hit = _mm_load_ps(reinterpret_cast<const float*>(hitData + i));
		
		int triangleIndex = _mm_cvtsi128_si32(_mm_castps_si128(hit));
		unsigned pixel = lightPaths[i].pixel;
		unsigned depth = pixel >> 24;
		unsigned head = rayHeads[i];
		
		active[activeCount] = i;
		contributing[contributingCount] = i;
		terminatedHeads[terminatedCount] = head;
		
		bool active = (unsigned)triangleIndex < triangleCount && depth < maxDepth;
		activeCount += active;
		terminatedCount += (!active && head);
		contributingCount += (triangleIndex == -1);
	}
	
	if (activeCount) {
		unsigned last = activeCount-1;
		
		for (unsigned i = 0; i < 8; ++i)
			active[activeCount + i] = active[last];
	}
	
	racc::Ray* recursiveRays = allocateArray<racc::Ray>(arena, rayCount + 8);
	LightPath* recursiveLightPaths = allocateArray<LightPath>(arena, rayCount + 8);
	uint32_t* recursiveIndices = allocateArray<uint32_t>(arena, rayCount + 8);
	unsigned recursiveRayCount = 0;
	
	unsigned n = 0;
	
	for (; n < activeCount; n += 8) {
		unsigned i0 = active[n+0];
		unsigned i1 = active[n+1];
		unsigned i2 = active[n+2];
		unsigned i3 = active[n+3];
		unsigned i4 = active[n+4];
		unsigned i5 = active[n+5];
		unsigned i6 = active[n+6];
		unsigned i7 = active[n+7];
		
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
		const uint32_t* indices4 = indices + triangleIndices[4]*3;
		const uint32_t* indices5 = indices + triangleIndices[5]*3;
		const uint32_t* indices6 = indices + triangleIndices[6]*3;
		const uint32_t* indices7 = indices + triangleIndices[7]*3;
		
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
		
		__m256 w = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(u, v));
		
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
		
		// Material.
		__m256 materialR = _mm256_set1_ps(0.3f);
		__m256 materialG = _mm256_set1_ps(0.3f);
		__m256 materialB = _mm256_set1_ps(0.3f);
		
		// Shade.
		__m256 rdDotGn = _mm256_mul_ps(rayDx, geometryNormalX);
		rdDotGn = _mm256_fmadd_ps(rayDy, geometryNormalY, rdDotGn);
		rdDotGn = _mm256_fmadd_ps(rayDz, geometryNormalZ, rdDotGn);
		__m256 sgn0 = _mm256_and_ps(rdDotGn, _mm256_set1_ps(-0.0f));
		
		normalX = _mm256_xor_ps(sgn0, normalX);
		normalY = _mm256_xor_ps(sgn0, normalY);
		normalZ = _mm256_xor_ps(sgn0, normalZ);
		
		__m256 lightX = _mm256_set1_ps(0.57f);
		__m256 lightY = _mm256_set1_ps(0.57f);
		__m256 lightZ = _mm256_set1_ps(0.57f);
		
		__m256 light = _mm256_mul_ps(normalX, lightX);
		light = _mm256_fmadd_ps(normalY, lightY, light);
		light = _mm256_fmadd_ps(normalZ, lightZ, light);
		light = _mm256_max_ps(light, _mm256_setzero_ps());
		
		weightR = _mm256_mul_ps(weightR, materialR);
		weightG = _mm256_mul_ps(weightG, materialG);
		weightB = _mm256_mul_ps(weightB, materialB);
		
		__m256 radianceR = _mm256_mul_ps(weightR, light);
		__m256 radianceG = _mm256_mul_ps(weightG, light);
		__m256 radianceB = _mm256_mul_ps(weightB, light);
		__m256 radianceA = _mm256_setzero_ps();
		
		_MM256_TRANSPOSE4_PS(radianceR, radianceG, radianceB, radianceA);
		
		ALIGNED(32) unsigned pixels[8];
		_mm256_store_ps(reinterpret_cast<float*>(pixels), _mm256_and_ps(pixel, _mm256_castsi256_ps(_mm256_set1_epi32(0xffffff))));
		
		__m256 pixel04 = _mm256_castps128_ps256(_mm_load_ps(&frameBuffer[pixels[0]].x));
		__m256 pixel15 = _mm256_castps128_ps256(_mm_load_ps(&frameBuffer[pixels[1]].x));
		__m256 pixel26 = _mm256_castps128_ps256(_mm_load_ps(&frameBuffer[pixels[2]].x));
		__m256 pixel37 = _mm256_castps128_ps256(_mm_load_ps(&frameBuffer[pixels[3]].x));
		
		pixel04 = _mm256_insertf128_ps(pixel04, _mm_load_ps(&frameBuffer[pixels[4]].x), 1);
		pixel15 = _mm256_insertf128_ps(pixel15, _mm_load_ps(&frameBuffer[pixels[5]].x), 1);
		pixel26 = _mm256_insertf128_ps(pixel26, _mm_load_ps(&frameBuffer[pixels[6]].x), 1);
		pixel37 = _mm256_insertf128_ps(pixel37, _mm_load_ps(&frameBuffer[pixels[7]].x), 1);
		
		pixel04 = _mm256_add_ps(pixel04, radianceR);
		pixel15 = _mm256_add_ps(pixel15, radianceG);
		pixel26 = _mm256_add_ps(pixel26, radianceB);
		pixel37 = _mm256_add_ps(pixel37, radianceA);
		
		_mm_store_ps(&frameBuffer[pixels[0]].x, _mm256_castps256_ps128(pixel04));
		_mm_store_ps(&frameBuffer[pixels[1]].x, _mm256_castps256_ps128(pixel15));
		_mm_store_ps(&frameBuffer[pixels[2]].x, _mm256_castps256_ps128(pixel26));
		_mm_store_ps(&frameBuffer[pixels[3]].x, _mm256_castps256_ps128(pixel37));
		
		_mm_store_ps(&frameBuffer[pixels[4]].x, _mm256_extractf128_ps(pixel04, 1));
		_mm_store_ps(&frameBuffer[pixels[5]].x, _mm256_extractf128_ps(pixel15, 1));
		_mm_store_ps(&frameBuffer[pixels[6]].x, _mm256_extractf128_ps(pixel26, 1));
		_mm_store_ps(&frameBuffer[pixels[7]].x, _mm256_extractf128_ps(pixel37, 1));
		
#if 1
		// Terminate paths at some low contribution.
		unsigned mask0 = _mm256_movemask_ps(_mm256_cmp_ps(weightR, _mm256_set1_ps(0.01f), _CMP_NLE_US));
		unsigned mask1 = _mm256_movemask_ps(_mm256_cmp_ps(weightG, _mm256_set1_ps(0.01f), _CMP_NLE_US));
		unsigned mask2 = _mm256_movemask_ps(_mm256_cmp_ps(weightB, _mm256_set1_ps(0.01f), _CMP_NLE_US));
		
		unsigned mask = mask0 | mask1 | mask2;
#else
		// Always continue paths until max depth. (Possibly more interesting for performance numbers.)
		unsigned mask = 0xff;
#endif
		
		__m256 dDotN = _mm256_fmadd_ps(rayDz, normalZ, _mm256_fmadd_ps(rayDy, normalY, _mm256_mul_ps(rayDx, normalX)));

		// Reflection.
		__m256 cosI = _mm256_mul_ps(dDotN, _mm256_set1_ps(-2.0f));

		__m256 reflectDirX = _mm256_fmadd_ps(cosI, normalX, rayDx);
		__m256 reflectDirY = _mm256_fmadd_ps(cosI, normalY, rayDy);
		__m256 reflectDirZ = _mm256_fmadd_ps(cosI, normalZ, rayDz);
		
		unsigned reflectMask = mask;
		
		// Refraction.
		__m256 eta0 = _mm256_set1_ps(1.0f / 1.1f);
		__m256 eta1 = _mm256_set1_ps(1.1f);
		
		__m256 eta = _mm256_blendv_ps(eta0, eta1, sgn0);
		
		__m256 r = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_mul_ps(eta, eta), _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(dDotN, dDotN))));
		
		__m256 mu = _mm256_fmadd_ps(eta, dDotN, _mm256_sqrt_ps(r));
		
		__m256 refractDirX = _mm256_fmsub_ps(eta, rayDx, _mm256_mul_ps(mu, normalX));
		__m256 refractDirY = _mm256_fmsub_ps(eta, rayDy, _mm256_mul_ps(mu, normalY));
		__m256 refractDirZ = _mm256_fmsub_ps(eta, rayDz, _mm256_mul_ps(mu, normalZ));
		
		unsigned refractMask = mask & _mm256_movemask_ps(_mm256_cmp_ps(r, _mm256_setzero_ps(), _CMP_GT_OQ));

		// Shade.
		__m256 reflectSgn1 = _mm256_fmadd_ps(reflectDirZ, geometryNormalZ, _mm256_fmadd_ps(reflectDirY, geometryNormalY, _mm256_mul_ps(reflectDirX, geometryNormalX)));
		__m256 refractSgn1 = _mm256_fmadd_ps(refractDirZ, geometryNormalZ, _mm256_fmadd_ps(refractDirY, geometryNormalY, _mm256_mul_ps(refractDirX, geometryNormalX)));

		reflectMask &= _mm256_movemask_ps(_mm256_xor_ps(sgn0, reflectSgn1));
		refractMask &= ~_mm256_movemask_ps(_mm256_xor_ps(sgn0, refractSgn1));
		
		reflectSgn1 = _mm256_and_ps(reflectSgn1, _mm256_set1_ps(-0.0f));
		refractSgn1 = _mm256_and_ps(refractSgn1, _mm256_set1_ps(-0.0f));
		
		__m256 posX = _mm256_fmadd_ps(rayDx, t, rayOx);
		__m256 posY = _mm256_fmadd_ps(rayDy, t, rayOy);
		__m256 posZ = _mm256_fmadd_ps(rayDz, t, rayOz);
		
		__m256 reflectPosX = _mm256_fmadd_ps(_mm256_xor_ps(reflectSgn1, geometryNormalX), _mm256_set1_ps(1e-4f), posX);
		__m256 reflectPosY = _mm256_fmadd_ps(_mm256_xor_ps(reflectSgn1, geometryNormalY), _mm256_set1_ps(1e-4f), posY);
		__m256 reflectPosZ = _mm256_fmadd_ps(_mm256_xor_ps(reflectSgn1, geometryNormalZ), _mm256_set1_ps(1e-4f), posZ);
		
		__m256 refractPosX = _mm256_fmadd_ps(_mm256_xor_ps(refractSgn1, geometryNormalX), _mm256_set1_ps(1e-4f), posX);
		__m256 refractPosY = _mm256_fmadd_ps(_mm256_xor_ps(refractSgn1, geometryNormalY), _mm256_set1_ps(1e-4f), posY);
		__m256 refractPosZ = _mm256_fmadd_ps(_mm256_xor_ps(refractSgn1, geometryNormalZ), _mm256_set1_ps(1e-4f), posZ);
		
		pixel = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(pixel), _mm256_set1_epi32(0x1000000)));
		
		// NAN check.
		reflectMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(reflectPosX, reflectPosX, _CMP_EQ_OQ), _mm256_cmp_ps(reflectDirX, reflectDirX, _CMP_EQ_OQ)));
		reflectMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(reflectPosY, reflectPosY, _CMP_EQ_OQ), _mm256_cmp_ps(reflectDirY, reflectDirY, _CMP_EQ_OQ)));
		reflectMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(reflectPosZ, reflectPosZ, _CMP_EQ_OQ), _mm256_cmp_ps(reflectDirZ, reflectDirZ, _CMP_EQ_OQ)));
		
		refractMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(refractPosX, refractPosX, _CMP_EQ_OQ), _mm256_cmp_ps(refractDirX, refractDirX, _CMP_EQ_OQ)));
		refractMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(refractPosY, refractPosY, _CMP_EQ_OQ), _mm256_cmp_ps(refractDirY, refractDirY, _CMP_EQ_OQ)));
		refractMask &= _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(refractPosZ, refractPosZ, _CMP_EQ_OQ), _mm256_cmp_ps(refractDirZ, refractDirZ, _CMP_EQ_OQ)));
		
		minT = _mm256_set1_ps(1e-3f);
		maxT = _mm256_set1_ps(1e+6f);
		
		__m256 ray0, ray1, ray2, ray3, ray4, ray5, ray6, ray7;
		
		ALIGNED(32) float rays0[8*8];
		ALIGNED(32) float rays1[8*8];
		ALIGNED(32) float paths[4*8];
		
		ray0 = reflectPosX;
		ray1 = reflectPosY;
		ray2 = reflectPosZ;
		ray3 = minT;
		ray4 = reflectDirX;
		ray5 = reflectDirY;
		ray6 = reflectDirZ;
		ray7 = maxT;
		
		_MM256_TRANSPOSE8_PS(ray0, ray1, ray2, ray3, ray4, ray5, ray6, ray7);
		
		_mm256_store_ps(rays0 + 0, ray0);
		_mm256_store_ps(rays0 + 8, ray1);
		_mm256_store_ps(rays0 + 16, ray2);
		_mm256_store_ps(rays0 + 24, ray3);
		_mm256_store_ps(rays0 + 32, ray4);
		_mm256_store_ps(rays0 + 40, ray5);
		_mm256_store_ps(rays0 + 48, ray6);
		_mm256_store_ps(rays0 + 56, ray7);
		
		ray0 = refractPosX;
		ray1 = refractPosY;
		ray2 = refractPosZ;
		ray3 = minT;
		ray4 = refractDirX;
		ray5 = refractDirY;
		ray6 = refractDirZ;
		ray7 = maxT;
		
		_MM256_TRANSPOSE8_PS(ray0, ray1, ray2, ray3, ray4, ray5, ray6, ray7);
		
		_mm256_store_ps(rays1 + 0, ray0);
		_mm256_store_ps(rays1 + 8, ray1);
		_mm256_store_ps(rays1 + 16, ray2);
		_mm256_store_ps(rays1 + 24, ray3);
		_mm256_store_ps(rays1 + 32, ray4);
		_mm256_store_ps(rays1 + 40, ray5);
		_mm256_store_ps(rays1 + 48, ray6);
		_mm256_store_ps(rays1 + 56, ray7);
		
		_MM256_TRANSPOSE4_PS(weightR, weightG, weightB, pixel);
		
		_mm256_store_ps(paths + 0, weightR);
		_mm256_store_ps(paths + 8, weightG);
		_mm256_store_ps(paths + 16, weightB);
		_mm256_store_ps(paths + 24, pixel);
		
		unsigned bits = reflectMask | refractMask;
		
		while (bits) {
			unsigned index = ctz(bits);
			unsigned bit = (bits & -(int)bits);
			bits &= bits-1;
			
			bool hasRay0 = (reflectMask & bit) != 0;
			bool hasRay1 = (refractMask & bit) != 0;
			
			__m256 ray0 = _mm256_load_ps(rays0 + 8*index);
			__m256 ray1 = _mm256_load_ps(rays1 + 8*index);
			__m128 path = _mm_load_ps(paths + 4*(((index << 1) & 7) + (index >> 2)));
			
			if (n+index >= activeCount)
				break;
			
			_mm256_store_ps(outputRays[writtenRays].origin, ray0);
			_mm_store_ps(outputLightPaths[writtenRays].weight, path);
			outputRayHeads[writtenRays] = rayHeads[active[n + index]];
			
			if (!hasRay0) {
				_mm256_store_ps(outputRays[writtenRays].origin, ray1);
			}
			else if (hasRay1) {
				_mm256_store_ps(recursiveRays[recursiveRayCount].origin, ray1);
				_mm_store_ps(recursiveLightPaths[recursiveRayCount].weight, path);
				recursiveIndices[recursiveRayCount] = writtenRays;
				++recursiveRayCount;
			}
			
			++writtenRays;
		}
		
		bits = (reflectMask | refractMask) ^ 0xff;
		
		while (bits) {
			unsigned index = ctz(bits);
			bits &= bits-1;
			
			if (n+index < activeCount) {
				unsigned head = rayHeads[active[n + index]];
				
				terminatedHeads[terminatedCount] = head;
				
				if (head)
					++terminatedCount;
			}
		}
	}
	
	n = 0;
	
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
	
	writtenRays = loopHandling(this->loopData, writtenRays,
							   recursiveRays, recursiveLightPaths, recursiveIndices, recursiveRayCount,
							   terminatedHeads, terminatedCount,
							   outputLightPaths, outputRayHeads, outputRays,
							   mutex, freeList, freeCount, arena);
	
	output->count = writtenRays;
}

WhittedRenderer::~WhittedRenderer() {
	_mm_free(payloads);
	_mm_free(loopHeads);
	_mm_free(loopData);
	_mm_free(freeList);
}
