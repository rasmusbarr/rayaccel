//
//  Scene.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-27.
//  Copyright (c) 2014 Rasmus Barringer and Tomas Akenine-Möller. All rights reserved.
//

#include "Scene.h"
#include "Bvh2.h"
#include "Context.h"
#include "Environment.h"
#include <embree2/rtcore_ray.h>
#include <assert.h>
#include <string.h>
#include <vector>

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

namespace {
	struct TriangleIndices {
		uint32_t x, y, z;
	};
	
	struct Vector3 {
		float x, y, z;
	};
	
	struct Node {
		uint32_t kind, parent;
		uint32_t first, last;
		Vector3 leftMin, leftMax;
		Vector3 rightMin, rightMax;
	};
	
	// Two triangles packed in a pair, with e1 as shared edge.
	// TriangleIndices 0 has vertices p0, p0+e1, and p0+e2.
	// TriangleIndices 1 has vertices p0, p0+e3, and p0+e1.
	struct TrianglePair {
		Vector3 e1; float e3x;
		Vector3 e2; float e3y;
		Vector3 p0; float e3z;
	};
}

static inline Vector3 xyz(racc::Vertex v) {
	Vector3 r = { v.x, v.y, v.z };
	return r;
}

static inline Vector3 operator - (Vector3 a, Vector3 b) {
	Vector3 r = { a.x - b.x, a.y - b.y, a.z - b.z };
	return r;
}

static inline TriangleIndices reorder(const uint32_t* src, unsigned first) {
	TriangleIndices dst = {
		src[first%3],
		src[(first+1)%3],
		src[(first+2)%3],
	};
	return dst;
}

static bool findSharedEdgeTriangle(const uint32_t* tri0, const uint32_t* tri1, unsigned& sharedEdge0, unsigned& sharedEdge1) {
	for (unsigned edge0 = 0; edge0 < 3; ++edge0) {
		for (unsigned edge1 = 0; edge1 < 3; ++edge1) {
			if (tri0[edge0] == tri1[(edge1+1)%3] && tri0[(edge0+1)%3] == tri1[edge1]) {
				sharedEdge0 = edge0;
				sharedEdge1 = edge1;
				return true;
			}
		}
	}
	return false;
}

static void mergeTriangle(unsigned firstTriIdx, std::vector<unsigned>& candidates, const racc::Vertex* vertices, const uint32_t* indices,
						  std::vector<TriangleIndices>& trianglePairIndices, std::vector<TrianglePair>& trianglePairs, std::vector<unsigned>& triangleRemap) {
	const uint32_t* firstTri = indices + firstTriIdx*3;
	
	// Try to pair the triangle with one in the list.
	for (unsigned i = 0; i < (unsigned)candidates.size(); ++i) {
		const uint32_t* secondTri = indices + candidates[i]*3;
		
		unsigned edge0, edge1;
		if (findSharedEdgeTriangle(firstTri, secondTri, edge0, edge1)) {
			triangleRemap[(unsigned)trianglePairIndices.size()] = firstTriIdx | (edge0 << 30);
			triangleRemap[(unsigned)trianglePairIndices.size() + 1] = candidates[i] | ((edge1+1) << 30);
			
			trianglePairIndices.push_back(reorder(firstTri, edge0));
			trianglePairIndices.push_back(reorder(secondTri, edge1+1));
			
			candidates.erase(candidates.begin()+i);
			
			// Create a triangle pair.
			Vector3 verts0[] = { xyz(vertices[firstTri[0]]),  xyz(vertices[firstTri[1]]),  xyz(vertices[firstTri[2]])};
			Vector3 verts1[] = { xyz(vertices[secondTri[0]]), xyz(vertices[secondTri[1]]), xyz(vertices[secondTri[2]])};
			
			Vector3 p0 = verts0[edge0];
			Vector3 p1 = verts0[(edge0+1)%3];
			Vector3 p2 = verts0[(edge0+2)%3];
			Vector3 p3 = verts1[(edge1+2)%3];
			
			TrianglePair pair = {
				p0 - p1, p3.x - p0.x,
				p2 - p0, p3.y - p0.y,
				p0, p3.z - p0.z,
			};
			
			trianglePairs.push_back(pair);
			return;
		}
	}
	
	// Make degenerate pair from the single triangle that did not match.
	triangleRemap[(unsigned)trianglePairIndices.size()] = firstTriIdx;
	
	TriangleIndices triangle0 = { firstTri[0], firstTri[1], firstTri[2] };
	TriangleIndices triangle1 = { 0xffffffff, 0xffffffff, 0xffffffff};
	
	trianglePairIndices.push_back(triangle0);
	trianglePairIndices.push_back(triangle1);
	
	Vector3 p0 = xyz(vertices[firstTri[0]]);
	Vector3 p1 = xyz(vertices[firstTri[1]]);
	Vector3 p2 = xyz(vertices[firstTri[2]]);
	Vector3 p3 = xyz(vertices[firstTri[1]]);
	
	TrianglePair pair = {
		p0 - p1, p3.x - p0.x,
		p2 - p0, p3.y - p0.y,
		p0, p3.z - p0.z,
	};
	
	trianglePairs.push_back(pair);
}

racc::Scene* racc::createScene(Context* context, const Vertex* vertices, unsigned vertexCount, const uint32_t* indices, unsigned indexCount) {
	using namespace racc_internal;
	
	assert(indexCount % 3 == 0);
	assert((uintptr_t)vertices % 16 == 0);
	
	unsigned triangleCount = indexCount/3;
	
	Scene* scene = static_cast<Scene*>(_mm_malloc(sizeof(Scene), 64));
	
	if (!scene)
		return 0;
	
	// Create Embree scene.
	{
		RTCScene embreeScene = rtcNewScene(RTC_SCENE_STATIC, RTC_INTERSECT1|RTC_INTERSECT8);
		
		unsigned geomID = rtcNewTriangleMesh(embreeScene, RTC_GEOMETRY_STATIC, triangleCount, vertexCount);
		
		void* meshVertices = rtcMapBuffer(embreeScene, geomID, RTC_VERTEX_BUFFER);
		memcpy(meshVertices, vertices, sizeof(Vertex)*vertexCount);
		rtcUnmapBuffer(embreeScene, geomID, RTC_VERTEX_BUFFER);
		
		void* meshTriangles = rtcMapBuffer(embreeScene, geomID, RTC_INDEX_BUFFER);
		memcpy(meshTriangles, indices, sizeof(uint32_t)*indexCount);
		rtcUnmapBuffer(embreeScene, geomID, RTC_INDEX_BUFFER);
		
		rtcSetMask(embreeScene, geomID, 0xffffffff);
		rtcCommit(embreeScene);
		
		scene->embreeScene = embreeScene;
	}
	
	// Create GPU scene.
	if (context->configuration.gpuContext) {
		// Create BVH.
		Bvh2* bvh = createBvh2(vertices, vertexCount, indices, triangleCount);
		
		unsigned nodeCount = bvh->nodeCount;
		
		// Make triangle pairs in leaf nodes.
		std::vector<TrianglePair> trianglePairs;
		std::vector<unsigned> pairIndexToOriginal;
		
		trianglePairs.reserve(triangleCount);
		{
			std::vector<TriangleIndices> trianglePairIndices;
			std::vector<unsigned> triangleRemap;
			std::vector<unsigned> candidates;
			
			trianglePairIndices.reserve(triangleCount*2);
			triangleRemap.resize(triangleCount*2, ~0u);
			candidates.reserve(128);
			
			for (unsigned i = 0; i < nodeCount; ++i) {
				Bvh2Node node = bvh->nodes[i];
				
				if (!node.kind) {
					// Setup list of triangles in leaf.
					candidates.resize(0);
					
					for (unsigned j = node.first; j < node.last; ++j)
						candidates.push_back(bvh->triangles[j]);
					
					// Update start triangle in BVH.
					bvh->nodes[i].first = (unsigned)trianglePairs.size();
					
					// Try to merge them.
					while (!candidates.empty()) {
						unsigned first = candidates.front();
						candidates.erase(candidates.begin());
						
						mergeTriangle(first, candidates, vertices, indices, trianglePairIndices, trianglePairs, triangleRemap);
					}
					
					// Update end triangle in BVH.
					bvh->nodes[i].last = (unsigned)trianglePairs.size();
				}
			}
			
			// Create pair to original triangle mapping.
			pairIndexToOriginal.resize(trianglePairs.size()*2, 0);
			
			for (unsigned i = 0; i < (unsigned)pairIndexToOriginal.size(); ++i) {
				unsigned j = triangleRemap[i];
				
				if (j != ~0u)
					pairIndexToOriginal[i] = j;
			}
		}
		
		// Translate BVH nodes to GPU format.
		std::vector<Node> nodes;
		nodes.reserve(nodeCount);
		{
			std::vector<unsigned> indexRemap;
			indexRemap.resize(nodeCount);
			
			for (unsigned i = 0; i < nodeCount; ++i) {
				Bvh2Node node = bvh->nodes[i];
				
				if (!node.kind)
					continue;
				
				const float* leftMin = bvh->nodes[node.first].bbMin;
				const float* leftMax = bvh->nodes[node.first].bbMax;
				const float* rightMin = bvh->nodes[node.last].bbMin;
				const float* rightMax = bvh->nodes[node.last].bbMax;
				
				indexRemap[i] = (unsigned)nodes.size();
				
				if (!bvh->nodes[node.first].kind) {
					unsigned start = bvh->nodes[node.first].first;
					unsigned end = bvh->nodes[node.first].last;
					
					node.first = ((end-start) << 24) | start;
				}
				else {
					node.first |= 0x80000000;
				}
				
				if (!bvh->nodes[node.last].kind) {
					unsigned start = bvh->nodes[node.last].first;
					unsigned end = bvh->nodes[node.last].last;
					
					node.last = ((end-start) << 24) | start;
				}
				else {
					node.last |= 0x80000000;
				}
				
				Node n = {
					node.kind, node.parent,
					node.first, node.last,
					{ leftMin[0], leftMin[1], leftMin[2] },
					{ leftMax[0], leftMax[1], leftMax[2] },
					{ rightMin[0], rightMin[1], rightMin[2] },
					{ rightMax[0], rightMax[1], rightMax[2] },
				};
				
				nodes.push_back(n);
			}
			
			for (unsigned i = 0; i < (unsigned)nodes.size(); ++i) {
				if (nodes[i].first & 0x80000000)
					nodes[i].first = 0x80000000 | indexRemap[nodes[i].first & (~0x80000000)];
				
				if (nodes[i].last & 0x80000000)
					nodes[i].last = 0x80000000 | indexRemap[nodes[i].last & (~0x80000000)];
			}
			
			// Add padding.
			do {
				trianglePairs.push_back(trianglePairs[0]);
			}
			while ((trianglePairs.size() * (sizeof(TrianglePair) / (sizeof(float)*4))) % 32 != 0);
		}
		
		// Create buffers.
		scene->gpuNodes = clCreateBuffer(context->configuration.gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Node)*(unsigned)nodes.size(), &nodes[0], 0);
		
		scene->gpuTriangles = clCreateBuffer(context->configuration.gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(TrianglePair)*(unsigned)trianglePairs.size(), &trianglePairs[0], 0);
		
		scene->gpuTriangleIndices = clCreateBuffer(context->configuration.gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned)*(unsigned)pairIndexToOriginal.size(), &pairIndexToOriginal[0], 0);
		
		destroy(bvh);
	}
	else {
		scene->gpuNodes = 0;
		scene->gpuTriangles = 0;
		scene->gpuTriangleIndices = 0;
	}
	
	return scene;
}

void racc::destroy(Scene* scene) {
	rtcDeleteScene(scene->embreeScene);
	
	if (scene->gpuNodes)
		clReleaseMemObject(scene->gpuNodes);
	
	if (scene->gpuTriangles)
		clReleaseMemObject(scene->gpuTriangles);
	
	if (scene->gpuTriangleIndices)
		clReleaseMemObject(scene->gpuTriangleIndices);
	
	_mm_free(scene);
}

void racc_internal::executeRayQueryCPU(racc::Scene* scene, racc::RayStream* rayStream, racc::Environment* environment, unsigned start, unsigned end) {
	using namespace racc;
	
	const Ray* rays = rayStream->rays;
	Result* results = rayStream->results;
	
	unsigned i = start;
	
	RACC_ALIGNED(32) uint32_t valid[8];
	_mm256_store_si256(reinterpret_cast<__m256i*>(valid), _mm256_set1_epi32(-1));
	
	for (; i+7 < end; i += 8) {
		__m256 r0 = _mm256_load_ps(rays[i+0].origin);
		__m256 r1 = _mm256_load_ps(rays[i+1].origin);
		__m256 r2 = _mm256_load_ps(rays[i+2].origin);
		__m256 r3 = _mm256_load_ps(rays[i+3].origin);
		__m256 r4 = _mm256_load_ps(rays[i+4].origin);
		__m256 r5 = _mm256_load_ps(rays[i+5].origin);
		__m256 r6 = _mm256_load_ps(rays[i+6].origin);
		__m256 r7 = _mm256_load_ps(rays[i+7].origin);
		
		_MM256_TRANSPOSE8_PS(r0, r1, r2, r3, r4, r5, r6, r7);
		
		RTCRay8 embreeRay;
		
		_mm256_store_ps(embreeRay.orgx, r0);
		_mm256_store_ps(embreeRay.orgy, r1);
		_mm256_store_ps(embreeRay.orgz, r2);
		
		_mm256_store_ps(embreeRay.dirx, r4);
		_mm256_store_ps(embreeRay.diry, r5);
		_mm256_store_ps(embreeRay.dirz, r6);
		
		_mm256_store_ps(embreeRay.tnear, r3);
		_mm256_store_ps(embreeRay.tfar, r7);
		
		_mm256_store_si256(reinterpret_cast<__m256i*>(embreeRay.geomID), _mm256_set1_epi32(RTC_INVALID_GEOMETRY_ID));
		_mm256_store_si256(reinterpret_cast<__m256i*>(embreeRay.primID), _mm256_set1_epi32(RTC_INVALID_GEOMETRY_ID));
		_mm256_store_si256(reinterpret_cast<__m256i*>(embreeRay.instID), _mm256_set1_epi32(RTC_INVALID_GEOMETRY_ID));
		_mm256_store_si256(reinterpret_cast<__m256i*>(embreeRay.mask), _mm256_set1_epi32(0xFFFFFFFF));
		_mm256_store_ps(embreeRay.time, _mm256_setzero_ps());
		
		rtcIntersect8(valid, scene->embreeScene, embreeRay);
		
		r0 = _mm256_load_ps(reinterpret_cast<const float*>(embreeRay.primID));
		r1 = _mm256_load_ps(embreeRay.tfar);
		r2 = _mm256_load_ps(embreeRay.u);
		r3 = _mm256_load_ps(embreeRay.v);
		
		_MM256_TRANSPOSE4_PS(r0, r1, r2, r3);
		
		_mm256_store_ps(reinterpret_cast<float*>(results + i+0), _mm256_permute2f128_ps(r0, r1, (0) | ((2) << 4)));
		_mm256_store_ps(reinterpret_cast<float*>(results + i+2), _mm256_permute2f128_ps(r2, r3, (0) | ((2) << 4)));
		_mm256_store_ps(reinterpret_cast<float*>(results + i+4), _mm256_permute2f128_ps(r0, r1, (1) | ((3) << 4)));
		_mm256_store_ps(reinterpret_cast<float*>(results + i+6), _mm256_permute2f128_ps(r2, r3, (1) | ((3) << 4)));
		
		for (unsigned index = i; index < i+8; ++index) {
			Result& result = results[index];
			
			if (result.triangle == racc::invalidTriangle) {
				RACC_ALIGNED(16) float sample[4];
				_mm_store_ps(sample, racc_internal::sample(environment, _mm_load_ps(rays[index].dir)));
				result.miss.r = sample[0];
				result.miss.g = sample[1];
				result.miss.b = sample[2];
			}
		}
	}
	
	for (; i < end; ++i) {
		Ray ray = rays[i];
		Result result = { 0xffffffff };
		
		RTCRay embreeRay;
		
		embreeRay.org[0] = ray.origin[0];
		embreeRay.org[1] = ray.origin[1];
		embreeRay.org[2] = ray.origin[2];
		
		embreeRay.dir[0] = ray.dir[0];
		embreeRay.dir[1] = ray.dir[1];
		embreeRay.dir[2] = ray.dir[2];
		
		embreeRay.tnear = ray.minT;
		embreeRay.tfar = ray.maxT;
		
		embreeRay.geomID = RTC_INVALID_GEOMETRY_ID;
		embreeRay.primID = RTC_INVALID_GEOMETRY_ID;
		embreeRay.instID = RTC_INVALID_GEOMETRY_ID;
		embreeRay.mask = 0xFFFFFFFF;
		embreeRay.time = 0.0f;
		
		rtcIntersect(scene->embreeScene, embreeRay);
		
		if (embreeRay.geomID != RTC_INVALID_GEOMETRY_ID) {
			result.triangle = embreeRay.primID;
			result.hit.t = embreeRay.tfar;
			result.hit.u = embreeRay.u;
			result.hit.v = embreeRay.v;
		}
		else {
			RACC_ALIGNED(16) float sample[4];
			_mm_store_ps(sample, racc_internal::sample(environment, _mm_load_ps(ray.dir)));
			result.miss.r = sample[0];
			result.miss.g = sample[1];
			result.miss.b = sample[2];
		}
		
		results[i] = result;
	}
}
