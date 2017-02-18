//
//  Bvh2.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-18.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Bvh2.h"
#include "GroupAllocation.h"
#include "ThreadPool.h"
#include "Threading.h"
#include <limits>
#include <algorithm>

namespace {
	struct RACC_ALIGNED(64) Bvh2BuildState {
		int counter;
		racc_internal::ThreadPool* threadPool;
		
		RACC_ALIGNED(64) uint32_t* __restrict sortedIndices[3];
		uint32_t* __restrict temporaryIndices;
		float* __restrict accumulatedSah;
		uint8_t* __restrict leftPartition;
		float* __restrict triangleBounds;
	};
	
	struct RACC_ALIGNED(64) BoundsTaskData {
		Bvh2BuildState* state;
		racc_internal::Bvh2* bvh;
		const racc::Vertex* vertices;
		const uint32_t* indices;
	};
	
	struct RACC_ALIGNED(64) SortTaskData {
		racc_internal::Bvh2* bvh;
		uint32_t* sortedIndices;
	};
	
	struct BuildTaskData {
		Bvh2BuildState* state;
		racc_internal::Bvh2* bvh;
		unsigned node;
	};
}

#ifdef _WIN32
static inline unsigned ctz(unsigned x) {
	DWORD r = 0;
	_BitScanForward(&r, x);
	return r;
}
#else
static inline unsigned ctz(unsigned x) {
	return __builtin_ctz(x);
}
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

static inline __m256 combine(__m128 a, __m128 b) {
	return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
}

static inline float surfaceArea(__m256 bounds) {
	__m128 d = _mm_add_ps(_mm256_castps256_ps128(bounds), _mm256_extractf128_ps(bounds, 1));
	d = _mm_mul_ps(_mm_shuffle_ps(d, d, _MM_SHUFFLE(3,1,0,0)), _mm_shuffle_ps(d, d, _MM_SHUFFLE(3,2,2,1)));
	return _mm_cvtss_f32(_mm_add_ss(_mm_add_ss(d, _mm_shuffle_ps(d, d, _MM_SHUFFLE(2,2,2,2))), _mm_shuffle_ps(d, d, _MM_SHUFFLE(1,1,1,1))));
}

static inline __m256 computeBounds(const float* triangleBounds, const unsigned* remap, unsigned first, unsigned last) {
	__m256 bounds0 = _mm256_load_ps(triangleBounds + remap[first]*8);
	__m256 bounds1 = bounds0;
	__m256 bounds2 = bounds0;
	__m256 bounds3 = bounds0;
	
	unsigned i = first+1;
	
	for (; i+7 < last; i += 8) {
		unsigned i0 = remap[i+0];
		unsigned i1 = remap[i+1];
		unsigned i2 = remap[i+2];
		unsigned i3 = remap[i+3];
		
		__m256 a0 = _mm256_load_ps(triangleBounds + i0*8);
		__m256 a1 = _mm256_load_ps(triangleBounds + i1*8);
		__m256 a2 = _mm256_load_ps(triangleBounds + i2*8);
		__m256 a3 = _mm256_load_ps(triangleBounds + i3*8);
		
		unsigned i4 = remap[i+4];
		unsigned i5 = remap[i+5];
		unsigned i6 = remap[i+6];
		unsigned i7 = remap[i+7];
		
		__m256 a4 = _mm256_load_ps(triangleBounds + i4*8);
		__m256 a5 = _mm256_load_ps(triangleBounds + i5*8);
		__m256 a6 = _mm256_load_ps(triangleBounds + i6*8);
		__m256 a7 = _mm256_load_ps(triangleBounds + i7*8);
		
		a1 = _mm256_max_ps(a1, a0);
		a3 = _mm256_max_ps(a3, a2);
		a5 = _mm256_max_ps(a5, a4);
		a7 = _mm256_max_ps(a7, a6);
		
		bounds0 = _mm256_max_ps(bounds0, a1);
		bounds1 = _mm256_max_ps(bounds1, a3);
		bounds2 = _mm256_max_ps(bounds2, a5);
		bounds3 = _mm256_max_ps(bounds3, a7);
	}
	
	for (; i < last; ++i)
		bounds0 = _mm256_max_ps(bounds0, _mm256_load_ps(triangleBounds + remap[i]*8));
	
	return _mm256_max_ps(_mm256_max_ps(bounds0, bounds1), _mm256_max_ps(bounds2, bounds3));
}

static inline void radixSortUint64High32(uint64_t* __restrict dataA, uint64_t* __restrict tempA, unsigned count) {
	unsigned bucketsA0[257];
	unsigned bucketsA1[257];
	unsigned bucketsA2[257];
	unsigned bucketsA3[257];
	
	for (unsigned i = 0; i < 257; ++i) {
		bucketsA0[i] = 0;
		bucketsA1[i] = 0;
		bucketsA2[i] = 0;
		bucketsA3[i] = 0;
	}
	
	unsigned* histogramA0 = bucketsA0+1;
	unsigned* histogramA1 = bucketsA1+1;
	unsigned* histogramA2 = bucketsA2+1;
	unsigned* histogramA3 = bucketsA3+1;
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t da = dataA[i];
		++histogramA0[(da >> (4 << 3)) & 0xff];
		++histogramA1[(da >> (5 << 3)) & 0xff];
		++histogramA2[(da >> (6 << 3)) & 0xff];
		++histogramA3[(da >> (7 << 3)) & 0xff];
	}
	
	for (unsigned i = 1; i < 256; ++i) {
		bucketsA0[i] += bucketsA0[i-1];
		bucketsA1[i] += bucketsA1[i-1];
		bucketsA2[i] += bucketsA2[i-1];
		bucketsA3[i] += bucketsA3[i-1];
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t da = dataA[i];
		unsigned indexA = bucketsA0[(da >> (4 << 3)) & 0xff]++;
		tempA[indexA] = da;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t da = tempA[i];
		unsigned indexA = bucketsA1[(da >> (5 << 3)) & 0xff]++;
		dataA[indexA] = da;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t da = dataA[i];
		unsigned indexA = bucketsA2[(da >> (6 << 3)) & 0xff]++;
		tempA[indexA] = da;
	}
	
	for (unsigned i = 0; i < count; ++i) {
		uint64_t da = tempA[i];
		unsigned indexA = bucketsA3[(da >> (7 << 3)) & 0xff]++;
		dataA[indexA] = da;
	}
}

static inline void inPlaceUnpackIndicesFromLowUint64ToUint32(uint64_t* keys, unsigned count) {
	for (unsigned i = 0; i < count; i += 32) {
		const float* input = reinterpret_cast<const float*>(keys + i);
		float* output = reinterpret_cast<float*>(keys) + i;
		
		__m256 value0 = _mm256_load_ps(input + 0);
		__m256 value1 = _mm256_load_ps(input + 8);
		__m256 value2 = _mm256_load_ps(input + 16);
		__m256 value3 = _mm256_load_ps(input + 24);
		__m256 value4 = _mm256_load_ps(input + 32);
		__m256 value5 = _mm256_load_ps(input + 40);
		__m256 value6 = _mm256_load_ps(input + 48);
		__m256 value7 = _mm256_load_ps(input + 56);
		
		__m256 i0 = _mm256_shuffle_ps(value0, value1, _MM_SHUFFLE(2,0,2,0));
		__m256 i1 = _mm256_shuffle_ps(value2, value3, _MM_SHUFFLE(2,0,2,0));
		__m256 i2 = _mm256_shuffle_ps(value4, value5, _MM_SHUFFLE(2,0,2,0));
		__m256 i3 = _mm256_shuffle_ps(value6, value7, _MM_SHUFFLE(2,0,2,0));
		
		i0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(i0), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
		i1 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(i1), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
		i2 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(i2), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
		i3 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(i3), (0) | ((2) << 2) | ((1) << 4) | ((3) << 6)));
		
		_mm256_store_ps(output + 0, i0);
		_mm256_store_ps(output + 8, i1);
		_mm256_store_ps(output + 16, i2);
		_mm256_store_ps(output + 24, i3);
	}
}

static inline void partitionAccordingToRef(Bvh2BuildState* state, unsigned axis, unsigned first, unsigned last) {
	uint32_t* __restrict indices = state->sortedIndices[axis];
	
	uint32_t* source = indices + first;
	uint32_t* end = indices + last;
	
	uint32_t* left = source;
	uint32_t* __restrict right = state->temporaryIndices + first;
	
	while (source < end) {
		uint32_t index = *source;
		uint32_t l = state->leftPartition[index];
		
		*left = index;
		*right = index;
		
		left += l;
		right += 1-l;
		
		++source;
	}
	
	std::copy(state->temporaryIndices + first, right, left);
}

static inline void partition(Bvh2BuildState* state, unsigned axis, unsigned first, unsigned last, unsigned pivot) {
	uint32_t* __restrict ref = state->sortedIndices[axis];
	
	for (unsigned i = first; i < pivot; ++i)
		state->leftPartition[ref[i]] = 1;
	
	for (unsigned i = pivot; i < last; ++i)
		state->leftPartition[ref[i]] = 0;
	
	partitionAccordingToRef(state, (axis+1)%3, first, last);
	partitionAccordingToRef(state, (axis+2)%3, first, last);
}

static void buildTask(void* parameter, unsigned thread);

static inline void build(Bvh2BuildState* state, racc_internal::Bvh2* bvh, unsigned nodeIndex) {
	using namespace racc_internal;
	
	Bvh2Node& node = bvh->nodes[nodeIndex];
	
	unsigned first = node.first;
	unsigned last = node.last;
	
	if (nodeIndex != 0) {
		__m256 bounds = computeBounds(state->triangleBounds, state->sortedIndices[0], first, last);
		__m256 flipSign = combine(_mm_set1_ps(-0.0f), _mm_set1_ps(0.0f));
		
		_mm256_storeu_ps(node.bbMin, _mm256_xor_ps(bounds, flipSign));
	}
	
	if (last - first <= 2)
		return;
	
	__m256 bounds = _mm256_loadu_ps(node.bbMin);
	__m256 flipSign = combine(_mm_set1_ps(-0.0f), _mm_set1_ps(0.0f));
	
	bounds = _mm256_xor_ps(bounds, flipSign);
	float psa = surfaceArea(bounds);
	
	unsigned bestDim = 0xffffffff;
	unsigned pivot = 0xffffffff;
	
	if (psa > 0.0f) {
		__m256 bestSah = _mm256_set1_ps(std::numeric_limits<float>::infinity());
		
		for (unsigned dim = 0; dim < 3; ++dim) {
			const unsigned* __restrict sortedIndices = state->sortedIndices[dim];
			const float* __restrict triangleBounds = state->triangleBounds;
			float* __restrict accumulatedSah = state->accumulatedSah;
			
			int i = first;
			
			__m256i increment = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
			
			__m256 bounds = _mm256_load_ps(triangleBounds + sortedIndices[first]*8);
			
			for (; (int)i < (int)last-8; i += 8) {
				__m256 bb0 = _mm256_load_ps(triangleBounds + sortedIndices[i+0]*8);
				__m256 bb1 = _mm256_load_ps(triangleBounds + sortedIndices[i+1]*8);
				__m256 bb2 = _mm256_load_ps(triangleBounds + sortedIndices[i+2]*8);
				__m256 bb3 = _mm256_load_ps(triangleBounds + sortedIndices[i+3]*8);
				
				bb0 = _mm256_max_ps(bb0, bounds);
				
				__m256 bb4 = _mm256_load_ps(triangleBounds + sortedIndices[i+4]*8);
				__m256 bb5 = _mm256_load_ps(triangleBounds + sortedIndices[i+5]*8);
				__m256 bb6 = _mm256_load_ps(triangleBounds + sortedIndices[i+6]*8);
				__m256 bb7 = _mm256_load_ps(triangleBounds + sortedIndices[i+7]*8);
				
				bb1 = _mm256_max_ps(bb1, bb0);
				bb3 = _mm256_max_ps(bb3, bb2);
				bb5 = _mm256_max_ps(bb5, bb4);
				bb7 = _mm256_max_ps(bb7, bb6);
				
				bb3 = _mm256_max_ps(bb3, bb1);
				bb7 = _mm256_max_ps(bb7, bb5);
				
				bb7 = _mm256_max_ps(bb7, bb3);
				bb5 = _mm256_max_ps(bb5, bb3);
				
				bb4 = _mm256_max_ps(bb4, bb3);
				bb2 = _mm256_max_ps(bb2, bb1);
				bb6 = _mm256_max_ps(bb6, bb5);
				
				bounds = bb7;
				
				__m256 d0 = _mm256_add_ps(_mm256_permute2f128_ps(bb0, bb4, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb0, bb4, (1) | ((3) << 4)));
				__m256 d1 = _mm256_add_ps(_mm256_permute2f128_ps(bb1, bb5, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb1, bb5, (1) | ((3) << 4)));
				__m256 d2 = _mm256_add_ps(_mm256_permute2f128_ps(bb2, bb6, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb2, bb6, (1) | ((3) << 4)));
				__m256 d3 = _mm256_add_ps(_mm256_permute2f128_ps(bb3, bb7, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb3, bb7, (1) | ((3) << 4)));
				
				_MM256_TRANSPOSE4_PS(d0, d1, d2, d3);
				
				__m256 lsa = _mm256_fmadd_ps(d0, d1, _mm256_fmadd_ps(d0, d2, _mm256_mul_ps(d1, d2)));
				
				__m256i leftIndices = _mm256_add_epi32(_mm256_set1_epi32(i-first+1), increment);
				__m256 leftSah = _mm256_mul_ps(lsa, _mm256_cvtepi32_ps(leftIndices));
				
				_mm256_storeu_ps(accumulatedSah + i, leftSah);
				
				unsigned worseLhs = _mm256_movemask_ps(_mm256_cmp_ps(leftSah, bestSah, _CMP_GT_OQ));
				
				if (worseLhs) {
					i += ctz(worseLhs);
					goto endRightSweep;
				}
			}
			
			for (; i < (int)last-1; ++i) {
				bounds = _mm256_max_ps(bounds, _mm256_load_ps(triangleBounds + sortedIndices[i]*8));
				accumulatedSah[i] = surfaceArea(bounds) * (float)(i-(int)first+1);
			}
			
		endRightSweep:
			bounds = computeBounds(triangleBounds, sortedIndices, i, last);
			
			unsigned bestPivot = 0xffffffff;
			
			for (; i > (int)first+7; i -= 8) {
				__m256 bb0 = _mm256_load_ps(triangleBounds + sortedIndices[i-0]*8);
				__m256 bb1 = _mm256_load_ps(triangleBounds + sortedIndices[i-1]*8);
				__m256 bb2 = _mm256_load_ps(triangleBounds + sortedIndices[i-2]*8);
				__m256 bb3 = _mm256_load_ps(triangleBounds + sortedIndices[i-3]*8);
				
				bb0 = _mm256_max_ps(bb0, bounds);
				
				__m256 bb4 = _mm256_load_ps(triangleBounds + sortedIndices[i-4]*8);
				__m256 bb5 = _mm256_load_ps(triangleBounds + sortedIndices[i-5]*8);
				__m256 bb6 = _mm256_load_ps(triangleBounds + sortedIndices[i-6]*8);
				__m256 bb7 = _mm256_load_ps(triangleBounds + sortedIndices[i-7]*8);
				
				bb1 = _mm256_max_ps(bb1, bb0);
				bb3 = _mm256_max_ps(bb3, bb2);
				bb5 = _mm256_max_ps(bb5, bb4);
				bb7 = _mm256_max_ps(bb7, bb6);
				
				bb3 = _mm256_max_ps(bb3, bb1);
				bb7 = _mm256_max_ps(bb7, bb5);
				
				bb7 = _mm256_max_ps(bb7, bb3);
				bb5 = _mm256_max_ps(bb5, bb3);
				
				bb4 = _mm256_max_ps(bb4, bb3);
				bb2 = _mm256_max_ps(bb2, bb1);
				bb6 = _mm256_max_ps(bb6, bb5);
				
				bounds = bb7;
				
				__m256 d0 = _mm256_add_ps(_mm256_permute2f128_ps(bb0, bb4, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb0, bb4, (1) | ((3) << 4)));
				__m256 d1 = _mm256_add_ps(_mm256_permute2f128_ps(bb1, bb5, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb1, bb5, (1) | ((3) << 4)));
				__m256 d2 = _mm256_add_ps(_mm256_permute2f128_ps(bb2, bb6, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb2, bb6, (1) | ((3) << 4)));
				__m256 d3 = _mm256_add_ps(_mm256_permute2f128_ps(bb3, bb7, (0) | ((2) << 4)),
										  _mm256_permute2f128_ps(bb3, bb7, (1) | ((3) << 4)));
				
				_MM256_TRANSPOSE4_PS(d0, d1, d2, d3);
				
				__m256 rsa = _mm256_fmadd_ps(d0, d1, _mm256_fmadd_ps(d0, d2, _mm256_mul_ps(d1, d2)));
				
				__m256 leftSah = _mm256_loadu_ps(accumulatedSah + (i - 8));
				
				leftSah = _mm256_shuffle_ps(leftSah, leftSah, _MM_SHUFFLE(0,1,2,3));
				leftSah = _mm256_permute2f128_ps(leftSah, leftSah, (1) | ((0) << 4));
				
				__m256i rightIndices = _mm256_add_epi32(_mm256_set1_epi32(last-i), increment);
				
				__m256 rightSah = _mm256_mul_ps(rsa, _mm256_cvtepi32_ps(rightIndices));
				__m256 sah = _mm256_add_ps(leftSah, rightSah);
				
				unsigned better = _mm256_movemask_ps(_mm256_cmp_ps(sah, bestSah, _CMP_LT_OQ));
				unsigned worseRhs = _mm256_movemask_ps(_mm256_cmp_ps(rightSah, bestSah, _CMP_GT_OQ));
				
				__m256 minSah = _mm256_min_ps(sah, _mm256_shuffle_ps(sah, sah, _MM_SHUFFLE(2,3,0,1)));
				minSah = _mm256_min_ps(minSah, _mm256_shuffle_ps(minSah, minSah, _MM_SHUFFLE(1,0,3,2)));
				minSah = _mm256_min_ps(minSah, _mm256_permute2f128_ps(minSah, minSah, (1) | ((0) << 4)));
				
				unsigned mask = _mm256_movemask_ps(_mm256_cmp_ps(minSah, sah, _CMP_EQ_OQ));
				
				bestSah = _mm256_min_ps(minSah, bestSah); // Note: The operand order is important here to avoid NaN.
				
				if (better)
					bestPivot = i - ctz(mask);
				
				if (worseRhs)
					goto endLeftSweep;
			}
			
			{
				float bestSahScalar = _mm_cvtss_f32(_mm256_castps256_ps128(bestSah));
				
				for (; i > (int)first; --i) {
					bounds = _mm256_max_ps(bounds, _mm256_load_ps(triangleBounds + sortedIndices[i]*8));
					
					float leftSah = accumulatedSah[i - 1];
					float rsa = surfaceArea(bounds);
					
					float sah = leftSah + rsa * (float)((int)last - i);
					
					if (sah < bestSahScalar) {
						bestSahScalar = sah;
						bestPivot = i;
					}
				}
				
				bestSah = _mm256_set1_ps(bestSahScalar);
			}
			
		endLeftSweep:
			if (bestPivot != 0xffffffff) {
				pivot = bestPivot;
				bestDim = dim;
			}
		}
		
		const float traversalCost = 2.0f;
		const float intersectionCost = 1.0f;
		
		float cost = traversalCost + intersectionCost * _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(psa))) * _mm_cvtss_f32(_mm256_castps256_ps128(bestSah));
		
		if (cost > (float)((int)(last-first)) * intersectionCost) {
			if (last - first >= 127) {
				bestDim = 0;
				pivot = (first + last) >> 1;
			}
			else {
				return;
			}
		}
	}
	else {
		if (last - first >= 127) {
			bestDim = 0;
			pivot = (first + last) >> 1;
		}
		else {
			return;
		}
	}
	
	partition(state, bestDim, first, last, pivot);
	
	unsigned counter = atomicIncrement(&state->counter)*2 + 1;
	
	unsigned left = counter-2;
	unsigned right = counter-1;
	
	node.kind = bestDim + 1;
	node.first = left;
	node.last = right;
	
	Bvh2Node& leftNode = bvh->nodes[left];
	Bvh2Node& rightNode = bvh->nodes[right];
	
	leftNode.kind = 0;
	leftNode.parent = nodeIndex;
	leftNode.first = first;
	leftNode.last = pivot;
	
	rightNode.kind = 0;
	rightNode.parent = nodeIndex;
	rightNode.first = pivot;
	rightNode.last = last;
	
	if (first + 512 < pivot) {
		// Build left subtree in parallel.
		BuildTaskData taskData = {
			state,
			bvh,
			left,
		};
		
		// Store task data in unused part of the node struct to avoid allocating memory.
		void* memory = leftNode.bbMin;
		memcpy(memory, &taskData, sizeof(taskData));
		
		ThreadPoolTask task = {
			memory,
			buildTask,
		};
		
		spawn(state->threadPool, task);
	}
	else {
		build(state, bvh, left);
	}
	
	build(state, bvh, right);
}

static inline void computeBoundsAndKeysThread(Bvh2BuildState* state, racc_internal::Bvh2* bvh, unsigned thread, const racc::Vertex* __restrict vertices, const uint32_t* __restrict indices) {
	using namespace racc_internal;
	
	unsigned alignedTriangleCount = bvh->triangleCount & (~31);
	
	float* __restrict midX = reinterpret_cast<float*>(state->sortedIndices[0]);
	float* __restrict midY = reinterpret_cast<float*>(state->sortedIndices[1]);
	float* __restrict midZ = reinterpret_cast<float*>(state->sortedIndices[2]);
	float* __restrict bounds = state->triangleBounds;
	
	__m256 sceneBounds0 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
	__m256 sceneBounds1 = sceneBounds0;
	__m256 sceneBounds2 = sceneBounds0;
	__m256 sceneBounds3 = sceneBounds0;
	
	__m256 half = _mm256_set1_ps(0.5f);
	__m256 neg = _mm256_set1_ps(-0.0f);
	
	for (;;) {
		static const unsigned trianglesPerBatch = 4*1024;
		
		unsigned start = (atomicIncrement(&state->counter)-1) * trianglesPerBatch;
		
		if (start >= alignedTriangleCount)
			break;
		
		unsigned end = std::min(start + trianglesPerBatch, alignedTriangleCount);
		
		for (unsigned i = start; i < end; i += 4) { // Compute bounds and mid points.
			const uint32_t* indices0 = indices + (i+0)*3;
			const uint32_t* indices1 = indices + (i+1)*3;
			const uint32_t* indices2 = indices + (i+2)*3;
			const uint32_t* indices3 = indices + (i+3)*3;
			
			__m256 p00 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices0[0]].x));
			__m256 p10 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices0[1]].x));
			__m256 p20 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices0[2]].x));
			
			__m256 p01 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices2[0]].x));
			__m256 p11 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices2[1]].x));
			__m256 p21 = _mm256_castps128_ps256(_mm_load_ps(&vertices[indices2[2]].x));
			
			p00 = _mm256_insertf128_ps(p00, _mm_load_ps(&vertices[indices1[0]].x), 1);
			p10 = _mm256_insertf128_ps(p10, _mm_load_ps(&vertices[indices1[1]].x), 1);
			p20 = _mm256_insertf128_ps(p20, _mm_load_ps(&vertices[indices1[2]].x), 1);
			
			p01 = _mm256_insertf128_ps(p01, _mm_load_ps(&vertices[indices3[0]].x), 1);
			p11 = _mm256_insertf128_ps(p11, _mm_load_ps(&vertices[indices3[1]].x), 1);
			p21 = _mm256_insertf128_ps(p21, _mm_load_ps(&vertices[indices3[2]].x), 1);
			
			__m256 bbMin0 = _mm256_min_ps(_mm256_min_ps(p00, p10), p20);
			__m256 bbMax0 = _mm256_max_ps(_mm256_max_ps(p00, p10), p20);
			__m256 bbMin1 = _mm256_min_ps(_mm256_min_ps(p01, p11), p21);
			__m256 bbMax1 = _mm256_max_ps(_mm256_max_ps(p01, p11), p21);
			
			__m256 mid01 = _mm256_add_ps(bbMin0, bbMax0);
			__m256 mid23 = _mm256_add_ps(bbMin1, bbMax1);
			
			bbMin0 = _mm256_xor_ps(bbMin0, neg);
			bbMin1 = _mm256_xor_ps(bbMin1, neg);
			
			__m256 bb00 = _mm256_permute2f128_ps(bbMin0, bbMax0, (0) | ((2) << 4));
			__m256 bb10 = _mm256_permute2f128_ps(bbMin0, bbMax0, (1) | ((3) << 4));
			__m256 bb01 = _mm256_permute2f128_ps(bbMin1, bbMax1, (0) | ((2) << 4));
			__m256 bb11 = _mm256_permute2f128_ps(bbMin1, bbMax1, (1) | ((3) << 4));
			
			mid01 = _mm256_mul_ps(mid01, half);
			mid23 = _mm256_mul_ps(mid23, half);
			
			sceneBounds0 = _mm256_max_ps(sceneBounds0, bb00);
			sceneBounds1 = _mm256_max_ps(sceneBounds1, bb10);
			sceneBounds2 = _mm256_max_ps(sceneBounds2, bb01);
			sceneBounds3 = _mm256_max_ps(sceneBounds3, bb11);
			
			__m128 mid0 = _mm256_castps256_ps128(mid01);
			__m128 mid1 = _mm256_extractf128_ps(mid01, 1);
			__m128 mid2 = _mm256_castps256_ps128(mid23);
			__m128 mid3 = _mm256_extractf128_ps(mid23, 1);
			
			_MM_TRANSPOSE4_PS(mid0, mid1, mid2, mid3);
			
			_mm256_store_ps(bounds + i*8, bb00);
			_mm256_store_ps(bounds + i*8 + 8, bb10);
			_mm256_store_ps(bounds + i*8 + 16, bb01);
			_mm256_store_ps(bounds + i*8 + 24, bb11);
			
			unsigned midIndex = (i<<1) - (i&4);
			
			_mm_store_ps(midX + midIndex, mid0);
			_mm_store_ps(midY + midIndex, mid1);
			_mm_store_ps(midZ + midIndex, mid2);
		}
		
		for (unsigned dim = 0; dim < 3; ++dim) { // Prepare for sorting; pack mid point and index in 64-bit keys.
			int64_t* keys = reinterpret_cast<int64_t*>(state->sortedIndices[dim]) + start;
			float* values = reinterpret_cast<float*>(keys);
			
			unsigned count = end - start;
			
			__m256i i0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
			__m256i i1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
			__m256i i2 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
			__m256i i3 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
			
			__m256i offset = _mm256_set1_epi32(start);
			
			i0 = _mm256_add_epi32(i0, offset);
			i1 = _mm256_add_epi32(i1, offset);
			i2 = _mm256_add_epi32(i2, offset);
			i3 = _mm256_add_epi32(i3, offset);
			
			__m256i increment = _mm256_set1_epi32(32);
			__m256 signMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
			__m256 fullMask = _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff));
			
			for (unsigned i = 0; i < count; i += 32) {
				const float* input = values + i*2;
				float* output = reinterpret_cast<float*>(keys + i);
				
				__m256 value0 = _mm256_load_ps(input + 0*2*8);
				__m256 value1 = _mm256_load_ps(input + 1*2*8);
				__m256 value2 = _mm256_load_ps(input + 2*2*8);
				__m256 value3 = _mm256_load_ps(input + 3*2*8);
				
				__m256 mask0 = _mm256_blendv_ps(signMask, fullMask, value0);
				__m256 mask1 = _mm256_blendv_ps(signMask, fullMask, value1);
				__m256 mask2 = _mm256_blendv_ps(signMask, fullMask, value2);
				__m256 mask3 = _mm256_blendv_ps(signMask, fullMask, value3);
				
				__m256 encodedValue0 = _mm256_xor_ps(mask0, value0);
				__m256 encodedValue1 = _mm256_xor_ps(mask1, value1);
				__m256 encodedValue2 = _mm256_xor_ps(mask2, value2);
				__m256 encodedValue3 = _mm256_xor_ps(mask3, value3);
				
				__m256 key00 = _mm256_unpacklo_ps(_mm256_castsi256_ps(i0), encodedValue0);
				__m256 key01 = _mm256_unpackhi_ps(_mm256_castsi256_ps(i0), encodedValue0);
				__m256 key10 = _mm256_unpacklo_ps(_mm256_castsi256_ps(i1), encodedValue1);
				__m256 key11 = _mm256_unpackhi_ps(_mm256_castsi256_ps(i1), encodedValue1);
				__m256 key20 = _mm256_unpacklo_ps(_mm256_castsi256_ps(i2), encodedValue2);
				__m256 key21 = _mm256_unpackhi_ps(_mm256_castsi256_ps(i2), encodedValue2);
				__m256 key30 = _mm256_unpacklo_ps(_mm256_castsi256_ps(i3), encodedValue3);
				__m256 key31 = _mm256_unpackhi_ps(_mm256_castsi256_ps(i3), encodedValue3);
				
				_mm256_store_ps(output + 0*8, key00);
				_mm256_store_ps(output + 1*8, key01);
				_mm256_store_ps(output + 2*8, key10);
				_mm256_store_ps(output + 3*8, key11);
				_mm256_store_ps(output + 4*8, key20);
				_mm256_store_ps(output + 5*8, key21);
				_mm256_store_ps(output + 6*8, key30);
				_mm256_store_ps(output + 7*8, key31);
				
				i0 = _mm256_add_epi32(i0, increment);
				i1 = _mm256_add_epi32(i1, increment);
				i2 = _mm256_add_epi32(i2, increment);
				i3 = _mm256_add_epi32(i3, increment);
			}
		}
	}
	
	__m256 sceneBounds = _mm256_max_ps(_mm256_max_ps(sceneBounds0, sceneBounds1), _mm256_max_ps(sceneBounds2, sceneBounds3));
	_mm256_store_ps(state->accumulatedSah + thread*8, sceneBounds);
}

static inline void computeBoundsAndKeysUnaligned(Bvh2BuildState* state, racc_internal::Bvh2* bvh, const racc::Vertex* vertices, const uint32_t* indices) {
	using namespace racc_internal;
	
	float* __restrict bounds = state->triangleBounds;
	
	uint64_t* __restrict keysX = reinterpret_cast<uint64_t*>(state->sortedIndices[0]);
	uint64_t* __restrict keysY = reinterpret_cast<uint64_t*>(state->sortedIndices[1]);
	uint64_t* __restrict keysZ = reinterpret_cast<uint64_t*>(state->sortedIndices[2]);
	
	__m128 half = _mm_set1_ps(0.5f);
	__m128 neg = _mm_set1_ps(-0.0f);
	
	__m256 sceneBounds = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
	
	unsigned alignedTriangleCount = bvh->triangleCount & (~31);
	
	for (unsigned i = alignedTriangleCount; i < bvh->triangleCount; ++i) {
		const uint32_t* indices0 = indices + i*3;
		
		__m128 p0 = _mm_load_ps(&vertices[indices0[0]].x);
		__m128 p1 = _mm_load_ps(&vertices[indices0[1]].x);
		__m128 p2 = _mm_load_ps(&vertices[indices0[2]].x);
		
		__m128 bbMin = _mm_min_ps(_mm_min_ps(p0, p1), p2);
		__m128 bbMax = _mm_max_ps(_mm_max_ps(p0, p1), p2);
		
		__m128 mid = _mm_mul_ps(_mm_add_ps(bbMin, bbMax), half);
		
		RACC_ALIGNED(16) unsigned midCoords[4];
		_mm_store_ps(reinterpret_cast<float*>(midCoords), mid);
		
		bbMin = _mm_xor_ps(bbMin, neg);
		
		int inputX = midCoords[0];
		int inputY = midCoords[1];
		int inputZ = midCoords[2];
		
		__m256 bb = combine(bbMin, bbMax);
		sceneBounds = _mm256_max_ps(sceneBounds, bb);
		
		_mm256_store_ps(bounds + i*8, bb);
		
		inputX ^= inputX < 0 ? 0xffffffff : 0x80000000;
		inputY ^= inputY < 0 ? 0xffffffff : 0x80000000;
		inputZ ^= inputZ < 0 ? 0xffffffff : 0x80000000;
		
		keysX[i] = ((uint64_t)inputX << 32) | i;
		keysY[i] = ((uint64_t)inputY << 32) | i;
		keysZ[i] = ((uint64_t)inputZ << 32) | i;
	}
	
	_mm256_store_ps(state->accumulatedSah + threadCount(state->threadPool)*8, sceneBounds);
}

static void boundsTask(void* parameter, unsigned thread) {
	BoundsTaskData* data = static_cast<BoundsTaskData*>(parameter);
	computeBoundsAndKeysThread(data->state, data->bvh, thread, data->vertices, data->indices);
}

static void sortTask(void* parameter, unsigned) {
	SortTaskData* data = static_cast<SortTaskData*>(parameter);
	uint64_t* indices = reinterpret_cast<uint64_t*>(data->sortedIndices);
	radixSortUint64High32(indices, indices + data->bvh->triangleCount, data->bvh->triangleCount);
	inPlaceUnpackIndicesFromLowUint64ToUint32(indices, data->bvh->triangleCount);
}

static void buildTask(void* parameter, unsigned) {
	BuildTaskData data = *static_cast<BuildTaskData*>(parameter);
	build(data.state, data.bvh, data.node);
}

racc_internal::Bvh2* racc_internal::createBvh2(const racc::Vertex* __restrict vertices, unsigned vertexCount, const uint32_t* __restrict indices, unsigned triangleCount) {
	(void)vertexCount; // Unused.
	
	unsigned threadCount = cpuCount();
	
	// Allocate BVH memory as a single block.
	void* bvhMemory = 0;
	void* nodeMemory = 0;
	void* triangleMemory = 0;
	
	Allocation bvhAllocations[] = {
		{ sizeof(Bvh2), 64, &bvhMemory },
		{ (unsigned)sizeof(Bvh2Node)*(triangleCount*2), 64, &nodeMemory },
		{ (unsigned)sizeof(uint32_t)*triangleCount*4+256, 128, &triangleMemory },
	};
	
	if (!allocateGroup(bvhAllocations, 128)) {
		fprintf(stderr, "RayAccelerator: Unable to allocate memory.");
		return 0;
	}
	
	Bvh2* bvh = static_cast<Bvh2*>(bvhMemory);
	
	bvh->nodes = static_cast<Bvh2Node*>(nodeMemory);
	bvh->triangles = reinterpret_cast<uint32_t*>(triangleMemory);
	bvh->triangleCount = triangleCount;
	
	// Allocate build state memory as a single block.
	void* buildStateMemory = 0;
	void* sortedIndices1Memory = 0;
	void* sortedIndices2Memory = 0;
	void* temporaryIndicesMemory = 0;
	void* accumulatedSahMemory = 0;
	void* leftPartitionMemory = 0;
	void* triangleBoundsMemory = 0;
	
	Allocation stateAllocations[] = {
		{ sizeof(Bvh2BuildState), 64, &buildStateMemory },
		{ (unsigned)sizeof(uint32_t)*triangleCount*4+256, 128, &sortedIndices1Memory },
		{ (unsigned)sizeof(uint32_t)*triangleCount*4+256, 128, &sortedIndices2Memory },
		{ (unsigned)sizeof(uint32_t)*triangleCount, 64, &temporaryIndicesMemory },
		{ (unsigned)sizeof(float)*std::max(triangleCount, (threadCount + 1)*8), 64, &accumulatedSahMemory },
		{ (unsigned)sizeof(uint8_t)*triangleCount, 64, &leftPartitionMemory },
		{ (unsigned)sizeof(float)*triangleCount*8, 64, &triangleBoundsMemory },
	};
	
	if (!allocateGroup(stateAllocations, 128)) {
		fprintf(stderr, "RayAccelerator: Unable to allocate memory.");
		_mm_free(bvh);
		return 0;
	}
	
	Bvh2BuildState* state = static_cast<Bvh2BuildState*>(buildStateMemory);
	
	state->counter = 0;
	state->threadPool = createThreadPool(threadCount);
	
	state->sortedIndices[0] = bvh->triangles;
	state->sortedIndices[1] = reinterpret_cast<uint32_t*>(sortedIndices1Memory);
	state->sortedIndices[2] = reinterpret_cast<uint32_t*>(sortedIndices2Memory);
	
	state->temporaryIndices = reinterpret_cast<uint32_t*>(temporaryIndicesMemory);
	state->accumulatedSah = reinterpret_cast<float*>(accumulatedSahMemory);
	state->leftPartition = reinterpret_cast<uint8_t*>(leftPartitionMemory);
	
	state->triangleBounds = reinterpret_cast<float*>(triangleBoundsMemory);
	
	__m256 sceneBounds = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
	
	for (unsigned i = 0; i <= threadCount; ++i)
		_mm256_store_ps(state->accumulatedSah + i*8, sceneBounds);
	
	// Compute bounds and keys for all triangles in parallel.
	{
		BoundsTaskData taskData = {
			state,
			bvh,
			vertices,
			indices,
		};
		
		ThreadPoolTask task = {
			&taskData,
			boundsTask,
		};
		
		spawn(state->threadPool, task, threadCount);
		computeBoundsAndKeysUnaligned(state, bvh, vertices, indices);
		join(state->threadPool);
	}
	
	// Sort each axis in parallel with computing bounds for the root node.
	{
		SortTaskData taskData[] = {
			{ bvh, state->sortedIndices[0] },
			{ bvh, state->sortedIndices[1] },
			{ bvh, state->sortedIndices[2] },
		};
		
		ThreadPoolTask tasks[] = {
			{ &taskData[0], sortTask },
			{ &taskData[1], sortTask },
			{ &taskData[2], sortTask },
		};
		
		spawnArray(state->threadPool, tasks, 3);
		
		for (unsigned i = 0; i <= threadCount; ++i)
			sceneBounds = _mm256_max_ps(sceneBounds, _mm256_load_ps(state->accumulatedSah + i*8));
		
		Bvh2Node& node = bvh->nodes[0];
		
		node.kind = 0;
		node.parent = 0xffffffff;
		node.first = 0;
		node.last = triangleCount;
		
		__m256 flipSign = combine(_mm_set1_ps(-0.0f), _mm_set1_ps(0.0f));
		
		_mm256_storeu_ps(node.bbMin, _mm256_xor_ps(sceneBounds, flipSign));
		
		join(state->threadPool);
	}
	
	// Do recursive build.
	state->counter = 0;
	build(state, bvh, 0);
	join(state->threadPool);
	
	bvh->nodeCount = state->counter*2 + 1;
	
	destroy(state->threadPool);
	_mm_free(state);
	
	return bvh;
}

void racc_internal::destroy(Bvh2* bvh) {
	_mm_free(bvh);
}
