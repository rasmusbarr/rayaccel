//
//  Bvh2.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-18.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_Bvh2_h
#define RayAccelerator_Bvh2_h

#include "RayAccelerator.h"

namespace racc_internal {
	struct Bvh2Node {
		uint32_t kind, parent;
		uint32_t first, last;
		float bbMin[3];
		uint32_t dummy0;
		float bbMax[3];
		uint32_t dummy1;
	};
	
	struct Bvh2 {
		Bvh2Node* nodes;
		uint32_t* triangles;
		uint32_t nodeCount;
		uint32_t triangleCount;
	};
	
	Bvh2* createBvh2(const racc::Vertex* __restrict vertices, unsigned vertexCount, const uint32_t* __restrict indices, unsigned triangleCount);
	
	void destroy(Bvh2* bvh);
}

#endif
