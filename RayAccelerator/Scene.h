//
//  Scene.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-27.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_Scene_h
#define RayAccelerator_Scene_h

#include "RayAccelerator.h"
#include <embree2/rtcore.h>

namespace racc {
	struct Scene {
		RTCScene embreeScene;
		cl_mem gpuNodes;
		cl_mem gpuTriangles;
		cl_mem gpuTriangleIndices;
	};
}

namespace racc_internal {
	void executeRayQueryCPU(racc::Scene* scene, racc::RayStream* rayStream, racc::Environment* environment, unsigned start, unsigned end);
}

#endif
