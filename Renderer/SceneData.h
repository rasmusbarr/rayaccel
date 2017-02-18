//
//  SceneData.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-03-04.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_SceneData_h
#define Renderer_SceneData_h

#include "VectorMath.h"

struct SceneData {
	uint16_t maxDepth;
	
	uint16_t viewportWidth;
	uint16_t viewportHeight;
	
	class Material** materials;
	uint32_t* indices;
	uint16_t* triangleMaterials;
	float4* triangleNormals;
	
	float4* normals;
	float2* texcoords;
	
	uint32_t triangleCount;
	uint32_t vertexCount;
};

#endif
