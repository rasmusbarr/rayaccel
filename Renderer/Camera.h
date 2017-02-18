//
//  Camera.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-18.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_Camera_h
#define Renderer_Camera_h

#include "Renderer.h"
#include "VectorMath.h"

struct Camera {
	float3 origin;
	float3 view;
	float3 right;
	float3 up;

	void lookAt(float3 origin, float3 target, float3 up, float fov, float near, float far, int width, int height);
	
	void rotate(float angle, float3 axis);
	
	void rotate(float angle, float3 axis, float3 pivot);
	
	float3 forward() const;
};

void generateTileRays(racc::Ray* rays, Camera camera, unsigned tileX, unsigned tileY, unsigned tileSize);

#endif
