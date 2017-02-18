//
//  Environment.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-03-13.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Environment.h"
#include "Context.h"
#include <string.h>

racc::Environment* racc::createEnvironment(Context* context, const Color* colors, unsigned width, unsigned height) {
	unsigned pixelCount = width*height;
	Environment* environment = static_cast<Environment*>(_mm_malloc(sizeof(Environment) + pixelCount*4*sizeof(float), 64));
	
	if (!environment)
		return 0;
	
	environment->dimensions[0] = (float)(int)width;
	environment->dimensions[1] = (float)(int)height;
	environment->dimensions[2] = (float)(int)width;
	environment->dimensions[3] = (float)(int)height;
	
	environment->bounds[0] = (int)width-1;
	environment->bounds[1] = (int)height-1;
	environment->bounds[2] = (int)width-1;
	environment->bounds[3] = (int)height-1;
	
	environment->width = width;
	environment->height = height;
	
	memcpy(environment->pixels, colors, pixelCount*4*sizeof(float));
	
	if (context->configuration.gpuContext) {
		cl_image_format fmt = {
			CL_RGBA,
			CL_FLOAT,
		};
		
		cl_image_desc desc = {
			CL_MEM_OBJECT_IMAGE2D,
			width,
			height,
			0,
			0,
			width*16,
		};
		
		environment->gpuImage = clCreateImage(context->configuration.gpuContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, const_cast<Color*>(colors), 0);
		
		if (!environment->gpuImage) {
			_mm_free(environment);
			return 0;
		}
	}
	else {
		environment->gpuImage = 0;
	}
	
	return environment;
}

void racc::destroy(Environment* environment) {
	if (environment->gpuImage)
		clReleaseMemObject(environment->gpuImage);
	
	_mm_free(environment);
}
