//
//  Materials.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-03-15.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_Materials_h
#define Renderer_Materials_h

#include "Renderer.h"
#include "VectorMath.h"

class Material {
public:
	virtual void sample8(const float3_8* rnd, const float3_8* normal, const float3_8* uvt, const float3_8* wo, float3_8* wi, float3_8* color, unsigned* transmitted) = 0;
	
	virtual ~Material();
};

class ReflectiveDiffuseMaterial : public Material {
private:
	ALIGNED(16) float ke[4];
	
public:
	ReflectiveDiffuseMaterial(float3 k, float eta);
	
	virtual void sample8(const float3_8* rnd, const float3_8* normal, const float3_8* uvt, const float3_8* wo, float3_8* wi, float3_8* color, unsigned* transmitted);
};

#endif
