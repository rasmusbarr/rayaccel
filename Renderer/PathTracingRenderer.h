//
//  PathTracingRenderer.h
//  Renderer
//
//  Created by Rasmus Barringer on 2015-09-16.
//  Copyright (c) 2015 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_PathTracingRenderer_h
#define Renderer_PathTracingRenderer_h

#include "TiledRenderer.h"

struct Camera;
struct SceneData;
struct LightPath;

class PathTracingRenderer : public TiledRenderer {
private:
	Camera& camera;
	SceneData& scene;
	LightPath* payloads;
	unsigned rayStreamStride;
	
public:
	PathTracingRenderer(racc::Context* context, Camera& camera, SceneData& scene);
	
	void spawnPrimary(unsigned thread, unsigned tileX, unsigned tileY, unsigned viewportWidth, unsigned viewportHeight, racc::RayStream* output);
	
	void shade(unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output);
	
	virtual ~PathTracingRenderer();
};

#endif
