//
//  WhittedRenderer.h
//  Renderer
//
//  Created by Rasmus Barringer on 2015-09-19.
//  Copyright (c) 2015 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_WhittedRenderer_h
#define Renderer_WhittedRenderer_h

#include "TiledRenderer.h"
#include <mutex>

struct Camera;
struct SceneData;
struct LightPath;
struct LoopData;

class WhittedRenderer : public TiledRenderer {
private:
	Camera& camera;
	SceneData& scene;
	LightPath* payloads;
	uint32_t* loopHeads;
	LoopData* loopData;
	uint32_t* freeList;
	unsigned rayStreamStride;
	unsigned maxShadingItems;
	
	ALIGNED(64) std::mutex mutex;
	unsigned freeCount;
	
public:
	WhittedRenderer(racc::Context* context, Camera& camera, SceneData& scene);
	
	virtual void endFrame();
	
	void spawnPrimary(unsigned thread, unsigned tileX, unsigned tileY, unsigned viewportWidth, unsigned viewportHeight, racc::RayStream* output);
	
	void shade(unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output);
	
	~WhittedRenderer();
};

#endif
