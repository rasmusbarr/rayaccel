//
//  TiledRenderer.h
//  Renderer
//
//  Created by Rasmus Barringer on 2016-03-24.
//  Copyright (c) 2016 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_TiledRenderer_h
#define Renderer_TiledRenderer_h

#include "Renderer.h"
#include "VectorMath.h"
#include <assert.h>
#include <atomic>

struct Arena {
	char* start;
	char* end;
};

inline void* allocate(Arena& arena, unsigned size) {
	size = (size + 63u) & ~63u;
	void* p = arena.start;
	arena.start += size;
	assert(arena.start < arena.end);
	return p;
}

template<class T>
inline T* allocateArray(Arena& arena, unsigned count) {
	return static_cast<T*>(allocate(arena, sizeof(T)*count));
}

class TiledRenderer {
public:
	static const unsigned tileSize = 128;
	static const unsigned hostThreadArenaSize = 4*1024*1024;
	
private:
	ALIGNED(64) std::atomic_int tile;
	
	ALIGNED(64) unsigned threadCount;
	unsigned width;
	unsigned height;
	
	unsigned tileWidth;
	unsigned tileHeight;
	unsigned maxTileCount;
	
public:
	float4* frameBuffer;
	Arena* threadArenas;
	
	TiledRenderer(racc::Context* context, unsigned width, unsigned height);
	
	void clear();
	
	virtual void endFrame();
	
	bool spawnPrimary(unsigned thread, racc::RayStream* output);
	
	virtual void shade(unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output) = 0;
	
	virtual void spawnPrimary(unsigned thread, unsigned tileX, unsigned tileY, unsigned viewportWidth, unsigned viewportHeight, racc::RayStream* output) = 0;
	
	virtual ~TiledRenderer();
};

#endif
