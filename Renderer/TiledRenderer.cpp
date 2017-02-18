//
//  TiledRenderer.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2016-03-24.
//  Copyright (c) 2016 Rasmus Barringer. All rights reserved.
//

#include "TiledRenderer.h"

TiledRenderer::TiledRenderer(racc::Context* context, unsigned width, unsigned height) {
	racc::ContextInfo info = racc::info(context);
	
	threadCount = info.threadCount;
	tile = 0;
	
	this->width = width;
	this->height = height;
	
	tileWidth = width/tileSize;
	tileHeight = height/tileSize;
	maxTileCount = tileWidth*tileHeight;
	
	frameBuffer = static_cast<float4*>(_mm_malloc(sizeof(float4)*width*height, 64));
	
	threadArenas = new Arena[threadCount];
	
	for (unsigned i = 0; i < threadCount; ++i) {
		Arena arena = {};
		
		arena.start = static_cast<char*>(_mm_malloc(hostThreadArenaSize, 64));
		arena.end = arena.start + hostThreadArenaSize;
		
		threadArenas[i] = arena;
	}
	
	clear();
}

void TiledRenderer::clear() {
	__m256 zero = _mm256_setzero_ps();
	
	for (unsigned i = 0; i < width*height; i += 8) {
		_mm256_store_ps(&frameBuffer[i+0].x, zero);
		_mm256_store_ps(&frameBuffer[i+2].x, zero);
		_mm256_store_ps(&frameBuffer[i+4].x, zero);
		_mm256_store_ps(&frameBuffer[i+6].x, zero);
	}
}

void TiledRenderer::endFrame() {
	tile = 0;
}

bool TiledRenderer::spawnPrimary(unsigned thread, racc::RayStream* output) {
	int tile = this->tile++;
	
	if ((unsigned)tile >= maxTileCount)
		return false;
	
	unsigned tileX = (tile % tileWidth) * tileSize;
	unsigned tileY = (tile / tileWidth) * tileSize;
	
	spawnPrimary(thread, tileX, tileY, width, height, output);
	
	return tile != maxTileCount-1;
}

TiledRenderer::~TiledRenderer() {
	_mm_free(frameBuffer);
	
	for (unsigned i = 0; i < threadCount; ++i) {
		_mm_free(threadArenas[i].end - hostThreadArenaSize);
	}
	
	delete [] threadArenas;
}
