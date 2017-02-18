//
//  LightPath.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-10-08.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_LightPath_h
#define Renderer_LightPath_h

#include "Renderer.h"

struct ALIGNED(16) LightPath {
	float weight[3];
	unsigned pixel;
};

void generateTileLightPaths(LightPath* lightPaths, unsigned viewportStride, unsigned tileX, unsigned tileY, unsigned tileSize);

#endif
