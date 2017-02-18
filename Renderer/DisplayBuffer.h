//
//  DisplayBuffer.h
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-20.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_DisplayBuffer_h
#define Renderer_DisplayBuffer_h

#include "VectorMath.h"
#include <GLUT/GLUT.h>

class DisplayBuffer {
private:
	static const unsigned bufferCount = 3;
	
	unsigned width;
	unsigned height;
	unsigned currentBuffer;
	GLuint pbos[bufferCount];
	GLuint textures[bufferCount];
	
public:
	DisplayBuffer(unsigned width, unsigned height);
	
	void present(float4* pixels, unsigned spp);
	
	~DisplayBuffer();
};

#endif
