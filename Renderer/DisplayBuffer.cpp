//
//  DisplayBuffer.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-20.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "DisplayBuffer.h"
#include <immintrin.h>

#ifdef _WIN32
#include <GLUT/glext.h>
static PFNGLGENBUFFERSPROC glGenBuffers;
static PFNGLBINDBUFFERPROC glBindBuffer;
static PFNGLMAPBUFFERPROC glMapBuffer;
static PFNGLUNMAPBUFFERPROC glUnmapBuffer;
static PFNGLBUFFERDATAPROC glBufferData;
static PFNGLDELETEBUFFERSPROC glDeleteBuffers;
#endif

static inline void colorConvert(const float4* pixels, uint32_t* image, unsigned width, unsigned height, unsigned spp) {
	__m256 scale = _mm256_set1_ps(255.0f/(int)spp);
	__m256 offset = _mm256_set1_ps(0.5f);
	
	__m256i shuffle = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
	
	for (unsigned i = 0; i < width*height; i += 16) {
		__m256 c0a = _mm256_load_ps(&pixels[i+0].x);
		__m256 c1a = _mm256_load_ps(&pixels[i+2].x);
		__m256 c2a = _mm256_load_ps(&pixels[i+4].x);
		__m256 c3a = _mm256_load_ps(&pixels[i+6].x);
		
		__m256 c0b = _mm256_load_ps(&pixels[i+8].x);
		__m256 c1b = _mm256_load_ps(&pixels[i+10].x);
		__m256 c2b = _mm256_load_ps(&pixels[i+12].x);
		__m256 c3b = _mm256_load_ps(&pixels[i+14].x);
		
		c0a = _mm256_fmadd_ps(c0a, scale, offset);
		c1a = _mm256_fmadd_ps(c1a, scale, offset);
		c2a = _mm256_fmadd_ps(c2a, scale, offset);
		c3a = _mm256_fmadd_ps(c3a, scale, offset);
		
		c0b = _mm256_fmadd_ps(c0b, scale, offset);
		c1b = _mm256_fmadd_ps(c1b, scale, offset);
		c2b = _mm256_fmadd_ps(c2b, scale, offset);
		c3b = _mm256_fmadd_ps(c3b, scale, offset);
		
		__m256i ci0a = _mm256_cvttps_epi32(c0a);
		__m256i ci1a = _mm256_cvttps_epi32(c1a);
		__m256i ci2a = _mm256_cvttps_epi32(c2a);
		__m256i ci3a = _mm256_cvttps_epi32(c3a);
		
		__m256i ci0b = _mm256_cvttps_epi32(c0b);
		__m256i ci1b = _mm256_cvttps_epi32(c1b);
		__m256i ci2b = _mm256_cvttps_epi32(c2b);
		__m256i ci3b = _mm256_cvttps_epi32(c3b);
		
		__m256i ci01a = _mm256_packs_epi32(ci0a, ci1a);
		__m256i ci23a = _mm256_packs_epi32(ci2a, ci3a);
		
		__m256i ci01b = _mm256_packs_epi32(ci0b, ci1b);
		__m256i ci23b = _mm256_packs_epi32(ci2b, ci3b);
		
		__m256i ci0123a = _mm256_packus_epi16(ci01a, ci23a);
		__m256i ci0123b = _mm256_packus_epi16(ci01b, ci23b);
		
		ci0123a = _mm256_permutevar8x32_epi32(ci0123a, shuffle);
		ci0123b = _mm256_permutevar8x32_epi32(ci0123b, shuffle);
		
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(&image[i]), ci0123a);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(&image[i + 8]), ci0123b);
	}
}

DisplayBuffer::DisplayBuffer(unsigned width, unsigned height) : width(width), height(height) {
#ifdef _WIN32
	glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
	glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
	glMapBuffer = (PFNGLMAPBUFFERPROC)wglGetProcAddress("glMapBuffer");
	glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)wglGetProcAddress("glUnmapBuffer");
	glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
	glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
#endif

	currentBuffer = 0;
	
	glEnable(GL_TEXTURE_2D);
	
	glGenBuffers(bufferCount, pbos);
	glGenTextures(bufferCount, textures);
	
	for (unsigned i = 0; i < bufferCount; ++i) {
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[i]);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, 0, GL_STREAM_DRAW);
	}
}

void DisplayBuffer::present(float4* pixels, unsigned spp) {
	glClear(GL_COLOR_BUFFER_BIT);
	
	glBindTexture(GL_TEXTURE_2D, textures[currentBuffer]);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
	glEnd();
	
	currentBuffer = (currentBuffer + 1) % bufferCount;
	
	glBindTexture(GL_TEXTURE_2D, textures[currentBuffer]);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[currentBuffer]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	
	unsigned nextBuffer = (currentBuffer + 1) % bufferCount;
	
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[nextBuffer]);
	uint32_t* image = static_cast<uint32_t*>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
	
	colorConvert(pixels, image, width, height, spp);
	
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

DisplayBuffer::~DisplayBuffer() {
	glDeleteBuffers(bufferCount, pbos);
	glDeleteTextures(bufferCount, textures);
}
