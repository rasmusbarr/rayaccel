//
//  Camera.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-18.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Camera.h"
#include "SimdRandom.h"
#include <math.h>

void Camera::lookAt(float3 origin, float3 target, float3 up, float fov, float near, float far, int width, int height) {
	float3 forward = normalize(target - origin);
	float3 right = normalize(cross(forward, up));
	float3 cameraUp = cross(right, forward);
	
	float aspect = (float)width / (float)height;
	float imageExtentX = tanf(0.5f * fov * (M_PI/180.0f)) * aspect;
	float imageExtentY = tanf(0.5f * fov * (M_PI/180.0f));
	
	this->origin = origin;
	this->right = right * (-2.0f/(float)width  * imageExtentX);
	this->up = cameraUp * (-2.0f/(float)height * imageExtentY);
	this->view = forward + right * imageExtentX + cameraUp * imageExtentY;
}

void Camera::rotate(float angle, float3 axis) {
	rotate(angle, axis, origin);
}

void Camera::rotate(float angle, float3 axis, float3 pivot) {
	float4x4 transform = transforms4x4::rotation(angle, axis);
	
	view = xyz(transform * view);
	right = xyz(transform * right);
	up = xyz(transform * up);
	
	origin -= pivot;
	origin = xyz(transform * origin);
	origin += pivot;
}

float3 Camera::forward() const {
	float3 n = normalize(right);
	float3 t = normalize(up);
	
	float3 forward = view;
	forward -= n * dot(forward, n);
	forward -= t * dot(forward, t);
	
	return normalize(forward);
}

void generateTileRays(racc::Ray* rays, Camera camera, unsigned tileX, unsigned tileY, unsigned tileSize) {
	__m128 originTmin = _mm_setr_ps(camera.origin.x, camera.origin.y, camera.origin.z, 0.0f);
	
	SimdRandom8 rng(rand());
	
	for (unsigned y = 0; y < tileSize; ++y) {
		float fy = (float)(int)(y+tileY);
		float fx = (float)(int)tileX;
		
		__m256 opx = _mm256_add_ps(_mm256_set1_ps(fx), _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));
		__m256 opy = _mm256_set1_ps(fy);
		
		for (unsigned x = 0; x < tileSize; x += 8) {
			__m256 px = _mm256_add_ps(opx, rng.nextFloat());
			__m256 py = _mm256_add_ps(opy, rng.nextFloat());
			
			__m256 eX = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.up.x), py, _mm256_broadcast_ss(&camera.view.x));
			__m256 eY = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.up.y), py, _mm256_broadcast_ss(&camera.view.y));
			__m256 eZ = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.up.z), py, _mm256_broadcast_ss(&camera.view.z));
			
			__m256 dX = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.right.x), px, eX);
			__m256 dY = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.right.y), px, eY);
			__m256 dZ = _mm256_fmadd_ps(_mm256_broadcast_ss(&camera.right.z), px, eZ);
			
			__m256 scale = _mm256_rsqrt_ps(_mm256_fmadd_ps(dZ, dZ, _mm256_fmadd_ps(dY, dY, _mm256_mul_ps(dX, dX))));
			
			dX = _mm256_mul_ps(dX, scale);
			dY = _mm256_mul_ps(dY, scale);
			dZ = _mm256_mul_ps(dZ, scale);
			
			__m256 tMax = _mm256_set1_ps(1e+6f);
			
			_MM256_TRANSPOSE4_PS(dX, dY, dZ, tMax);
			
			__m256 r0 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dX, (0) | ((2) << 4));
			__m256 r1 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dX, (0) | ((3) << 4));
			__m256 r2 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dY, (0) | ((2) << 4));
			__m256 r3 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dY, (0) | ((3) << 4));
			
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 0, r0);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 8, r1);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 16, r2);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 24, r3);
			
			__m256 r4 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dZ, (0) | ((2) << 4));
			__m256 r5 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), dZ, (0) | ((3) << 4));
			__m256 r6 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), tMax, (0) | ((2) << 4));
			__m256 r7 = _mm256_permute2f128_ps(_mm256_castps128_ps256(originTmin), tMax, (0) | ((3) << 4));
			
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 32, r4);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 40, r5);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 48, r6);
			_mm256_store_ps(reinterpret_cast<float*>(rays) + 56, r7);
			
			rays += 8;
			
			opx = _mm256_add_ps(opx, _mm256_set1_ps(8.0f));
		}
	}
}
