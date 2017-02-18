//
//  VectorMath.h
//  Renderer
//
//  Created by Rasmus Barringer on 2012-10-21.
//  Copyright (c) 2012 Rasmus Barringer. All rights reserved.
//

#ifndef Renderer_VectorMath_h
#define Renderer_VectorMath_h

#include "Renderer.h"
#include <math.h>

#ifdef M_PI
#undef M_PI
#endif
#define M_PI 3.14159265f

struct float2 {
	float x;
	float y;
};

struct float3 {
	float x;
	float y;
	float z;
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct float4x4 {
	float4 c0;
	float4 c1;
	float4 c2;
	float4 c3;
};

struct ALIGNED(32) float_8 {
	float x[8];
};

struct ALIGNED(32) float2_8 {
	float_8 x;
	float_8 y;
};

struct ALIGNED(32) float3_8 {
	float_8 x;
	float_8 y;
	float_8 z;
};

// Note: These x_as_y conversions rely on union aliasing and are technically not portable.
inline unsigned float_as_uint32(float x) {
	union { float a; unsigned b; } u;
	u.a = x;
	return u.b;
}

inline float uint32_as_float(unsigned x) {
	union { unsigned a; float b; } u;
	u.a = x;
	return u.b;
}

inline float sign(float a) {
    return _mm_cvtss_f32(_mm_or_ps(_mm_and_ps(_mm_set_ss(a), _mm_set_ss(-0.0f)), _mm_set_ss(1.0f)));
}

inline float rsqrt(float x) {
	return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x)));
}

inline float recip(float x) {
	return _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(x)));
}

inline float2 make_float2(float x) {
	float2 r = { x, x };
	return r;
}

inline float2 make_float2(float x, float y) {
	float2 r = { x, y };
	return r;
}

inline float2 xy(float3 v) {
	float2 r = { v.x, v.y };
	return r;
}

inline float2 operator + (float2 a) {
	return a;
}

inline float2 operator - (float2 a) {
	float2 r = { -a.x, -a.y };
	return r;
}

inline float2& operator += (float2& a, float2 b) {
	a.x += b.x;
	a.y += b.y;
	return a;
}

inline float2& operator += (float2& a, float b) {
	a.x += b;
	a.y += b;
	return a;
}

inline float2& operator -= (float2& a, float2 b) {
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

inline float2& operator -= (float2& a, float b) {
	a.x -= b;
	a.y -= b;
	return a;
}

inline float2& operator *= (float2& a, float2 b) {
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

inline float2& operator *= (float2& a, float b) {
	a.x *= b;
	a.y *= b;
	return a;
}

inline float2 operator + (float2 a, float2 b) {
	float2 r = { a.x + b.x, a.y + b.y };
	return r;
}

inline float2 operator + (float a, float2 b) {
	float2 r = { a + b.x, a + b.y };
	return r;
}

inline float2 operator + (float2 a, float b) {
	float2 r = { a.x + b, a.y + b };
	return r;
}

inline float2 operator - (float2 a, float2 b) {
	float2 r = { a.x - b.x, a.y - b.y };
	return r;
}

inline float2 operator - (float a, float2 b) {
	float2 r = { a - b.x, a - b.y };
	return r;
}

inline float2 operator - (float2 a, float b) {
	float2 r = { a.x - b, a.y - b };
	return r;
}

inline float2 operator * (float2 a, float2 b) {
	float2 r = { a.x * b.x, a.y * b.y };
	return r;
}

inline float2 operator * (float a, float2 b) {
	float2 r = { a * b.x, a * b.y };
	return r;
}

inline float2 operator * (float2 a, float b) {
	float2 r = { a.x * b, a.y * b };
	return r;
}

inline float3 make_float3(float x) {
	float3 r = { x, x, x };
	return r;
}

inline float3 make_float3(float x, float y, float z) {
	float3 r = { x, y, z };
	return r;
}

inline float3 xyz(float4 v) {
	float3 r = { v.x, v.y, v.z };
	return r;
}

inline float3 operator + (float3 a) {
	return a;
}

inline float3 operator - (float3 a) {
	float3 r = { -a.x, -a.y, -a.z };
	return r;
}

inline float3& operator += (float3& a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

inline float3& operator += (float3& a, float b) {
	a.x += b;
	a.y += b;
	a.z += b;
	return a;
}

inline float3& operator -= (float3& a, float3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

inline float3& operator -= (float3& a, float b) {
	a.x -= b;
	a.y -= b;
	a.z -= b;
	return a;
}

inline float3& operator *= (float3& a, float3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

inline float3& operator *= (float3& a, float b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}

inline float3 operator + (float3 a, float3 b) {
	float3 r = { a.x + b.x, a.y + b.y, a.z + b.z };
	return r;
}

inline float3 operator + (float a, float3 b) {
	float3 r = { a + b.x, a + b.y, a + b.z };
	return r;
}

inline float3 operator + (float3 a, float b) {
	float3 r = { a.x + b, a.y + b, a.z + b };
	return r;
}

inline float3 operator - (float3 a, float3 b) {
	float3 r = { a.x - b.x, a.y - b.y, a.z - b.z };
	return r;
}

inline float3 operator - (float a, float3 b) {
	float3 r = { a - b.x, a - b.y, a - b.z };
	return r;
}

inline float3 operator - (float3 a, float b) {
	float3 r = { a.x - b, a.y - b, a.z - b };
	return r;
}

inline float3 operator * (float3 a, float3 b) {
	float3 r = { a.x * b.x, a.y * b.y, a.z * b.z };
	return r;
}

inline float3 operator * (float a, float3 b) {
	float3 r = { a * b.x, a * b.y, a * b.z };
	return r;
}

inline float3 operator * (float3 a, float b) {
	float3 r = { a.x * b, a.y * b, a.z * b };
	return r;
}

inline float4 make_float4(float x) {
	float4 r = { x, x, x, x };
	return r;
}

inline float4 make_float4(float3 v, float w) {
	float4 r = { v.x, v.y, v.z, w };
	return r;
}

inline float4 make_float4(float x, float y, float z, float w) {
	float4 r = { x, y, z, w };
	return r;
}

inline float4 operator + (float4 a) {
	return a;
}

inline float4 operator - (float4 a) {
	float4 r = { -a.x, -a.y, -a.z, -a.w };
	return r;
}

inline float4& operator += (float4& a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

inline float4& operator += (float4& a, float b) {
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
	return a;
}

inline float4& operator -= (float4& a, float4 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

inline float4& operator -= (float4& a, float b) {
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
	return a;
}

inline float4& operator *= (float4& a, float4 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

inline float4& operator *= (float4& a, float b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
	return a;
}

inline float4 operator + (float4 a, float4 b) {
	float4 r = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
	return r;
}

inline float4 operator + (float a, float4 b) {
	float4 r = { a + b.x, a + b.y, a + b.z, a + b.w };
	return r;
}

inline float4 operator + (float4 a, float b) {
	float4 r = { a.x + b, a.y + b, a.z + b, a.w + b };
	return r;
}

inline float4 operator - (float4 a, float4 b) {
	float4 r = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
	return r;
}

inline float4 operator - (float a, float4 b) {
	float4 r = { a - b.x, a - b.y, a - b.z, a - b.w };
	return r;
}

inline float4 operator - (float4 a, float b) {
	float4 r = { a.x - b, a.y - b, a.z - b, a.w - b };
	return r;
}

inline float4 operator * (float4 a, float4 b) {
	float4 r = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
	return r;
}

inline float4 operator * (float a, float4 b) {
	float4 r = { a * b.x, a * b.y, a * b.z, a * b.w };
	return r;
}

inline float4 operator * (float4 a, float b) {
	float4 r = { a.x * b, a.y * b, a.z * b, a.w * b };
	return r;
}

inline float dot(float2 a, float2 b) {
	return a.x*b.x + a.y*b.y;
}

inline float length2(float2 v) {
	return dot(v, v);
}

inline float length(float2 v) {
	return sqrtf(length2(v));
}

inline float2 normalize(float2 v) {
	return v * rsqrt(length2(v));
}

inline float dot(float3 a, float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float length2(float3 v) {
	return dot(v, v);
}

inline float length(float3 v) {
	return sqrtf(length2(v));
}

inline float3 normalize(float3 v) {
	return v * rsqrt(length2(v));
}

inline float3 cross(float3 a, float3 b) {
	float3 r = { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
	return r;
}

inline float4 operator * (const float4x4& lhs, float4 rhs) {
	return lhs.c0*rhs.x + lhs.c1*rhs.y + lhs.c2*rhs.z + lhs.c3*rhs.w;
}

inline float4 operator * (const float4x4& lhs, float3 rhs) {
	return lhs.c0*rhs.x + lhs.c1*rhs.y + lhs.c2*rhs.z + lhs.c3;
}

namespace transforms4x4 {
	inline float4x4 identity() {
		float4x4 m = {
			{ 1.0f, 0.0f, 0.0f, 0.0f },
			{ 0.0f, 1.0f, 0.0f, 0.0f },
			{ 0.0f, 0.0f, 1.0f, 0.0f },
			{ 0.0f, 0.0f, 0.0f, 1.0f },
		};
		return m;
	}
	
	inline float4x4 rotation(float angle, float3 axis) {
		float3 a = axis;
		
		float c = cosf(angle);
		float s = sinf(angle);
		float bc = 1.0f - c;
		
		float axx = a.x*a.x;
		float ayy = a.y*a.y;
		float azz = a.z*a.z;
		
		float axy = a.x*a.y;
		float axz = a.x*a.z;
		float ayz = a.y*a.z;
		
		float4x4 m = {
			{ c + axx*bc, axy*bc + a.z*s, axz*bc - a.y*s, 0.0f },
			{ axy*bc - a.z*s, c + ayy*bc, ayz*bc + a.x*s, 0.0f },
			{ axz*bc + a.y*s, ayz*bc - a.x*s, c + azz*bc, 0.0f },
			{ 0.0f, 0.0f, 0.0f, 1.0f },
		};
		
		return m;
	}
}

#endif
