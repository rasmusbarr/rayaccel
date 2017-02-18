//
//  RayAccelerator.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-12.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_RayAccelerator_h
#define RayAccelerator_RayAccelerator_h

#include <stdint.h>
#include <immintrin.h>

#ifdef _WIN32
#include <CL/OpenCL.h>
#define RACC_ALIGNED(n) __declspec(align(n))
#elif __APPLE__
#include <OpenCL/cl.h>
#define RACC_ALIGNED(n) __attribute__((aligned(n)))
#else
#error "Unsupported platform."
#endif

namespace racc {
	static const uint32_t invalidTriangle = ~(uint32_t)0;
	
	struct Context;
	struct Scene;
	struct Environment;
	
	struct Configuration {
		cl_context gpuContext;
		bool allowCpuTracing;			// Do intersection testing on the CPU?
		uint8_t cpuThreads;				// The number of CPU threads used for shading and possibly intersection testing.
		uint8_t gpuSubmissionThreads;	// The number of CPU threads that submit work to the GPU.
		uint32_t maxRaysInFlight;		// The maximum number of rays that can be active in the system.
		uint16_t maxRaysPerSpawn;		// The maximum number of rays that are returned by a spawn callback.
		uint16_t cpuTestBatch;			// The number of rays to test at a time when doing ray queries on the CPU.
		uint16_t cpuShadeBatch;			// The number of rays to shade at a time.
		uint16_t rayStreamBatchSize;	// The size a ray stream must have before being scheduled for intersection testing.
	};
	
	struct ContextInfo {
		uint16_t threadCount;
		uint16_t rayStreamCount;
		uint32_t rayStreamSize;
		uint32_t maxRaysInFlight;
	};
	
	struct RACC_ALIGNED(16) Vertex {
		float x, y, z, w;
	};
	
	struct RACC_ALIGNED(16) Color {
		float r, g, b, a;
	};
	
	struct RACC_ALIGNED(32) Ray {
		float origin[3];
		float minT;
		float dir[3];
		float maxT;
	};
	
	struct RACC_ALIGNED(16) Result {
		uint32_t triangle;
		union {
			struct {
				float t, u, v;
			} hit;
			struct {
				float r, g, b;
			} miss;
		};
	};
	
	struct RayStream {
		uint32_t index;
		uint32_t count;
		Ray* rays;
		Result* results;
	};
	
	struct Stats {
		uint64_t raysTraced;
	};
	
	struct RenderCallbacks {
		void* data;
		bool (*spawn)(void* data, unsigned thread, RayStream* output);
		void (*shade)(void* data, unsigned thread, const RayStream* input, unsigned start, unsigned end, RayStream* output);
	};
	
	void init();
	
	void deinit();
	
	Configuration defaultConfiguration(cl_context gpuContext);
	
	Context* createContext(Configuration configuration);
	
	void destroy(Context* context);
	
	ContextInfo info(Context* context);
	
	Scene* createScene(Context* context, const Vertex* vertices, unsigned vertexCount, const uint32_t* indices, unsigned indexCount);
	
	void destroy(Scene* scene);
	
	Environment* createEnvironment(Context* context, const Color* colors, unsigned width, unsigned height);
	
	void destroy(Environment* environment);
	
	Stats render(Context* context, Scene* scene, Environment* environment, RenderCallbacks callbacks);
}

#endif
