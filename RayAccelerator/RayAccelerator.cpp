//
//  RayAccelerator.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-12.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "GroupAllocation.h"
#include "Environment.h"
#include "Threading.h"
#include "Context.h"
#include "Kernels.h"
#include "Scene.h"
#include <assert.h>
#include <stdio.h>

#define RACC_WORK_GROUP_SIZE 8

#define RACC_STRING(a) #a
#define RACC_STRING_EXPANDED(a) RACC_STRING(a)

namespace {
	struct RACC_ALIGNED(64) CpuWorkerData {
		racc::Context* context;
		unsigned index;
	};
	
	struct RACC_ALIGNED(64) GpuWorkerData {
		racc::Context* context;
		cl_command_queue queue;
		cl_kernel kernel;
	};
}

struct RACC_ALIGNED(64) racc_internal::GpuRayStream {
	racc::RayStream data;
	cl_mem rays;
	cl_mem results;
};

struct racc_internal::CpuTestJob {
	uint16_t rayStream;
	uint16_t outstandingThreads;
	uint32_t raysTested;
};

static inline bool spawnRays(racc::Context* context, unsigned thread, unsigned maxRaysPerSpawn, unsigned maxRaysInFlight) {
	using namespace racc_internal;
	using namespace racc;
	
	if (!context->moreRaysExist || context->raysInFlight + maxRaysPerSpawn > maxRaysInFlight)
		return false;
	
	RenderCallbacks callbacks = context->currentCallbacks;
	
	unsigned rayStreamBatchSize = context->configuration.rayStreamBatchSize;
	
	// Pick a ray stream.
	unsigned rayStreamIndex = context->waitingToBeFilledCount ? context->waitingToBeFilled[--context->waitingToBeFilledCount] : context->empty[--context->emptyCount];
	RayStream* rayStream = &context->rayStreams[rayStreamIndex].data;
	
	context->raysInFlight += maxRaysPerSpawn;
	exit(context->schedulingMutex);
	
	// Do spawn.
	unsigned previousCount = rayStream->count;
	bool more = callbacks.spawn(callbacks.data, thread, rayStream);
	unsigned added = rayStream->count - previousCount;
	
	assert(added <= maxRaysPerSpawn);
	
	enter(context->schedulingMutex);
	context->raysInFlight -= maxRaysPerSpawn - added;
	
	// Put ray stream back.
	if (rayStream->count >= rayStreamBatchSize)
		context->readyForTest[context->readyForTestCount++] = rayStreamIndex;
	else if (rayStream->count == 0)
		context->empty[context->emptyCount++] = rayStreamIndex;
	else
		context->waitingToBeFilled[context->waitingToBeFilledCount++] = rayStreamIndex;
	
	notifyAll(context->schedulingMutex);
	
	if (!more && context->moreRaysExist)
		context->moreRaysExist = 0;
	
	return true;
}

static inline bool shadeRays(racc::Context* context, unsigned thread) {
	using namespace racc_internal;
	using namespace racc;
	
	if (!context->readyForShadeCount)
		return false;
	
	RenderCallbacks callbacks = context->currentCallbacks;
	
	unsigned cpuShadeBatch = context->configuration.cpuShadeBatch;
	unsigned rayStreamBatchSize = context->configuration.rayStreamBatchSize;
	
	// Pick a ray stream.
	unsigned rayStreamIndex = context->readyForShade[--context->readyForShadeCount];
	RayStream* rayStream = &context->rayStreams[rayStreamIndex].data;
	
	unsigned start = 0;
	unsigned end = (rayStream->count < cpuShadeBatch ? rayStream->count : cpuShadeBatch);
	
	for (;;) {
		// Get an output ray stream for this batch.
		unsigned outputRayStreamIndex = context->waitingToBeFilledCount ? context->waitingToBeFilled[--context->waitingToBeFilledCount] : context->empty[--context->emptyCount];
		RayStream* outputRayStream = &context->rayStreams[outputRayStreamIndex].data;
		
		exit(context->schedulingMutex);
		
		// Do shade.
		unsigned previousCount = outputRayStream->count;
		callbacks.shade(callbacks.data, thread, rayStream, start, end, outputRayStream);
		unsigned added = outputRayStream->count - previousCount;
		
		assert(added <= end - start);
		
		enter(context->schedulingMutex);
		
		context->raysInFlight += added;
		
		// Put output ray stream back.
		if (outputRayStream->count >= rayStreamBatchSize)
			context->readyForTest[context->readyForTestCount++] = outputRayStreamIndex;
		else if (outputRayStream->count == 0)
			context->empty[context->emptyCount++] = outputRayStreamIndex;
		else
			context->waitingToBeFilled[context->waitingToBeFilledCount++] = outputRayStreamIndex;
		
		notifyAll(context->schedulingMutex);
		
		if (end == rayStream->count)
			break;
		
		start += cpuShadeBatch;
		end += cpuShadeBatch;
		
		if (end > rayStream->count)
			end = rayStream->count;
	}
	
	// Clear and put ray stream back.
	context->raysInFlight -= rayStream->count;
	rayStream->count = 0;
	
	context->empty[context->emptyCount++] = rayStreamIndex;
	
	return true;
}

static inline void queryRays(racc::Context* context, unsigned slot, racc::RayStream* stream, unsigned start, unsigned end) {
	using namespace racc_internal;
	using namespace racc;
	
	++context->cpuThreadsTesting;
	exit(context->schedulingMutex);
	
	// Do ray query.
	executeRayQueryCPU(context->currentScene, stream, context->currentEnvironment, start, end);
	
	enter(context->schedulingMutex);
	--context->cpuThreadsTesting;
	
	// Free slot if finished (and existing).
	CpuTestJob& job = context->cpuTestJobs[slot];
	
	if (job.rayStream == stream->index) {
		if (--job.outstandingThreads == 0 && job.raysTested >= stream->count) {
			job.rayStream = 0xffff;
			context->readyForShade[context->readyForShadeCount++] = stream->index;
			
			notifyAll(context->schedulingMutex);
		}
	}
	else {
		context->readyForShade[context->readyForShadeCount++] = stream->index;
		
		notifyAll(context->schedulingMutex);
	}
}

static inline void startRayQuery(racc::Context* context, unsigned rayStreamIndex, unsigned freeSlot) {
	using namespace racc_internal;
	using namespace racc;
	
	CpuTestJob& job = context->cpuTestJobs[freeSlot];
	
	RayStream* rayStream = &context->rayStreams[rayStreamIndex].data;
	
	unsigned cpuTestBatch = context->configuration.cpuTestBatch;
	unsigned count = rayStream->count;
	
	context->rayCount += count;
	
	if (count > cpuTestBatch) {
		// Put ray stream into job slot so that other threads can help out.
		job.rayStream = rayStreamIndex;
		job.raysTested = cpuTestBatch;
		job.outstandingThreads = 1;
		
		notifyAll(context->schedulingMutex);
	}
	
	// Query the first range.
	queryRays(context, freeSlot, rayStream, 0, (count < cpuTestBatch ? count : cpuTestBatch));
}

static inline bool continueRayQuery(racc::Context* context, unsigned cpuThreads, unsigned& freeSlot) {
	using namespace racc_internal;
	using namespace racc;
	
	unsigned cpuTestBatch = context->configuration.cpuTestBatch;
	
	for (unsigned i = 0; i < cpuThreads; ++i) {
		CpuTestJob& job = context->cpuTestJobs[i];
		
		if (job.rayStream != 0xffff) {
			RayStream* rayStream = &context->rayStreams[job.rayStream].data;
			
			if (job.raysTested < rayStream->count) {
				// Query next range in an existing job slot.
				unsigned start = job.raysTested;
				job.raysTested += cpuTestBatch;
				++job.outstandingThreads;
				
				unsigned count = rayStream->count;
				queryRays(context, i, rayStream, start, (count < start+cpuTestBatch ? count : start+cpuTestBatch));
				return true;
			}
		}
		else {
			freeSlot = i;
		}
	}
	
	return false;
}

// Note: Thread functions are passed to a template and cannot be static.
namespace {
	void cpuWorkerThread(void* memory) {
		using namespace racc_internal;
		using namespace racc;
		
		CpuWorkerData* data = static_cast<CpuWorkerData*>(memory);
		
		Context* context = data->context;
		Mutex& mutex = context->schedulingMutex;
		
		enter(mutex);
		
		unsigned thread = data->index;
		
		unsigned rayStreamCount = context->rayStreamCount;
		unsigned maxRaysPerSpawn = context->configuration.maxRaysPerSpawn;
		unsigned maxRaysInFlight = context->configuration.maxRaysInFlight;
		
		unsigned cpuThreads = context->configuration.cpuThreads;
		unsigned gpuThreads = context->configuration.gpuSubmissionThreads;
		
		// Note that a CPU context and a hybrid context will schedule differently.
		if (context->configuration.gpuContext) {
			bool allowCpuTracing = context->configuration.allowCpuTracing;
			
			for (;;) {
				if (context->shouldExit && !context->moreRaysExist && context->emptyCount == rayStreamCount)
					break;
				
				if (spawnRays(context, thread, maxRaysPerSpawn, maxRaysInFlight))
					continue;
				
				if (shadeRays(context, thread))
					continue;
				
				unsigned freeSlot = 0xffffffff;
				
				if (continueRayQuery(context, cpuThreads, freeSlot))
					continue;
				
				if (freeSlot != 0xffffffff && allowCpuTracing) {
					if (context->readyForTestCount > gpuThreads - context->gpuThreadsTesting) {
						startRayQuery(context, context->readyForTest[--context->readyForTestCount], freeSlot);
						continue;
					}
					
					if (context->waitingToBeFilledCount > 0) {
						startRayQuery(context, context->waitingToBeFilled[--context->waitingToBeFilledCount], freeSlot);
						continue;
					}
				}
				
				wait(mutex);
			}
		}
		else {
			for (;;) {
				if (context->shouldExit && !context->moreRaysExist && context->emptyCount == rayStreamCount)
					break;
				
				unsigned freeSlot = 0xffffffff;
				
				if (continueRayQuery(context, cpuThreads, freeSlot))
					continue;
				
				if (shadeRays(context, thread))
					continue;
				
				if (freeSlot != 0xffffffff && context->readyForTestCount > 0) {
					startRayQuery(context, context->readyForTest[--context->readyForTestCount], freeSlot);
					continue;
				}
				
				if (spawnRays(context, thread, maxRaysPerSpawn, maxRaysInFlight))
					continue;
				
				if (freeSlot != 0xffffffff && context->waitingToBeFilledCount > 0) {
					startRayQuery(context, context->waitingToBeFilled[--context->waitingToBeFilledCount], freeSlot);
					continue;
				}
				
				wait(mutex);
			}
		}
		
		exit(mutex);
	}
	
	void gpuWorkerThread(void* memory) {
		using namespace racc_internal;
		using namespace racc;
		
		GpuWorkerData* data = static_cast<GpuWorkerData*>(memory);
		
		Context* context = data->context;
		Mutex& mutex = context->schedulingMutex;
		
		enter(mutex);
		
		unsigned threshold = context->configuration.allowCpuTracing ? context->configuration.cpuThreads : 0;
		unsigned rayStreamCount = context->rayStreamCount;
		
		for (;;) {
			if (context->shouldExit && !context->moreRaysExist && context->emptyCount == rayStreamCount)
				break;
			
			unsigned rayStreamIndex = ~0u;
			
			// Get a ray stream for testing.
			if (context->readyForTestCount) {
				++context->gpuThreadsTesting;
				rayStreamIndex = context->readyForTest[--context->readyForTestCount];
			}
			else if (context->waitingToBeFilledCount > threshold) {
				++context->gpuThreadsTesting;
				rayStreamIndex = context->waitingToBeFilled[--context->waitingToBeFilledCount];
			}
			else {
				wait(mutex);
				continue;
			}
			
			GpuRayStream* rayStream = context->rayStreams + rayStreamIndex;
			
			// Update stats.
			context->rayCount += rayStream->data.count;
			
			// Perform ray query.
			++context->gpuThreadsTesting;
			exit(mutex);
			{
				unsigned count = rayStream->data.count;
				
				size_t globalWorkSize[] = { (count + (RACC_WORK_GROUP_SIZE-1)) & (~(RACC_WORK_GROUP_SIZE-1)) };
				size_t localWorkSize[] = { RACC_WORK_GROUP_SIZE };
				
				cl_mem arg0 = rayStream->rays;
				cl_mem arg1 = context->currentScene->gpuNodes;
				cl_mem arg2 = context->currentScene->gpuTriangles;
				cl_mem arg3 = context->currentScene->gpuTriangleIndices;
				cl_mem arg4 = rayStream->results;
				cl_int arg5 = static_cast<cl_int>(count);
				cl_mem arg6 = context->currentEnvironment->gpuImage;
				
				cl_kernel kernel = data->kernel;
				
				clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&arg0);
				clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&arg1);
				clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&arg2);
				clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&arg3);
				clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&arg4);
				clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&arg5);
				clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&arg6);
				
				clEnqueueNDRangeKernel(data->queue, kernel, 1, 0, globalWorkSize, localWorkSize, 0, 0, 0);
				
				clFinish(data->queue);
			}
			enter(mutex);
			--context->gpuThreadsTesting;
			
			// Ready for shading.
			context->readyForShade[context->readyForShadeCount++] = rayStreamIndex;
			notifyAll(mutex);
		}
		
		exit(mutex);
	}
}

void racc::init() {
	// Disable denormals for performance.
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	
	rtcInit("isa=avx2,accel=bvh8.triangle4,builder=binned_sah2");
}

void racc::deinit() {
	rtcExit();
}

racc::Configuration racc::defaultConfiguration(cl_context gpuContext) {
	Configuration configuration = {};
	
	configuration.gpuContext = gpuContext;
	configuration.allowCpuTracing = true;
	configuration.cpuThreads = racc_internal::cpuCount();
	configuration.gpuSubmissionThreads = 4;
	configuration.maxRaysInFlight = 128*128*16;
	configuration.maxRaysPerSpawn = 128*128;
	configuration.cpuTestBatch = 1024;
	configuration.cpuShadeBatch = 8*1024;
	configuration.rayStreamBatchSize = 11*1024; // Target can have 8960 work items in flight.
	
	if (gpuContext && configuration.cpuThreads > 1)
		--configuration.cpuThreads; // Compensate for the GPU submission threads.
	
	return configuration;
}

racc::Context* racc::createContext(Configuration configuration) {
	using namespace racc_internal;
	
	cl_context gpuContext = configuration.gpuContext;
	
	assert(configuration.cpuThreads);
	assert(gpuContext || configuration.allowCpuTracing);
	assert(!gpuContext || configuration.gpuSubmissionThreads);
	
	unsigned gpuThreads = 0;
	unsigned cpuThreads = configuration.cpuThreads;
	
	cl_device_id gpuDevice = 0;
	cl_program gpuProgram = 0;
	
	if (gpuContext) {
		gpuThreads = configuration.gpuSubmissionThreads;
		
		// Select device.
		cl_device_id devices[1];
		size_t deviceCount = 0;
		
		clGetContextInfo(gpuContext, CL_CONTEXT_DEVICES, sizeof(devices), devices, &deviceCount);
		deviceCount /= sizeof(cl_device_id);
		
		if (!deviceCount) {
			fprintf(stderr, "RayAccelerator: Cannot get an OpenCL device from context.");
			return 0;
		}
		
		gpuDevice = devices[0]; // Pick the first device.
		
		// Compile kernel.
		const char* source = traversalKernel;
		gpuProgram = clCreateProgramWithSource(gpuContext, 1, &source, 0, 0);
		
		if (!gpuProgram) {
			fprintf(stderr, "RayAccelerator: Cannot create OpenCL program.");
			return 0;
		}
		
		const char flags[] = "-cl-mad-enable -cl-strict-aliasing -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math"
		" -DWORK_GROUP=" RACC_STRING_EXPANDED(RACC_WORK_GROUP_SIZE)
#ifdef _WIN32
		" -DWIN32=1"
#endif
		;
		
		cl_int result = clBuildProgram(gpuProgram, 0, 0, flags, 0, 0);
		
		if (result) {
			char log[10240] = "";
			clGetProgramBuildInfo(gpuProgram, gpuDevice, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
			
			fprintf(stderr, "RayAccelerator: OpenCL kernel build failed with error(s):\n%s\n", log);
			
			clReleaseProgram(gpuProgram);
			return 0;
		}
		else {
			char log[10240] = "";
			clGetProgramBuildInfo(gpuProgram, gpuDevice, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
			
			if (log[0])
				fprintf(stderr, "RayAccelerator: OpenCL kernel had warnings:\n%s\n", log);
		}
	}
	
	// Determine number and size of ray streams.
	unsigned maxRaysInFlight = configuration.maxRaysInFlight;
	unsigned maxRayStreamsInFlight = gpuThreads + cpuThreads*2;
	
	unsigned rayStreamSize = (unsigned)configuration.rayStreamBatchSize + (unsigned)(configuration.maxRaysPerSpawn > configuration.cpuShadeBatch ? configuration.maxRaysPerSpawn : configuration.cpuShadeBatch);
	unsigned rayStreamCount = maxRayStreamsInFlight + (maxRaysInFlight + (unsigned)configuration.rayStreamBatchSize-1) / configuration.rayStreamBatchSize;
	
	unsigned rayStreamTotalSize = 0;
	
	for (unsigned i = 0; i < rayStreamCount; ++i) {
		rayStreamTotalSize = (rayStreamTotalSize + 4095) & ~4095;
		rayStreamTotalSize += sizeof(Ray)*rayStreamSize;
		rayStreamTotalSize = (rayStreamTotalSize + 4095) & ~4095;
		rayStreamTotalSize += sizeof(Result)*rayStreamSize;
	}
	
	// Allocate memory as a single block.
	void* contextMemory = 0;
	void* threadMemory = 0;
	void* cpuWorkerDataMemory = 0;
	void* gpuWorkerDataMemory = 0;
	void* rayStreamMemory = 0;
	void* rayStreamDataMemory = 0;
	void* emptyStackMemory = 0;
	void* waitingToBeFilledStackMemory = 0;
	void* readyForTestStackMemory = 0;
	void* readyForShadeStackMemory = 0;
	void* cpuTestJobMemory = 0;
	
	unsigned threadCount = cpuThreads + gpuThreads;
	
	Allocation allocations[] = {
		{ sizeof(Context), 64, &contextMemory },
		{ (unsigned)sizeof(Thread)*threadCount, 16, &threadMemory },
		{ (unsigned)sizeof(CpuWorkerData)*cpuThreads, 64, &cpuWorkerDataMemory },
		{ (unsigned)sizeof(GpuWorkerData)*gpuThreads, 64, &gpuWorkerDataMemory },
		{ (unsigned)sizeof(GpuRayStream)*rayStreamCount, 64, &rayStreamMemory },
		{ (unsigned)sizeof(uint16_t)*rayStreamCount, 64, &emptyStackMemory },
		{ (unsigned)sizeof(uint16_t)*rayStreamCount, 64, &waitingToBeFilledStackMemory },
		{ (unsigned)sizeof(uint16_t)*rayStreamCount, 64, &readyForTestStackMemory },
		{ (unsigned)sizeof(uint16_t)*rayStreamCount, 64, &readyForShadeStackMemory },
		{ (unsigned)sizeof(CpuTestJob)*cpuThreads, 64, &cpuTestJobMemory },
		{ rayStreamTotalSize, 4096, &rayStreamDataMemory },
	};
	
	if (!allocateGroup(allocations, 4096)) {
		fprintf(stderr, "RayAccelerator: Unable to allocate memory.");
		
		if (gpuProgram)
			clReleaseProgram(gpuProgram);
		
		return 0;
	}
	
	// Initialize context.
	Context* context = static_cast<Context*>(contextMemory);
	
	context->configuration = configuration;
	context->gpuProgram = gpuProgram;
	
	context->rayStreams = static_cast<GpuRayStream*>(rayStreamMemory);
	context->rayStreamCount = rayStreamCount;
	context->rayStreamSize = rayStreamSize;
	
	context->threadCount = threadCount;
	context->threads = static_cast<Thread*>(threadMemory);
	
	context->gpuWorkerDataMemory = gpuWorkerDataMemory;
	
	init(context->schedulingMutex);
	
	context->shouldExit = 0;
	context->moreRaysExist = 0;
	
	context->empty = static_cast<uint16_t*>(emptyStackMemory);
	context->waitingToBeFilled = static_cast<uint16_t*>(waitingToBeFilledStackMemory);
	context->readyForTest = static_cast<uint16_t*>(readyForTestStackMemory);
	context->readyForShade = static_cast<uint16_t*>(readyForShadeStackMemory);
	
	context->emptyCount = 0;
	context->waitingToBeFilledCount = 0;
	context->readyForTestCount = 0;
	context->readyForShadeCount = 0;
	
	context->cpuTestJobs = static_cast<CpuTestJob*>(cpuTestJobMemory);
	
	context->cpuThreadsTesting = 0;
	context->gpuThreadsTesting = 0;
	
	context->rayCount = 0;
	context->raysInFlight = 0;
	
	for (unsigned i = 0; i < cpuThreads; ++i)
		context->cpuTestJobs[i].rayStream = 0xffff;
	
	context->emptyCount = rayStreamCount;
	
	for (unsigned i = 0; i < rayStreamCount; ++i)
		context->empty[i] = i;
	
	// Initialize ray streams.
	unsigned offset = 0;
	char* data = static_cast<char*>(rayStreamDataMemory);
	
	for (unsigned i = 0; i < rayStreamCount; ++i) {
		context->rayStreams[i].data.index = i;
		context->rayStreams[i].data.count = 0;
		
		offset = (offset + 4095) & ~4095;
		context->rayStreams[i].data.rays = reinterpret_cast<Ray*>(data + offset);
		offset += sizeof(Ray)*rayStreamSize;
		
		offset = (offset + 4095) & ~4095;
		context->rayStreams[i].data.results = reinterpret_cast<Result*>(data + offset);
		offset += sizeof(Result)*rayStreamSize;
	}
	
	assert(offset == rayStreamTotalSize);
	
	if (gpuContext) {
		for (unsigned i = 0; i < rayStreamCount; ++i) {
			unsigned raySize = sizeof(Ray)*rayStreamSize;
			unsigned resultSize = sizeof(Result)*rayStreamSize;
			
			raySize = (raySize + 4095) & ~4095;
			resultSize = (resultSize + 4095) & ~4095;
			
			context->rayStreams[i].rays = clCreateBuffer(gpuContext, CL_MEM_USE_HOST_PTR, raySize, context->rayStreams[i].data.rays, 0);
			context->rayStreams[i].results = clCreateBuffer(gpuContext, CL_MEM_USE_HOST_PTR, resultSize, context->rayStreams[i].data.results, 0);
		}
		
		// Clear all buffers to ensure that any remaining initialization is complete.
		const char* source = clearKernel;
		cl_program program = clCreateProgramWithSource(gpuContext, 1, &source, 0, 0);
		assert(program);
		
		cl_int result = clBuildProgram(program, 0, 0, "", 0, 0);
		assert(!result);
		
		cl_kernel kernel = clCreateKernel(program, "clearKernel", 0);
		cl_command_queue queue = clCreateCommandQueue(gpuContext, gpuDevice, 0, 0);
		
		for (unsigned i = 0; i < rayStreamCount; ++i) {
			unsigned workGroupSize = 8;
			
			// Clear rays.
			{
				unsigned count = rayStreamSize*2;
				
				size_t globalWorkSize[] = { (count + (workGroupSize - 1)) & (~(workGroupSize - 1)) };
				size_t localWorkSize[] = { workGroupSize };
				
				cl_mem arg0 = context->rayStreams[i].rays;
				clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&arg0);
				
				cl_int arg1 = static_cast<cl_int>(count);
				clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&arg1);
				
				clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalWorkSize, localWorkSize, 0, 0, 0);
			}
			
			// Clear results.
			{
				unsigned count = rayStreamSize;
				
				size_t globalWorkSize[] = { (count + (workGroupSize - 1)) & (~(workGroupSize - 1)) };
				size_t localWorkSize[] = { workGroupSize };
				
				cl_mem arg0 = context->rayStreams[i].results;
				clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&arg0);
				
				cl_int arg1 = static_cast<cl_int>(count);
				clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&arg1);
				
				clEnqueueNDRangeKernel(queue, kernel, 1, 0, globalWorkSize, localWorkSize, 0, 0, 0);
			}
		}
		
		clFinish(queue);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
	}
	
	// Initialize workers.
	CpuWorkerData* cpuWorkerData = static_cast<CpuWorkerData*>(cpuWorkerDataMemory);
	GpuWorkerData* gpuWorkerData = static_cast<GpuWorkerData*>(gpuWorkerDataMemory);
	
	for (unsigned i = 0; i < cpuThreads; ++i) {
		CpuWorkerData* data = cpuWorkerData + i;
		
		data->context = context;
		data->index = i;
	}
	
	for (unsigned i = 0; i < gpuThreads; ++i) {
		GpuWorkerData* data = gpuWorkerData + i;
		
		data->context = context;
		data->queue = clCreateCommandQueue(gpuContext, gpuDevice, 0, 0);
		data->kernel = clCreateKernel(gpuProgram, "traversal", 0);
	}
	
	// Start workers.
	for (unsigned i = 0; i < cpuThreads; ++i)
		context->threads[i] = createThread<cpuWorkerThread>(cpuWorkerData + i);
	
	for (unsigned i = 0; i < gpuThreads; ++i)
		context->threads[i + cpuThreads] = createThread<gpuWorkerThread>(gpuWorkerData + i);
	
	return context;
}

racc::ContextInfo racc::info(Context* context) {
	ContextInfo info = {};
	info.threadCount = context->configuration.cpuThreads;
	info.rayStreamCount = context->rayStreamCount;
	info.rayStreamSize = context->rayStreamSize;
	info.maxRaysInFlight = context->configuration.maxRaysInFlight;
	return info;
}

racc::Stats racc::render(Context* context, Scene* scene, Environment* environment, RenderCallbacks callbacks) {
	enter(context->schedulingMutex);
	
	context->currentScene = scene;
	context->currentEnvironment = environment;
	context->currentCallbacks = callbacks;
	
	context->moreRaysExist = 1;
	notifyAll(context->schedulingMutex);
	
	do {
		wait(context->schedulingMutex);
	}
	while (context->moreRaysExist || context->raysInFlight);
	
	exit(context->schedulingMutex);
	
	Stats stats = {};
	stats.raysTraced = context->rayCount;
	context->rayCount = 0;
	return stats;
}

void racc::destroy(Context* context) {
	using namespace racc_internal;
	
	enter(context->schedulingMutex);
	context->shouldExit = 1;
	notifyAll(context->schedulingMutex);
	exit(context->schedulingMutex);
	
	for (unsigned i = 0; i < context->threadCount; ++i)
		join(context->threads[i]);
	
	if (context->configuration.gpuContext) {
		GpuWorkerData* gpuWorkerData = static_cast<GpuWorkerData*>(context->gpuWorkerDataMemory);
		
		for (unsigned i = 0; i < context->configuration.gpuSubmissionThreads; ++i) {
			GpuWorkerData* data = gpuWorkerData + i;
			
			clReleaseKernel(data->kernel);
			clReleaseCommandQueue(data->queue);
		}
		
		clReleaseProgram(context->gpuProgram);
	}
	
	deinit(context->schedulingMutex);
	
	_mm_free(context);
}
