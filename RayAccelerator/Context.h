//
//  Context.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-12.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_Context_h
#define RayAccelerator_Context_h

#include "RayAccelerator.h"
#include "Threading.h"

namespace racc_internal {
	struct GpuRayStream;
	struct CpuTestJob;
}

namespace racc {
	struct Context {
		Configuration configuration;
		cl_program gpuProgram;
		
		racc_internal::GpuRayStream* rayStreams;
		uint32_t rayStreamCount;
		uint32_t rayStreamSize;
		
		uint16_t* empty;
		uint16_t* waitingToBeFilled;
		uint16_t* readyForTest;
		uint16_t* readyForShade;
		
		racc_internal::CpuTestJob* cpuTestJobs;
		
		Scene* currentScene;
		Environment* currentEnvironment;
		RenderCallbacks currentCallbacks;
		
		uint32_t threadCount;
		racc_internal::Thread* threads;
		void* gpuWorkerDataMemory;
		
		// Data below may be changed by multiple threads and are stored at a new cacheline boundary to avoid false sharing (mutex is aligned).
		racc_internal::Mutex schedulingMutex;
		
		uint8_t shouldExit;
		uint8_t moreRaysExist;
		uint8_t cpuThreadsTesting;
		uint8_t gpuThreadsTesting;
		
		uint16_t emptyCount;
		uint16_t waitingToBeFilledCount;
		uint16_t readyForTestCount;
		uint16_t readyForShadeCount;
		
		uint32_t raysInFlight;
		uint64_t rayCount;
	};
}

#endif
