//
//  Threading.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-24.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_Threading_h
#define RayAccelerator_Threading_h

#include "RayAccelerator.h"

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <pthread.h>
#else
#error "Unsupported platform."
#endif

namespace racc_internal {
	struct Thread {
#ifdef _WIN32
		HANDLE handle;
#else
		pthread_t handle;
#endif
	};
	
	struct RACC_ALIGNED(64) Mutex {
#ifdef _WIN32
		CRITICAL_SECTION m;
		CONDITION_VARIABLE c;
#else
		pthread_mutex_t m;
		pthread_cond_t c;
#endif
	};
	
	template<void (*start)(void*)>
	inline Thread createThread(void* data);
	
	void join(Thread& thread);
	
	void init(Mutex& mutex);
	
	void deinit(Mutex& mutex);
	
	void enter(Mutex& mutex);
	
	void wait(Mutex& mutex);
	
	void notifyAll(Mutex& mutex);
	
	void exit(Mutex& mutex);
	
	// Returns the new value.
	int atomicIncrement(int* i);
	
	unsigned cpuCount();
}

namespace racc_internal {
#ifdef _WIN32
	template<void (*start)(void*)>
	inline DWORD WINAPI _threadPlatformStart(LPVOID ptr) {
		// Disable denormals for performance.
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		start(ptr);
		return 0;
	}
#else
	template<void (*start)(void*)>
	inline void* _threadPlatformStart(void* ptr) {
		// Disable denormals for performance.
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
		start(ptr);
		return 0;
	}
#endif
}

template<void (*start)(void*)>
inline racc_internal::Thread racc_internal::createThread(void* data) {
	Thread thread;
	
#ifdef _WIN32
	DWORD id;
	thread.handle = CreateThread(0, 0, _threadPlatformStart<start>, data, 0, &id);
#else
	pthread_create(&thread.handle, 0, _threadPlatformStart<start>, data);
#endif
	
	return thread;
}

#endif
