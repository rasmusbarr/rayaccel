//
//  Threading.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-24.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Threading.h"

#if __APPLE__
#include <unistd.h>
#include <libkern/OSAtomic.h>
#endif

void racc_internal::join(Thread& thread) {
#ifdef _WIN32
	WaitForSingleObject(thread.handle, INFINITE);
	CloseHandle(thread.handle);
#else
	void* result;
	pthread_join(thread.handle, &result);
#endif
}

void racc_internal::init(Mutex& mutex) {
#ifdef _WIN32
	InitializeCriticalSection(&mutex.m);
	InitializeConditionVariable(&mutex.c);
#else
	pthread_mutex_init(&mutex.m, 0);
	pthread_cond_init(&mutex.c, 0);
#endif
}

void racc_internal::deinit(Mutex& mutex) {
#ifdef _WIN32
	DeleteCriticalSection(&mutex.m);
#else
	pthread_mutex_destroy(&mutex.m);
	pthread_cond_destroy(&mutex.c);
#endif
}

void racc_internal::enter(Mutex& mutex) {
#ifdef _WIN32
	EnterCriticalSection(&mutex.m);
#else
	pthread_mutex_lock(&mutex.m);
#endif
}

void racc_internal::wait(Mutex& mutex) {
#ifdef _WIN32
	SleepConditionVariableCS(&mutex.c, &mutex.m, INFINITE);
#else
	pthread_cond_wait(&mutex.c, &mutex.m);
#endif
}

void racc_internal::notifyAll(Mutex& mutex) {
#ifdef _WIN32
	WakeAllConditionVariable(&mutex.c);
#else
	pthread_cond_broadcast(&mutex.c);
#endif
}

void racc_internal::exit(Mutex& mutex) {
#ifdef _WIN32
	LeaveCriticalSection(&mutex.m);
#else
	pthread_mutex_unlock(&mutex.m);
#endif
}

int racc_internal::atomicIncrement(int* i) {
#ifdef _WIN32
	return InterlockedIncrement(reinterpret_cast<volatile LONG*>(i));
#else
	return OSAtomicIncrement32(reinterpret_cast<volatile int*>(i));
#endif
}

unsigned racc_internal::cpuCount() {
#ifdef _WIN32
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return info.dwNumberOfProcessors;
#else
	static unsigned cpuCount = 0;
	
	if (!cpuCount)
		cpuCount = (unsigned)sysconf(_SC_NPROCESSORS_ONLN);
	
	return cpuCount;
#endif
}
