//
//  ThreadPool.cpp
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-24.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "ThreadPool.h"
#include "GroupAllocation.h"
#include "Threading.h"
#include <vector>
#include <new>

namespace {
	struct RACC_ALIGNED(64) WorkerThreadData {
		racc_internal::Thread thread;
		racc_internal::ThreadPool* pool;
		unsigned index;
	};
}

struct RACC_ALIGNED(64) racc_internal::ThreadPool {
	WorkerThreadData* threadData;
	unsigned threadCount;
	
	Mutex mutex;
	std::vector<ThreadPoolTask> tasks;
	unsigned tasksInFlight;
};

// Note: Thread function is passed to a template and cannot be static.
namespace {
	void workerThread(void* parameter) {
		using namespace racc_internal;
		
		WorkerThreadData* data = static_cast<WorkerThreadData*>(parameter);
		ThreadPool& pool = *data->pool;
		
		ThreadPoolTask task = {};
		
		for (;;) {
			enter(pool.mutex);
			
			--pool.tasksInFlight;
			
			if (pool.tasks.empty() && !pool.tasksInFlight)
				notifyAll(pool.mutex);
			
			for (;;) {
				if (pool.tasksInFlight == 0xffffffff) {
					exit(pool.mutex);
					return;
				}
				
				if (pool.tasks.size()) {
					task = pool.tasks.back();
					pool.tasks.pop_back();
					break;
				}
				
				wait(pool.mutex);
			}
			
			++pool.tasksInFlight;
			
			exit(pool.mutex);
			
			task.entry(task.data, data->index);
		}
	}
}

racc_internal::ThreadPool* racc_internal::createThreadPool(unsigned threadCount) {
	unsigned workerThreads = threadCount-1; // One thread is reserved for the main thread that executes "join".
	
	// Allocate memory as a single block.
	void* poolMemory = 0;
	void* threadDataMemory = 0;
	
	Allocation allocations[] = {
		{ sizeof(ThreadPool), 64, &poolMemory },
		{ (unsigned)sizeof(WorkerThreadData)*workerThreads, 64, &threadDataMemory },
	};
	
	if (!allocateGroup(allocations, 64))
		return 0;
	
	ThreadPool* pool = new (poolMemory) ThreadPool();
	
	pool->threadCount = threadCount;
	pool->tasksInFlight = workerThreads;
	pool->threadData = reinterpret_cast<WorkerThreadData*>(threadDataMemory);
	
	init(pool->mutex);
	
	for (unsigned i = 0; i < workerThreads; ++i) {
		pool->threadData[i].pool = pool;
		pool->threadData[i].index = (unsigned)i;
		pool->threadData[i].thread = createThread<workerThread>(pool->threadData + i);
	}
	
	return pool;
}

void racc_internal::join(ThreadPool* pool) {
	ThreadPoolTask task = {};
	
	enter(pool->mutex);
	
	while (!pool->tasks.empty() || pool->tasksInFlight) {
		if (!pool->tasks.empty()) {
			if (pool->tasksInFlight == 0xffffffff) {
				exit(pool->mutex);
				return;
			}
			
			task = pool->tasks.back();
			pool->tasks.pop_back();
			
			++pool->tasksInFlight;
			exit(pool->mutex);
			
			task.entry(task.data, pool->threadCount-1);
			
			enter(pool->mutex);
			--pool->tasksInFlight;
			
			if (pool->tasks.empty() && !pool->tasksInFlight)
				notifyAll(pool->mutex);
			
			continue;
		}
		
		wait(pool->mutex);
	}
	
	exit(pool->mutex);
}

unsigned racc_internal::threadCount(ThreadPool* pool) {
	return pool->threadCount;
}

void racc_internal::spawn(ThreadPool* pool, ThreadPoolTask task, unsigned count) {
	enter(pool->mutex);
	
	pool->tasks.resize(pool->tasks.size()+count, task);
	
	if (pool->tasks.size() == count)
		notifyAll(pool->mutex);
	
	exit(pool->mutex);
}

void racc_internal::spawnArray(ThreadPool* pool, const ThreadPoolTask* array, unsigned count) {
	enter(pool->mutex);
	
	pool->tasks.insert(pool->tasks.end(), array, array+count);
	
	if (pool->tasks.size() == count)
		notifyAll(pool->mutex);
	
	exit(pool->mutex);
}

void racc_internal::destroy(ThreadPool* pool) {
	join(pool);
	
	enter(pool->mutex);
	pool->tasksInFlight = 0xffffffff;
	notifyAll(pool->mutex);
	exit(pool->mutex);
	
	for (unsigned i = 0; i < pool->threadCount-1; ++i)
		racc_internal::join(pool->threadData[i].thread);
	
	deinit(pool->mutex);
	
	pool->~ThreadPool();
	_mm_free(pool);
}
