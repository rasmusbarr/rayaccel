//
//  ThreadPool.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2014-02-24.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_ThreadPool_h
#define RayAccelerator_ThreadPool_h

namespace racc_internal {
	struct ThreadPool;
	
	struct ThreadPoolTask {
		void* data;
		void (*entry)(void* data, unsigned thread);
	};
	
	ThreadPool* createThreadPool(unsigned threadCount);
	
	unsigned threadCount(ThreadPool* pool);
	
	void spawn(ThreadPool* pool, ThreadPoolTask task, unsigned count = 1);
	
	void spawnArray(ThreadPool* pool, const ThreadPoolTask* array, unsigned count);
	
	void join(ThreadPool* pool);
	
	void destroy(ThreadPool* pool);
}

#endif
