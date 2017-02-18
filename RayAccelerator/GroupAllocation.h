//
//  GroupAllocation.h
//  RayAccelerator
//
//  Created by Rasmus Barringer on 2017-02-14.
//  Copyright (c) 2017 Rasmus Barringer. All rights reserved.
//

#ifndef RayAccelerator_GroupAllocation_h
#define RayAccelerator_GroupAllocation_h

#include <immintrin.h>

namespace racc_internal {
	struct Allocation {
		unsigned size;
		unsigned align;
		void** pointer;
	};
	
	template<unsigned count>
	inline bool allocateGroup(Allocation(&allocations)[count], unsigned alignment) {
		unsigned allocationOffsets[count];
		unsigned size = 0;
		
		for (unsigned i = 0; i < count; ++i) {
			unsigned mask = allocations[i].align-1;
			
			size = (size + mask) & ~mask;
			allocationOffsets[i] = size;
			size += allocations[i].size;
		}
		
		unsigned mask = alignment-1;
		size = (size + mask) & ~mask;
		
		char* memory = static_cast<char*>(_mm_malloc(size, alignment));
		
		if (!memory)
			return false;
		
		for (unsigned i = 0; i < count; ++i)
			*allocations[i].pointer = memory + allocationOffsets[i];
		
		return true;
	}
}

#endif
