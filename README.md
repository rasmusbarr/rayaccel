# Ray Accelerator

Ray Accelerator allows the CPU cores and the integrated GPU in a modern Intel CPU to work together for efficient ray traced rendering. The CPU takes care of shading in a vectorized manner and the GPU is used as an accelerator for intersection testing rays against the scene. Intel's Embree is used to do additional intersection testing on the CPU if there is time.

All source code for the Ray Accelerator system is present in the [RayAccelerator](RayAccelerator) folder. The API is exposed in [RayAccelerator.h](RayAccelerator/RayAccelerator.h).

The system requires an Intel CPU with AVX2 support and an integrated GPU with shared memory (Haswell and later). It has been specifically tailored for Intel Haswell Core i7 4960HQ.

# Example renderer

![screenshot](https://cloud.githubusercontent.com/assets/15862826/23090653/5b62f64c-f5a3-11e6-9dc9-af4c7d79c2d7.png)

The example renderer shows how to use Ray Accelerator to build a renderer, both a path tracer and a Whitted-style ray tracer.

[Windows](Win) and [Mac](Mac) are supported, with Visual Studio and Xcode projects.

Source code for the example renderer resides in the [Renderer](Renderer) folder.

# Paper

Details can be found in the published paper:

- Author generated version (coming soon)
- [CGF version](http://onlinelibrary.wiley.com/doi/10.1111/cgf.13071/full)
