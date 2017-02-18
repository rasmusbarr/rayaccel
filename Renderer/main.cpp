//
//  main.cpp
//  Renderer
//
//  Created by Rasmus Barringer on 2014-02-06.
//  Copyright (c) 2014 Rasmus Barringer. All rights reserved.
//

#include "Camera.h"
#include "DisplayBuffer.h"
#include "PathTracingRenderer.h"
#include "WhittedRenderer.h"
#include "SceneData.h"
#include "Materials.h"

#ifdef _WIN32
#include <Windows.h>
#include <GLUT/freeglut_ext.h>

static uint64_t performanceFrequency;

static inline uint64_t microseconds() {
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return li.QuadPart *1000000ull / performanceFrequency;
}
#else
#include <CoreAudio/CoreAudio.h>

static inline uint64_t microseconds() {
	return AudioConvertHostTimeToNanos(AudioGetCurrentHostTime()) / 1000ull;
}
#endif

static const unsigned movingAverageFrames = 32;
static std::pair<uint64_t, uint64_t> movingAverage[movingAverageFrames];
static std::pair<uint64_t, uint64_t> movingAverageSum;
static unsigned movingAveragePosition = 0;

static DisplayBuffer* displayBuffer;

static unsigned cameraMovement = 0;
static float cameraSpeed = 1.0f;
static float3 cameraPanAxis = {};
static int mouseX;
static int mouseY;

static SceneData sceneData = {};
static Camera camera = {};

static TiledRenderer* renderer;
static unsigned spp = 0;

static racc::Context* context;
static racc::Scene* scene;
static racc::Environment* environment;

static bool spawn(void*, unsigned thread, racc::RayStream* output) {
	return renderer->spawnPrimary(thread, output);
}

static void shade(void*, unsigned thread, const racc::RayStream* input, unsigned start, unsigned end, racc::RayStream* output) {
	renderer->shade(thread, input, start, end, output);
}

static const racc::RenderCallbacks callbacks = { 0, spawn, shade };

inline cl_context createClContext() {
	cl_platform_id igpuPlatform = 0;
	cl_device_id igpuDevice = 0;
	
	cl_platform_id platforms[128];
	cl_uint platformCount;
	
	if (clGetPlatformIDs(sizeof(platforms)/sizeof(platforms[0]), platforms, &platformCount) != CL_SUCCESS) {
		printf("Failed to enumerate OpenCL devices.\n");
		return 0;
	}
	
	for (cl_uint i = 0; i < platformCount; ++i) {
		cl_device_id devices[128];
		cl_uint deviceCount = 0;
		
		if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, sizeof(devices)/sizeof(devices[0]), devices, &deviceCount) != CL_SUCCESS)
			continue;
		
		for (cl_uint j = 0; j < deviceCount; ++j) {
			char vendor[1024];
			
			if (clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor), vendor, 0) == CL_SUCCESS) {
				if (strstr(vendor, "Intel")) {
					igpuPlatform = platforms[i];
					igpuDevice = devices[j];
				}
			}
		}
	}
	
	if (!igpuPlatform || !igpuDevice) {
		printf("Failed to find Intel GPU.\n");
		return 0;
	}
	
	char name[1024];
	
	if (clGetDeviceInfo(igpuDevice, CL_DEVICE_NAME, sizeof(name), name, 0) == CL_SUCCESS)
		printf("OpenCL device: %s\n", name);
	
	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties)igpuPlatform,
		0
	};
	
	return clCreateContext(props, 1, &igpuDevice, 0, 0, 0);
}

static bool loadScene(const char* filename) {
	struct SceneHeader {
		uint32_t maxDepth;
		uint32_t vertexCount;
		uint32_t triangleCount;
		
		uint16_t viewportWidth;
		uint16_t viewportHeight;
		
		uint16_t environmentWidth;
		uint16_t environmentHeight;
		
		float3 origin;
		float3 dir;
		float3 up;
		float fov;
	} header;
	
	FILE* file = fopen(filename, "rb");
	
	if (!file)
		return false;
	
	
	if (!fread(&header, sizeof(header), 1, file))
		return false;
	
	sceneData.maxDepth = header.maxDepth;
	sceneData.vertexCount = header.vertexCount;
	sceneData.triangleCount = header.triangleCount;
	
	sceneData.viewportWidth = header.viewportWidth;
	sceneData.viewportHeight = header.viewportHeight;
	
	camera.lookAt(header.origin, header.dir, header.up, header.fov, 1e-3f, 1e+6f, header.viewportWidth, header.viewportHeight);
	
	sceneData.indices = static_cast<uint32_t*>(_mm_malloc(sceneData.triangleCount*3*sizeof(uint32_t), 64));
	sceneData.triangleMaterials = static_cast<uint16_t*>(_mm_malloc(sceneData.triangleCount*sizeof(uint16_t), 64));
	sceneData.triangleNormals = static_cast<float4*>(_mm_malloc(sceneData.triangleCount*sizeof(float4), 64));
	
	racc::Vertex* vertices = static_cast<racc::Vertex*>(_mm_malloc(sceneData.vertexCount*sizeof(float4), 64));
	sceneData.normals = static_cast<float4*>(_mm_malloc(sceneData.vertexCount*sizeof(float4), 64));
	sceneData.texcoords = static_cast<float2*>(_mm_malloc(sceneData.vertexCount*sizeof(float2), 64));
	
	racc::Color* environmentPixels = static_cast<racc::Color*>(_mm_malloc((unsigned)header.environmentWidth*(unsigned)header.environmentHeight*sizeof(racc::Color), 64));
	
	sceneData.materials = new Material*[4];
	
	sceneData.materials[0] = new (_mm_malloc(sizeof(ReflectiveDiffuseMaterial), 64)) ReflectiveDiffuseMaterial(make_float3(0.8f), 1.0f/1.4f);
	sceneData.materials[1] = new (_mm_malloc(sizeof(ReflectiveDiffuseMaterial), 64)) ReflectiveDiffuseMaterial(make_float3(0.1f), 1.0f/1.4f);
	sceneData.materials[2] = new (_mm_malloc(sizeof(ReflectiveDiffuseMaterial), 64)) ReflectiveDiffuseMaterial(make_float3(0.6f), 1.0f/1.2f);
	sceneData.materials[3] = new (_mm_malloc(sizeof(ReflectiveDiffuseMaterial), 64)) ReflectiveDiffuseMaterial(make_float3(0.3f), 1.0f/1.2f);
	
	bool success = true;
	
	success &= fread(sceneData.indices, sceneData.triangleCount*3*sizeof(uint32_t), 1, file) != 0;
	success &= fread(sceneData.triangleMaterials, sceneData.triangleCount*sizeof(uint16_t), 1, file) != 0;
	success &= fread(sceneData.triangleNormals, sceneData.triangleCount*sizeof(float4), 1, file) != 0;
	
	success &= fread(vertices, sceneData.vertexCount*sizeof(racc::Vertex), 1, file) != 0;
	success &= fread(sceneData.normals, sceneData.vertexCount*sizeof(float4), 1, file) != 0;
	success &= fread(sceneData.texcoords, sceneData.vertexCount*sizeof(float2), 1, file) != 0;
	
	success &= fread(environmentPixels, (unsigned)header.environmentWidth*(unsigned)header.environmentHeight*sizeof(racc::Color), 1, file) != 0;
	
	fclose(file);
	
	if (!success)
		return false;
	
	scene = racc::createScene(context, vertices, sceneData.vertexCount, sceneData.indices, sceneData.triangleCount*3);
	environment = racc::createEnvironment(context, environmentPixels, header.environmentWidth, header.environmentHeight);
	
	return true;
}

static void render() {
	if (cameraMovement) {
		if (cameraMovement & 1)
			camera.origin += camera.forward() * cameraSpeed;
		if (cameraMovement & 4)
			camera.origin -= camera.forward() * cameraSpeed;
		if (cameraMovement & 2)
			camera.origin -= normalize(camera.right) * cameraSpeed;
		if (cameraMovement & 8)
			camera.origin += normalize(camera.right) * cameraSpeed;
		
		spp = 0;
		renderer->clear();
	}
	
	uint64_t start = microseconds();
	racc::Stats stats = racc::render(context, scene, environment, callbacks);
	renderer->endFrame();
	uint64_t end = microseconds();
	
	++spp;
	
	uint64_t rays = stats.raysTraced;
	uint64_t time = end - start;
	uint64_t rays2 = rays;
	uint64_t time2 = time;
	
	movingAverageSum.first -= movingAverage[movingAveragePosition].first;
	movingAverageSum.second -= movingAverage[movingAveragePosition].second;
	
	movingAverage[movingAveragePosition] = std::make_pair(rays, time);
	movingAveragePosition = (movingAveragePosition + 1) % movingAverageFrames;
	
	movingAverageSum.first = (rays += movingAverageSum.first);
	movingAverageSum.second = (time += movingAverageSum.second);
	
	double mrps = (double)rays / (double)time;
	
	printf("%5.1f mrps (instant) %5.1f mrps (sliding)\n", (float)((double)rays2/(double)time2), (float)mrps);
	
	displayBuffer->present(renderer->frameBuffer, spp);
	
	glutSwapBuffers();
	glutPostRedisplay();
}

static void mouse(int button, int state, int x, int y) {
	mouseX = x;
	mouseY = y;
}

static void motion(int x, int y) {
	int dx = x - mouseX;
	int dy = y - mouseY;
	
	spp = 0;
	camera.rotate(-dx * 0.002f, cameraPanAxis);
	camera.rotate(dy * 0.002f, normalize(camera.right));
	renderer->clear();
	
	mouseX = x;
	mouseY = y;
}

static void keyboardDown(unsigned char key, int, int) {
	if (key == 'w')
		cameraMovement |= 1;
	else if (key == 'a')
		cameraMovement |= 2;
	else if (key == 's')
		cameraMovement |= 4;
	else if (key == 'd')
		cameraMovement |= 8;
	
	if (key == 'r')
		cameraSpeed *= 1.5f;
	else if (key == 'f')
		cameraSpeed *= 1.0f/1.5f;
}

static void keyboardUp(unsigned char key, int, int) {
	if (key == 'w')
		cameraMovement &= ~1;
	else if (key == 'a')
		cameraMovement &= ~2;
	else if (key == 's')
		cameraMovement &= ~4;
	else if (key == 'd')
		cameraMovement &= ~8;
}

int main(int argc, const char* argv[]) {
	bool whitted = false;
	bool disableGpu = false;
	bool disableCpuTracing = false;
	
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], "--whitted") == 0) {
			whitted = true;
		}
		else if (strcmp(argv[i], "--no-gpu") == 0) {
			disableGpu = true;
		}
		else if (strcmp(argv[i], "--no-cpu-tracing") == 0) {
			disableCpuTracing = true;
		}
		else {
			printf("Unknown argument: %s\n", argv[i]);
		}
	}
	
	if (disableGpu && disableCpuTracing) {
		printf("Conflicting arguments: --no-gpu and --no-cpu-tracing are both used.\n");
		return 1;
	}
	
	glutInit(&argc, const_cast<char**>(argv));
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	
	cl_context gpuContext = 0;
	
	if (!disableGpu) {
		gpuContext = createClContext();
		
		if (!gpuContext) {
			printf("Failed to create OpenCL context for Intel GPU. You can try running with --no-gpu.\n");
			return 1;
		}
	}
	
	racc::init();
	racc::Configuration raConfig = racc::defaultConfiguration(gpuContext);
	
	if (disableCpuTracing)
		raConfig.allowCpuTracing = false;
	
	context = racc::createContext(raConfig);
	
	if (!loadScene("../battlefield.bin")) {
		printf("Unable to load ../battlefield.bin. Please make sure that the working directory set to the platform folder.\n");
		return 1;
	}
	
	float maxUpAxis = fmaxf(fabsf(camera.up.x), fmaxf(fabsf(camera.up.y), fabsf(camera.up.z)));
	
	if (fabsf(camera.up.x) == maxUpAxis)
		cameraPanAxis = make_float3(copysignf(1.0f, camera.up.x), 0.0f, 0.0f);
	else if (fabsf(camera.up.y) == maxUpAxis)
		cameraPanAxis = make_float3(0.0f, copysignf(1.0f, camera.up.y), 0.0f);
	else
		cameraPanAxis = make_float3(0.0f, 0.0f, copysignf(1.0f, camera.up.z));
	
	if (whitted) {
		sceneData.maxDepth = 8;
		renderer = new (_mm_malloc(sizeof(WhittedRenderer), 64)) WhittedRenderer(context, camera, sceneData);
	}
	else {
		renderer = new (_mm_malloc(sizeof(PathTracingRenderer), 64)) PathTracingRenderer(context, camera, sceneData);
	}
	
	glutInitWindowSize(sceneData.viewportWidth, sceneData.viewportHeight);
	glutCreateWindow("RayAccelerator");
	
	glutDisplayFunc(render);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	
	displayBuffer = new DisplayBuffer(sceneData.viewportWidth, sceneData.viewportHeight);
	
#ifdef _WIN32
	// Query performance frequency.
	LARGE_INTEGER li;
	
	if (!QueryPerformanceFrequency(&li))
		return 1;
	
	performanceFrequency = li.QuadPart;
	
	// Disable v-sync.
	BOOL (APIENTRY*swapInterval)(int) = (BOOL (APIENTRY*)(int))wglGetProcAddress("wglSwapIntervalEXT");
	
	if (swapInterval)
		swapInterval(0);
	
	glutCloseFunc([] () {
		exit(0);
	});
#endif
	
	glutMainLoop();
	return 0;
}
