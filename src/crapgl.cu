/*
 ============================================================================
 Name        : cudagl.cu
 Author      : Andrey Akhmetov
 Version     :
 Copyright   : May be reproduced, modified, used with attribution. NO WARRANTY IS GIVEN FOR THIS CODE. Educational intent only, not designed for production systems.
 Description : CUDA renderer
 ============================================================================
 WARNING: This is done for self-education purposes only. This code probably sucks in both design and implementation.
 Use only if you are confident in your ability to tolerate visual noise and error messages.
 */
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <memory>
#include <sstream>
#ifndef CRAPGL_CU
#define CRAPGL_CU
static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#ifndef NO_CHECK_CUDA
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#else
#define CUDA_CHECK_RETURN(value) value
#endif
#ifndef __CUDA_ARCH__
#define DIVIDE_INTRINSIC(x, y) ((x)/(y))
#else
#define DIVIDE_INTRINSIC(x, y) __fdividef((x), (y))
#endif
struct __align__(16) vec4 {
	float w;
	float x;
	float y;
	float z;
	vec4(float w = 0.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) :
			w(w), x(x), y(y), z(z) {
	}
	;

	vec4 operator-(const vec4& a) {
		return vec4(w - a.w, x - a.x, y - a.y, z - a.z);
	}

	vec4 operator+(const vec4& a) {
		return vec4(a.w + w, a.x + x, a.y + y, a.z + z);
	}
	float operator*(const vec4& a) {
		return (a.w * w) + (a.x * x) + (a.y * y) + (a.z * z);
	}

};

float dotXY(const vec4& a, const vec4& b) {
#ifdef __CUDA_ARCH__
	return fmaf(a.x, b.x, a.y * b.y);
#else
	return (a.x*b.x+a.y*b.y);
#endif
}
__host__ __device__ int roundUp32(int i) {
	return ((i - 1) / 32 + 1) * 32;
}

__host__ __device__ void reproject(vec4 &vec) {
	float w = vec.w;
	vec.x = vec.x / w;
	vec.y = vec.y / w;
	vec.z = vec.z / 2;
	vec.w = 1.0f;
}
__device__ unsigned int depthFToUInt(float depth) {
	return (unsigned int) UINT_MAX * saturate(depth);
}
// from https://devblogs.nvidia.com/parallelforall/lerp-faster-cuda/
template<typename T>
__host__         __device__
        inline T lerp(T v0, T v1, T t) {
	return fmaf(t, v1, fma(-t, v0, v0));
}
// modified from Arduino
float map(float x, float in_min, float in_max, float out_min, float out_max) {
	return fmaf((x - in_min),
			DIVIDE_INTRINSIC((out_max - out_min), (in_max - in_min)), out_min);
}

__device__ float blerp(float v00, float v10, float v01, float u, float v) {
	return lerp(v00, v10, u) + lerp(0.0f, v01 - v00, v);
}

class CrapGlException: public std::exception {
private:
	std::string message_;
public:

	CrapGlException(const std::string& message) :
			message_(message) {
	}
	virtual const char* what() const throw () {
		return message_.c_str();
	}
	virtual ~CrapGlException() throw () {
	}
	;
};

void* cudaHostMallocHelper(size_t size) {
	void* ptr;
	CUDA_CHECK_RETURN(cudaMallocHost(&ptr, size));
	return ptr;
}

void* cudaMallocHelper(size_t size) {
	void* ptr;
	CUDA_CHECK_RETURN(cudaMalloc(&ptr, size));
	return ptr;
}

size_t computeStride(short desiredStride) {
	// currently not trying to align anything.
	return (size_t) desiredStride;
}
size_t computeSize(size_t stride, short capacity) {
	// pending revision
	return stride * capacity;
}

/**
 * Struct identifying a VBO. Stride is number of bytes per vertex.
 * Capacity and cound are number of vertices, index capacity and count are number of indices.
 * cpu_XXX pointers are pointers in RAM
 * gpu_XXX pointers are in CUDA memory
 */
class Vbo {
	friend class ShaderPipeline;
public:
	Vbo(short vertexSize, short capacity, short indexCapacity) :
			stride(computeStride(vertexSize)), vtxsize(vertexSize), capacity(
					capacity), count(0), index_capacity(capacity), index_count(
					0), cpu_data(
					cudaHostMallocHelper(
							computeSize(computeStride(vertexSize), capacity))), cpu_index_data(
					(short*) cudaHostMallocHelper(
							computeIndexSize(indexCapacity))), gpu_data(
					cudaMallocHelper(
							computeSize(computeStride(vertexSize), capacity))), gpu_index_data(
					(short*) cudaMallocHelper(computeIndexSize(indexCapacity))), size(
					computeSize(computeStride(vertexSize), capacity)), index_size(
					computeIndexSize(indexCapacity)), dirty(false), indices_dirty(
					false) {
	}
	~Vbo() {
		CUDA_CHECK_RETURN(cudaFreeHost(cpu_data));
		CUDA_CHECK_RETURN(cudaFreeHost(cpu_index_data));
		CUDA_CHECK_RETURN(cudaFree(gpu_data));
		CUDA_CHECK_RETURN(cudaFree(gpu_index_data));
	}
	void markDirty() {
		dirty = true;
	}
	void markIndicesDirty() {
		indices_dirty = true;
	}

	void* getDataBuffer() {
		markDirty();
		return cpu_data;
	}
	short* getIndexBuffer() {
		markIndicesDirty();
		return cpu_index_data;
	}
	void updateBuffers() {
		if (dirty) {
			dirty = false;
			CUDA_CHECK_RETURN(
					cudaMemcpy(gpu_data, cpu_data, size,
							cudaMemcpyHostToDevice));
		}
		if (indices_dirty) {
			indices_dirty = false;
			CUDA_CHECK_RETURN(
					cudaMemcpy(gpu_index_data, cpu_index_data, index_size,
							cudaMemcpyHostToDevice));
		}
	}

	short getCount() const {
		return count;
	}

	void setCount(short count) {
		this->count = count;
	}

	short getIndexCount() const {
		return index_count;
	}

	void setIndexCount(short indexCount) {
		index_count = indexCount;
	}

private:

	size_t computeIndexSize(short index_capacity) {
		return sizeof(short) * index_capacity;
	}

	const size_t stride;
	const size_t vtxsize;
	const short capacity;
	short count;
	const short index_capacity;
	short index_count;
	// const void* is pointer to const data, while void* const is a constant pointer to mutable data
	void* const cpu_data;
	void* const gpu_data;
	short* const cpu_index_data;
	short* const gpu_index_data;
	const size_t size;
	const size_t index_size;
	bool dirty;
	bool indices_dirty;
};

// vtxid, vtx_in, position_out, vtx_out, uniforms
typedef void (*vertexShader_t)(short vtxid, void* vtx_in, vec4& position_out,
		float* vtx_out, void* uniforms);
// true if valid frag, false if discard
typedef bool (*fragmentShader_t)(vec4 position_in, float* vtx_in,
		unsigned int& color_out, void* uniforms);
typedef bool (*depthTest_t)(unsigned int, unsigned int*);

namespace depth_test {
__device__ bool less(unsigned int fragDepth, unsigned int* prevDepth) {
	unsigned int old = atomicMin(prevDepth, fragDepth);
	return fragDepth < old;
}
__device__ bool lessEqual(unsigned int fragDepth, unsigned int* prevDepth) {
	unsigned int old = atomicMin(prevDepth, fragDepth);
	return fragDepth <= old;
}
__device__ bool greater(unsigned int fragDepth, unsigned int* prevDepth) {
	unsigned int old = atomicMax(prevDepth, fragDepth);
	return fragDepth > old;
}
__device__ bool greaterEqual(unsigned int fragDepth, unsigned int* prevDepth) {
	unsigned int old = atomicMax(prevDepth, fragDepth);
	return fragDepth >= old;
}
}

// sizes in bytes
class VtxShaderDesc {
	friend class ShaderPipeline;
public:
	VtxShaderDesc(size_t inSize, size_t outSize, size_t uniformsSize,
			vertexShader_t kern) :
			vtxin_size(inSize), vtxout_size(outSize), uniforms_size(
					uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(vertexShader_t), 0,
				cudaMemcpyDeviceToHost);
	}
private:
	const size_t vtxin_size;
	const size_t vtxout_size;
	const size_t uniforms_size;
	vertexShader_t kern;
};

class FragShaderDesc {
	friend class ShaderPipeline;
	FragShaderDesc(size_t inSize, size_t uniformsSize, fragmentShader_t kern) :
			vtxin_size(inSize), uniforms_size(uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(fragmentShader_t), 0,
				cudaMemcpyDeviceToHost);
	}
private:
	const size_t vtxin_size;
	const size_t uniforms_size;
	fragmentShader_t kern;
};

// these live on the GPU! No CPU copy available (for now)
class TransformedVertexBuffer {
	friend class ShaderPipeline;
public:
	TransformedVertexBuffer(short capacity, size_t desiredStride) :
			vtx_count(0), vtx_capacity(capacity), stride(
					computeStride(desiredStride)), gpu_data(
					cudaMallocHelper(
							computeSize(computeStride(desiredStride),
									capacity))), gpu_vec_data(
					(vec4*) cudaMallocHelper(capacity * sizeof(vec4))) {
	}

	~TransformedVertexBuffer() {
		CUDA_CHECK_RETURN(cudaFree(gpu_data));
		CUDA_CHECK_RETURN(cudaFree(gpu_vec_data));
	}
private:
	short vtx_count;
	const short vtx_capacity;
	const size_t stride;
	void* const gpu_data;
	vec4* const gpu_vec_data;
};

class DepthBuffer {
public:
	DepthBuffer(int width, int height) {
		width = roundUp32(width);
		curSize = width * height;
		buffer = (unsigned int*) cudaMallocHelper(
				curSize * sizeof(unsigned int));
		curWidth = width;
		curHeight = height;
	}
	void reshape(int width, int height) {
		width = roundUp32(width);
		int newSize = width * height;
		if (newSize > curSize) {
			CUDA_CHECK_RETURN(cudaFree((void* ) buffer));
			buffer = (unsigned int*) cudaMallocHelper(
					newSize * sizeof(unsigned int));
			curSize = newSize;
		}
		curWidth = width;
		curHeight = height;
	}
private:
	unsigned int* buffer;
	int curSize;
	int curWidth;
	int curHeight;
};

enum FaceCulling {
	front, back, none
};

__global__ void crapGlClear(int width, int height, bool clearColor,
		bool clearDepth, int color, int depth, cudaSurfaceObject_t surf,
		int* depthBuf) {
	int depthBufferPitch = roundUp32(width);
	for (int y = blockIdx.x; y < height; y += gridDim.x) {
		for (int x = threadIdx.x; x < width; x += (blockDim.x)) {
			if (clearColor)
				surf2Dwrite(color, surf, x * sizeof(int), y);
			if (clearDepth)
				depthBuf[y * depthBufferPitch + x] = depth;
		}
	}

}

// Long-term TODO:
// Have the TVB track which indices have been computed or not, and don't run the vertex
// shader for vertices that have already been processed by index.
template<vertexShader_t shader>
__global__ void runVertexShaderIndexed(void* vtxin, short index_count,
		short* indices, size_t vtxin_stride, void* uniforms, float* vtxout,
		size_t vtxout_stride, void* positions) {
// We don't care about blocks; no shared memory. This may change when the long-term TODO is implemented
	int globId = threadIdx.x + (blockIdx.x * blockDim.x);
	for (int idx = globId; idx < index_count; idx += blockDim.x * gridDim.x) {
		//(short vtxid, void* vtx_in, vec4* position_out,
		//void* vtx_out, void* uniforms);
		short vboidx = indices[idx];
		shader(vboidx, (void*) ((char*) vtxin + (vboidx * vtxin_stride)),
				((vec4*) positions)[idx],
				(float*) ((float*) vtxout + (idx * vtxout_stride)), uniforms);
	}
}

template<vertexShader_t shader>
__global__ void runVertexShaderUnindexed(void* vtxin, short vertex_count,
		size_t vtxin_stride, void* uniforms, float* vtxout,
		size_t vtxout_stride, void* positions) {
	// We don't care about blocks; no shared memory.
	int globId = threadIdx.x + (blockIdx.x * blockDim.x);
	for (int idx = globId; idx < vertex_count; idx += blockDim.x * gridDim.x) {
		//(short vtxid, void* vtx_in, vec4* position_out,
		//void* vtx_out, void* uniforms);
		shader(idx, (void*) ((char*) vtxin + (idx * vtxin_stride)),
				((vec4*) positions)[idx], &vtxout[idx * vtxout_stride],
				uniforms);
	}
}

__host__ __device__ int windowSpaceToPixel(float coord, int pixels) {
	return (int) (((0.5 * coord + 0.5) * (pixels - 1))); // figure out if width-1 is actually what I want
}

__host__ __device__ int pixelToWindowSpace(int coord, int pixels) {
	return DIVIDE_INTRINSIC(coord, pixels-1) * 2 - 1; // again, do I want pixels-1 here?
}

template<fragmentShader_t shader, bool depthTest, bool earlyDepthTest,
		depthTest_t depthFunc, int chunkHeight, size_t vtxin_stride>
__global__ void runFragmentShader(cudaSurfaceObject_t surf, float* vtxin,
		vec4* vecData, short vertex_count, int width, int height,
		unsigned int* depthBuffer, void* uniforms) {
	// each block works on a separate triangle. Let's take the naive approach for now and profile/optimize later

	// warps should try to work on small areas together...
	int threadXOffset = threadIdx.x / chunkHeight;
	int threadYOffset = threadIdx.x % chunkHeight;
	int threadXStride = blockDim.x / chunkHeight;
	int threadYStride = chunkHeight; // for lack of better calculation
	float vtxin_temp[vtxin_stride];
	for (int i = blockIdx.x; i < (vertex_count) / 3; i += gridDim.x) {

		int depthBufferPitch = roundUp32(width);
		vec4 v0 = vecData[i * 3];
		vec4 v1 = vecData[i * 3 + 1];
		vec4 v2 = vecData[i * 3 + 2];
		reproject(v0);
		reproject(v1);
		reproject(v2);
		float xMin = min(min(v0.x, v1.x), v2.x);
		float xMax = max(max(v0.x, v1.x), v2.x);
		float yMin = min(min(v0.y, v1.y), v2.y);
		float yMax = max(max(v0.y, v1.y), v2.y);
		// map these to pixel coordinates, clamp to destination surface size
		int xMinScreen = max(windowSpaceToPixel(xMin, width), 0);
		int xMaxScreen = min(windowSpaceToPixel(xMax, width), width - 1);
		int yMinScreen = max(windowSpaceToPixel(yMin, height), 0);
		int yMaxScreen = min(windowSpaceToPixel(yMax, height), height - 1);
		// should have a different path for things that risk diverging madly due to very small X or Y
		// All the threads in a warp are working on the same triangle
		// This can be taken care of later though
		for (int x = xMinScreen + threadXOffset; x <= xMaxScreen; x +=
				threadXStride) {
			for (int y = yMinScreen + threadYOffset; y <= yMaxScreen; y +=
					threadYStride) {
				// first opportunity for divergence; the warp covering a small block should help here.
				// from http://blackpawn.com/texts/pointinpoly/
				vec4 frag = vec4(1.0f, pixelToWindowSpace(x, width),
						pixelToWindowSpace(y, height), 0); // W is 1 from reprojection. No idea what Z is yet; we'll interpolate it later if we must.
				vec4 d0 = v2 - v0;
				vec4 d1 = v1 - v0;
				vec4 d2 = frag - v0;
				float dot00 = dotXY(d0, d0);
				float dot01 = dotXY(d0, d1);
				float dot02 = dotXY(d0, d2);
				float dot11 = dotXY(d1, d1);
				float dot12 = dotXY(d1, d2);
				float invDenom = DIVIDE_INTRINSIC(1.0f,
						fmaf(dot00, dot11, -dot01 * dot01));
				float u = fmaf(dot11, dot02, -dot01 * dot12) * invDenom;
				float v = fmaf(dot00, dot12, -dot01 * dot02) * invDenom;
				if (u >= 0 && v >= 0 && u + v <= 1) {
					// in triangle!
					// Time to interpolate Z
					// U in terms of v2-v0
					// V in terms of v1-v0

					frag.z = blerp(v0.z, v2.z, v1.z, u, v);
					bool passedDepthTest = frag.z >= 0 && frag.z <= 1;
					if (depthTest && earlyDepthTest) {
						passedDepthTest = depthFunc(depthFToUInt(frag.z),
								&depthBuffer[y * depthBufferPitch + x]);
					}
					if (passedDepthTest) {
						unsigned int color;
						// If this isn't in L1 cache, it can be refactored to be manually cached in shared or lcl mem.
						for (int vtxDatum = 0; vtxDatum < vtxin_stride;
								vtxDatum++) {
							vtxin_temp[vtxDatum] =
									blerp(
											vtxin[i * 3 * vtxin_stride
													+ vtxDatum],
											vtxin[(i * 3 + 2) * vtxin_stride
													+ vtxDatum],
											vtxin[(i * 3 + 3) * vtxin_stride
													+ vtxDatum], u, v);
						}
						bool validFrag = shader(frag, &vtxin_temp, color,
								uniforms);
						if (depthTest & !earlyDepthTest & validFrag) {
							validFrag &= depthFunc(depthFToUInt(frag.z),
									&depthBuffer[y * depthBufferPitch + x]);
						}
						if (validFrag) {
							surf2Dwrite(color, surf, x * sizeof(unsigned int),
									y);
						}
					}
				}
			}
		}
	}
}

class ShaderPipeline {
public:
	ShaderPipeline(Vbo* vbo, VtxShaderDesc* vert, FragShaderDesc* frag) :
			vbo_(vbo), vert_(vert), frag_(frag), vert_dirty(false), frag_dirty(
					false) {
		if (vert->vtxin_size != vbo->vtxsize) {
			throw CrapGlException(
					"Mismatch between VBO vertex size and vertex shader input size.");
		}
		if (vert->vtxout_size != frag->vtxin_size) {
			throw CrapGlException(
					"Mismatch between vertex shader output size and fragment shader input size.");
		}
		tvb_ =
				std::unique_ptr < TransformedVertexBuffer
						> (new TransformedVertexBuffer(vbo->capacity,
								vert->vtxin_size));
		vert_uniforms = cudaHostMallocHelper(vert_->uniforms_size);
		frag_uniforms = cudaHostMallocHelper(frag_->uniforms_size);
		vert_uniforms_gpu = cudaMallocHelper(vert_->uniforms_size);
		frag_uniforms_gpu = cudaMallocHelper(frag_->uniforms_size);

	}

	~ShaderPipeline() {
		CUDA_CHECK_RETURN(cudaFreeHost(vert_uniforms));
		CUDA_CHECK_RETURN(cudaFreeHost(frag_uniforms));
		CUDA_CHECK_RETURN(cudaFree(vert_uniforms_gpu));CUDA_CHECK_RETURN(
				cudaFree(frag_uniforms_gpu));
	}

	Vbo* getVbo() {
		return vbo_;
	}

	VtxShaderDesc* getVert() {
		return vert_;
	}

	template<vertexShader_t vert, fragmentShader_t frag>
	void render(cudaSurfaceObject_t surface, DepthBuffer* depth, int width,
			int height) {
		if (vert != vert_->kern) {
			throw CrapGlException(
					"Vertex shader in function template does not match vertex shader in VtxShaderDesc for this pipeline");
		}
		if (frag != frag_->kern) {
			throw CrapGlException(
					"Fragment shader in function template does not match fragment shader in FragShaderDesc for this pipeline");
		}
		if (vert_dirty) {
			vert_dirty = false;
			CUDA_CHECK_RETURN(
					cudaMemcpy(vert_uniforms_gpu, vert_uniforms,
							vert_->uniforms_size, cudaMemcpyHostToDevice));
		}
		if (frag_dirty) {
			frag_dirty = false;
			CUDA_CHECK_RETURN(
					cudaMemcpy(frag_uniforms_gpu, frag_uniforms,
							frag_->uniforms_size, cudaMemcpyHostToDevice));
		}
		if (vbo_->index_count == 0) {
			runVertexShaderUnindexed<vert> <<<vbo_->count/32,32>>>(vbo_->gpu_data, vbo_->count, vbo_->stride,
					vert_uniforms_gpu, tvb_->gpu_data, tvb_->stride,
					tvb_->gpu_vec_data);
		} else {
			runVertexShaderIndexed<vert><<<vbo_->count/32,32>>>(vbo_->gpu_data, vbo_->index_count,
					vbo_->gpu_index_data, vbo_->stride,
					vert_uniforms_gpu, tvb_->gpu_vec_data, tvb_->stride,
					tvb_->gpu_vec_data);
		}

	}

	void markVertDirty() {
		vert_dirty = true;
	}

	void markFragDirty() {
		frag_dirty = true;
	}

	void* getFragUniforms() {
		markVertDirty();
		return frag_uniforms;
	}

	void* getVertUniforms() {
		markFragDirty();
		return vert_uniforms;
	}

private:
	Vbo* const vbo_;
	VtxShaderDesc* const vert_;
	FragShaderDesc* const frag_;
	std::unique_ptr<TransformedVertexBuffer> tvb_;
	void* vert_uniforms;
	void* frag_uniforms;
	void* vert_uniforms_gpu;
	void* frag_uniforms_gpu;
	bool vert_dirty;
	bool frag_dirty;
};

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::stringstream ss;
	ss << statement << " returned " << cudaGetErrorString(err) << "(" << err
			<< ") at " << file << ":" << line;
	std::cout << ss.str() << std::endl;
	throw CrapGlException(ss.str());
}

#endif
