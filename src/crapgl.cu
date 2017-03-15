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
#ifndef CRAPGL_CU
#define CRAPGL_CU
static void CheckCudaErrorAux(const char *, unsigned, const char *,
		cudaError_t);
#ifndef NO_CHECK_CUDA
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#else
#define CUDA_CHECK_RETURN(value) value
#endif
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
public:
	Vbo(short stride, short capacity, short indexCapacity) :
			stride(computeStride(stride)), capacity(capacity), count(0), index_capacity(
					capacity), index_count(0), cpu_data(
					cudaHostMallocHelper(
							computeSize(computeStride(stride), capacity))), cpu_index_data(
					(short*) cudaHostMallocHelper(
							computeIndexSize(indexCapacity))), gpu_data(
					cudaMallocHelper(
							computeSize(computeStride(stride), capacity))), gpu_index_data(
					(short*) cudaMallocHelper(computeIndexSize(indexCapacity))), size(
					computeSize(computeStride(stride), capacity)), index_size(
					computeIndexSize(indexCapacity)), dirty(false), indices_dirty(
					false) {
	}
	~Vbo() {
		CUDA_CHECK_RETURN(cudaFreeHost(cpu_data));
		CUDA_CHECK_RETURN(cudaFreeHost(cpu_index_data));
		CUDA_CHECK_RETURN(cudaFree(gpu_data));
		CUDA_CHECK_RETURN(cudaFree(gpu_index_data));
	}
	void* getDataBuffer() {
		dirty = true;
		return cpu_data;
	}
	short* getIndexBuffer() {
		indices_dirty = true;
		return cpu_index_data;
	}
	void updateBuffers() {
		if (dirty) {
			CUDA_CHECK_RETURN(
					cudaMemcpy(gpu_data, cpu_data, size,
							cudaMemcpyHostToDevice));
			dirty = false;
		}
		if (indices_dirty) {
			CUDA_CHECK_RETURN(
					cudaMemcpy(gpu_index_data, cpu_index_data, index_size,
							cudaMemcpyHostToDevice));
		}
	}
private:

	size_t computeIndexSize(short index_capacity) {
		return sizeof(short) * index_capacity;
	}

	const size_t stride;
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
struct vec4 {
	float x;
	float y;
	float z;
	float t;
};

// vtxid, vtx_in, position_out, vtx_out, uniforms
typedef void (*vertexShader_t)(short vtxid, void* vtx_in, vec4* position_out,
		void* vtx_out, void* uniforms);
// true if valid frag, false if discard
typedef bool (*fragmentShader_t)(vec4 position_in, void* vtx_in,
		vec4* color_out, void* uniforms);

// sizes in bytes
class VtxShaderDesc {
public:
	VtxShaderDesc(short inSize, short outSize, short uniformsSize,
			vertexShader_t* kern) :
			vtxin_size(inSize), vtxout_size(outSize), uniforms_size(
					uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(vertexShader_t));
	}
private:
	const short vtxin_size;
	const short vtxout_size;
	const short uniforms_size;
	vertexShader_t* kern;
};

class FragShaderDesc {
	FragShaderDesc(short inSize, short uniformsSize, fragmentShader_t* kern) :
			vtxin_size(inSize), uniforms_size(uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(fragmentShader_t));
	}
private:
	const short vtxin_size;
	const short uniforms_size;
	fragmentShader_t* kern;
};

// these live on the GPU! No CPU copy available (for now)
class TransformedVertexBuffer {
public:
	TransformedVertexBuffer(short capacity, size_t desiredStride) :
			vtx_count(0), vtx_capacity(capacity), stride(
					computeStride(desiredStride)), gpu_data(
					cudaMallocHelper(
							computeSize(computeStride(desiredStride),
									capacity))) {
	}

	~TransformedVertexBuffer() {
		CUDA_CHECK_RETURN(cudaFree(gpu_data));
	}
private:
	short vtx_count;
	const short vtx_capacity;
	const size_t stride;
	void* const gpu_data;
};

enum FaceCulling {
	front, back, none
};
enum DepthTest {
	greater, less, greater_or_equal, less_or_equal, always
};

class RenderOptions {
public:
	FaceCulling culling;
	DepthTest depthTest;
};

void render(Vbo* vbo, VtxShaderDesc* vertShader, FragShaderDesc* fragShader, TransformedVertexBuffer* tvb){

}



class CrapGlException: public std::exception {
private:
    std::string message_;
public:

    CrapGlException(const std::string& message) : message_(message) { }
    virtual const char* what() const throw() {
        return message_.c_str();
    }
    virtual ~CrapGlException() throw() {};
};



/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
		const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
	exit(1);
}


#endif
