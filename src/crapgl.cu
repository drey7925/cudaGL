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
	friend class ShaderPipeline;
public:
	VtxShaderDesc(size_t inSize, size_t outSize, size_t uniformsSize,
			vertexShader_t* kern) :
			vtxin_size(inSize), vtxout_size(outSize), uniforms_size(
					uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(vertexShader_t), 0,
				cudaMemcpyDeviceToHost);
	}
private:
	const size_t vtxin_size;
	const size_t vtxout_size;
	const size_t uniforms_size;
	vertexShader_t* kern;
};

class FragShaderDesc {
	friend class ShaderPipeline;
	FragShaderDesc(size_t inSize, size_t uniformsSize, fragmentShader_t* kern) :
			vtxin_size(inSize), uniforms_size(uniformsSize) {
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(fragmentShader_t), 0,
				cudaMemcpyDeviceToHost);
	}
private:
	const size_t vtxin_size;
	const size_t uniforms_size;
	fragmentShader_t* kern;
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

class ShaderPipeline {
public:
	ShaderPipeline(Vbo* vbo, VtxShaderDesc* vert, FragShaderDesc* frag) :
			vbo_(vbo), vert_(vert), frag_(frag), cull_(back), depth_(less), vert_dirty(
					false), frag_dirty(false) {
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
		CUDA_CHECK_RETURN(cudaFree(vert_uniforms_gpu));
		CUDA_CHECK_RETURN(cudaFree(frag_uniforms_gpu));
	}

	FaceCulling getCull() const {
		return cull_;
	}

	void setCull(FaceCulling cull) {
		cull_ = cull;
	}

	DepthTest getDepth() const {
		return depth_;
	}

	void setDepth(DepthTest depth) {
		depth_ = depth;
	}

	const Vbo* const & getVbo() const {
		return vbo_;
	}

	const VtxShaderDesc* const & getVert() const {
		return vert_;
	}

	void render(cudaSurfaceObject_t surface, int width, int height) {
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
	}

	void markVertDirty() {
		vert_dirty = true;
	}

	void markFragDirty() {
		frag_dirty = true;
	}

	void* getFragUniforms() const {
		markVertDirty();
		return frag_uniforms;
	}

	void* getVertUniforms() const {
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
	FaceCulling cull_;
	DepthTest depth_;
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
