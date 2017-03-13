/*
 ============================================================================
 Name        : cudagl.cu
 Author      : Andrey Akhmetov
 Version     :
 Copyright   : May be reproduced, modified, used with attribution. NO WARRANTY IS GIVEN FOR THIS CODE. Educational intent only, not designed for production systems.
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Struct identifying a VBO. Stride is number of bytes per vertex.
 * Size is number of vertices, index_count is number of indices.
 * cpu_XXX pointers are pointers in RAM
 * gpu_XXX pointers are in CUDA memory
 */
class Vbo {
public:

private:
	short stride;
	short size;
	short index_count;
	void* cpu_data;
	void* gpu_data;
	short* cpu_index_data;
	short* gpu_index_data;
	bool dirty;
};
struct vec4 {
	float x;
	float y;
	float z;
	float t;
};

// vtxid, vtx_in, position_out, vtx_out, uniforms
typedef void(*vertexShader_t)(short vtxid, void* vtx_in, vec4* position_out, void* vtx_out, void* uniforms);
// true if valid frag, false if discard
typedef bool(*fragmentShader_t)(vec4 position_in, void* vtx_in, vec4* color_out, void* uniforms);

// sizes in bytes
class VtxShaderDesc {
public:
	VtxShaderDesc(short inSize, short outSize, short uniformsSize, vertexShader_t* kern) :
		vtxin_size(inSize), vtxout_size(outSize), uniforms_size(uniformsSize)
	{
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
		vtxin_size(inSize), uniforms_size(uniformsSize)
	{
		cudaMemcpyFromSymbol(&this->kern, kern, sizeof(fragmentShader_t));
	}
private:
	const short vtxin_size;
	const short uniforms_size;
	fragmentShader_t* kern;
};

// these live on the GPU! No CPU copy available (for now)
class TransformedVertexBuffer {
private:
	short vtx_count;
	short stride;
	void* gpu_data;
};

class RenderOptions {

};

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}




int main(int argc, char** argv){
	vec4 v4;
	v4.t = 0;
	//if((*foo)(v4, NULL, &v4, NULL)) puts("Hello world\n");

}


