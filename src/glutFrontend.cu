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

#define NO_CHECK_CUDA
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include "crapgl.cu"
int count = 0;
GLuint glTexID;
cudaGraphicsResource_t cudaTexSurface;

int texWidth;
int texHeight;
bool runOnce = true;

__device__ bool fragDummy(vec4 pos, float* vtx_in,
		unsigned int& color_out, void* uniforms){color_out = (((int) (pos.x*256))) + (((int) (pos.y*256))<<8) + 0xFFFF0000u;
return ((int)(pos.x*32)+(int)(pos.y*32))%2==0;};

const fragmentShader_t fd = fragDummy;
const depthTest_t dt = depth_test::greater;
__global__ void dummyRenderKernel(cudaSurfaceObject_t surf, int width,
		int height) {

	unsigned int c4 = (threadIdx.x) | (threadIdx.x << 8) | (255 << 24);
	int mindim = min(width, height);
	for (int y = threadIdx.x; y < mindim; y += blockDim.x) {
		for (int x = y + blockIdx.x; x < width; x += (gridDim.x)) {
			if((x/2+y/2)%2==0) surf2Dwrite(c4, surf, x * sizeof(char4), y);
		}
	}

}

int iters = 0;
vec4 vertsCpu[] = {
	vec4(1, 1, 0, 1),
	vec4(1, 1, 1, 1),
	vec4(1, 0, 1, 1),
	vec4(1, 0, -1, 1),
	vec4(1, 0, 0, 1),
	vec4(1, 1, -1, 1)
};
unsigned int* depthBuffer;
vec4* vertsGpu;
void cudaDrawToTexture() {
	void* null = (void*) 0;

	if (runOnce) {
		cudaGraphicsMapResources(1, &cudaTexSurface);
		{
			cudaArray_t viewCudaArray;
			cudaGraphicsSubResourceGetMappedArray(&viewCudaArray,
					cudaTexSurface, 0, 0);
			cudaResourceDesc viewCudaArrayResourceDesc;
			{
				viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
				viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
			}
			cudaSurfaceObject_t viewCudaSurfaceObject;
			cudaCreateSurfaceObject(&viewCudaSurfaceObject,
					&viewCudaArrayResourceDesc);
			{
				crapGlClear<<<64,256>>>(texWidth, texHeight, true, false, 0xFFFFFF00, 0, viewCudaSurfaceObject, (int*) 0);

				cudaDeviceSynchronize();
				clock_t tStart = clock();
				CUDA_CHECK_RETURN(cudaMemcpy(vertsGpu, vertsCpu, sizeof(vec4)*6, cudaMemcpyHostToDevice));
				//dummyRenderKernel<<<64, 256>>>(viewCudaSurfaceObject, texWidth, texHeight);
				runFragmentShader<fd, false, false, dt, 4, 1><<<64,32>>>(viewCudaSurfaceObject, (float*) null, (vec4*) vertsGpu, (short) 6, texWidth, texHeight, depthBuffer, depthBuffer);
				cudaDeviceSynchronize();
				clock_t time = clock() - tStart;
				iters++;
				printf("%f shade megaops per second\n",
						texWidth * texHeight * CLOCKS_PER_SEC / (double) (time)
								/ 1000000);

			}
			cudaDestroySurfaceObject(viewCudaSurfaceObject);
		}
		cudaGraphicsUnmapResources(1, &cudaTexSurface);
		cudaDeviceSynchronize();

	}
	runOnce = false;
}

void RenderSceneCB() {
	cudaDrawToTexture();
	glClear(GL_COLOR_BUFFER_BIT);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glBindTexture(GL_TEXTURE_2D, glTexID);
	{
		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.0f, 0.0f);
			glVertex2f(-1.0f, -1.0f);
			glTexCoord2f(1.0f, 0.0f);
			glVertex2f(+1.0f, -1.0f);
			glTexCoord2f(1.0f, 1.0f);
			glVertex2f(+1.0f, +1.0f);
			glTexCoord2f(0.0f, 1.0f);
			glVertex2f(-1.0f, +1.0f);
		}
		glEnd();
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	glFinish();
	glutSwapBuffers();
	glutPostRedisplay();
	count++;
	//printf("%d\n", count);

}

void initTextures(int width, int height) {
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &glTexID);

	glBindTexture(GL_TEXTURE_2D, glTexID);
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	uint8_t data[4];
	data[0] = 255;
	data[1] = 255;
	data[2] = 0;
	data[3] = 255;
	glBindTexture(GL_TEXTURE_2D, glTexID);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			glTexSubImage2D(GL_TEXTURE_2D, 0, i, j, 1, 1,
			GL_RGBA,
			GL_UNSIGNED_BYTE, data);
		}
	}
	CUDA_CHECK_RETURN(
			cudaGraphicsGLRegisterImage(&cudaTexSurface, glTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	texWidth = width;
	texHeight = height;

}

void resize(int width, int height) {
	glViewport(0, 0, width, height);
	CUDA_CHECK_RETURN(cudaGraphicsUnregisterResource(cudaTexSurface));
	glBindTexture(GL_TEXTURE_2D, glTexID);
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	CUDA_CHECK_RETURN(
			cudaGraphicsGLRegisterImage(&cudaTexSurface, glTexID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	texWidth = width;
	texHeight = height;
	runOnce = true;
}

void keyb(unsigned char c, int x, int y) {
	if (c == 'a')
		runOnce = true;
}

int main(int argc, char** argv) {
	CUDA_CHECK_RETURN(cudaMalloc(&vertsGpu, sizeof(vec4)*6));
	CUDA_CHECK_RETURN(cudaMalloc(&depthBuffer, sizeof(unsigned int)*1920*1080));
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(1024, 768);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("CRAP-GL render surface (GLUT)");
	glutDisplayFunc(RenderSceneCB);
	glutKeyboardFunc(keyb);
	glutReshapeFunc(resize);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	initTextures(1024, 768);
	glutMainLoop();

}
