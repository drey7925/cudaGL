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
__global__ void dummyRenderKernel(cudaSurfaceObject_t surf, int width,
		int height) {
	int c4 = (threadIdx.x) | (threadIdx.x << 8) | (255 << 24);
	for (int x = blockIdx.x; x < width; x += gridDim.x) {
		for (int y = threadIdx.x; y < height; y += blockDim.x) {
			if (x > y)
				surf2Dwrite(c4, surf, x * sizeof(char4), y);
		}
	}
}

long totalTime = 0;
int iters = 0;
void cudaDrawToTexture() {
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

				clock_t tStart = clock();
				dummyRenderKernel<<<64,256>>>(viewCudaSurfaceObject, texWidth, texHeight);

				cudaStreamSynchronize(0);
				clock_t time = clock() - tStart;
				totalTime += time;
				iters++;
				printf("%.0f shade megaops per second (avg %0f)\n",
						texWidth * texHeight * CLOCKS_PER_SEC / (double) (time)
								/ 1000000,
						iters * texWidth * texHeight * CLOCKS_PER_SEC
								/ (double) (totalTime) / 1000000);

			}
			cudaDestroySurfaceObject(viewCudaSurfaceObject);
		}
		cudaGraphicsUnmapResources(1, &cudaTexSurface);

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
	void* ptr;
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
