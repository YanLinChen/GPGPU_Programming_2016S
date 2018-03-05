#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<string>
#include <stdio.h>


struct Screen_Rect{
	int width, height;
};

cudaError_t ThesholdWithCuda(Screen_Rect header, unsigned char**binData, const char *iconData);

__device__ unsigned char GetBoundaryValue(int width, int height, int x, int y){

	if (x >= width-1&&y >= height-1){
		//字串結束
		return '\0';
	}
	else if (x >= width - 1){
		//行結束
		return '\n';
	}
	else{
		//其他
		return ':';
	}
}
__device__ bool isBoundary(int width,int height,int x,int y){
	return x <= 0 || y <= 0 || x >= width-2 || y >= height-1;
}
__global__ void ThesholdKernel(int blocks, int iconWidth,  char *iconData, unsigned char *binData, int binData_width, int binData_height)
{
	int binX = blockIdx.x;
	int binY = threadIdx.x;
	if (binX + binY*blocks >= binData_width*binData_height){
		printf("thread Discard\n");
		return;
	}
	int dataX = binX - 1;
	int dataY = binY - 1;
	int pixelOffset = (dataX + dataY*iconWidth);
	if (isBoundary(binData_width, binData_height, binX, binY)){
		binData[binX + binY*blocks] = GetBoundaryValue(binData_width, binData_height, binX, binY);
	}
	else{
		binData[binX + binY*blocks] = iconData[pixelOffset];
	}
}

int main()
{   
	Screen_Rect nvdiaIconSize;
	nvdiaIconSize.width = 64;
	nvdiaIconSize.height = 32;
	const char *iconData = "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          *************************                                               ************                                      *****                  *****                                *****     *******                **                           ****                ****                                      ****     ****            ****                                 ****     ***    *******       ****                             ****    ****     ***** ****       ***    **                    ****    ****      ****     ***        ***                      *****      ***     *         ***                                   ****       ****         ***                                        ****            *****        ***                                   *****     ***           ****                                         ****           ********                                              ********************                                            ***********************                                                                                                                                                                                                                                                                                                                                                                                                               ";
	unsigned char *binData = NULL;
	cudaError_t cudaStatus = ThesholdWithCuda(nvdiaIconSize, &binData, iconData);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ThesholdWithCuda failed!");
		return 1;
	}
	printf("%s\n", binData);
	free(binData);
    return 0;
}

cudaError_t ThesholdWithCuda(Screen_Rect header, unsigned char**binData, const char *iconData){
	char* Device_IconData;
	unsigned char* Device_binData;
	unsigned char* Host_binData;
	int iconPixels = header.width*header.height;
	int iconDataLength =iconPixels;
	int binPixels = (header.width+3)*(header.height+2);
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&Device_IconData, iconDataLength * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&Device_binData, binPixels * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(Device_IconData, iconData, iconDataLength * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//+2/+3 for boundary
	int nblock = header.width + 3;
	int nthread = header.height + 2;
	ThesholdKernel << <nblock, nthread >> >(nblock, header.width, Device_IconData, Device_binData, (header.width + 3), (header.height + 2));
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ThesholdKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ThesholdKernel!\n", cudaStatus);
	}
	Host_binData = new unsigned char[binPixels];
	cudaStatus = cudaMemcpy(Host_binData, Device_binData, binPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	*binData = Host_binData;
	cudaFree(Device_binData);
	cudaFree(Device_IconData);
	return cudaStatus;
}
