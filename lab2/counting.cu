#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include<device_launch_parameters.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
#define  MAX_THREAD_PER_BLOCK 8.0
#define LOG2 0.30102999566
/////////////////////////////////////////////////////////////////////////////////////////////////////
struct is_linebreak : public thrust::unary_function<char, int>
{
	__host__ __device__
		int operator()(char c) { return c == '\n' ? 0 : 1; }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	is_linebreak op;
	thrust::transform(thrust::device, text, text + text_size, pos, op);
	thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ int Islinebreak(char c){
	return c == '\n' ? 0 : 1;
}
__global__ void Transform(const char * src, int * dst, int size, int blockSize){
	int index = blockIdx.x*blockSize + threadIdx.x;
	if (index<size)
		dst[index] = Islinebreak(src[index]);
}
int Higher_Power_of_two(int size, int &times){
	times = ceil(log(size) / log(2));
	return  pow(2, times);
}
__global__ void Reduce_Kernel(int* flag, int *val, int size, int newSize, int blockSize, int length){
	int index = blockIdx.x*blockSize + (threadIdx.x+1)*length * 2;
	int index2 = index-length-1;
	index--;
	if (flag[index]){
		val[index] = val[index2] + val[index];
	}
	flag[index] = flag[index] & flag[index2];
}
__global__ void DownSweep_Kernel(int* flag, int* flag2, int *val, int size, int newSize, int blockSize, int length){
	int index = blockIdx.x*blockSize + (threadIdx.x + 1)*length * 2;
	int index2 = index - length - 1;
	index--;

	int temp = val[index2];
	//int temp2 = flag[index2];
	val[index2]=val[index];
	//flag[index2] = flag[index];
	if (flag2[index2 + 1]==0){
		val[index] = 0;
		//flag[index] = 1;
	}
	else if (flag[index2] == 0){
		val[index] = temp;
		//flag[index] = temp2;
	}
	else{
		val[index] += temp;
		//flag[index] = temp2&flag[index];
	}
	flag[index2] = 1;
}
__global__ void inclusive_Kernel(int* flag, int *val, int *nval, int size, int blockSize){
	int index = blockIdx.x*blockSize + threadIdx.x;
	if (index >= size)return;
	val[index] = (nval[index] +flag[index])*flag[index];
	//printf("%d %d %d\n",index,val[index],flag[size+index]);
}

void SegmentedScan(int* flag, int *val, int text_size){
	int loopTimes = 0;
	int newSize = Higher_Power_of_two(text_size, loopTimes);
	int *n_flag,*o_flag,*n_val;
	cudaMalloc((void**)&n_flag, sizeof(int)*newSize);
	cudaMalloc((void**)&o_flag, sizeof(int)*newSize);
	cudaMalloc((void**)&n_val, sizeof(int)*newSize);
	cudaMemset(n_flag, 1, sizeof(int)*newSize);
	cudaMemset(o_flag, 1, sizeof(int)*newSize);
	cudaMemset(n_val, 0, sizeof(int)*newSize);
	cudaMemcpy(n_flag, flag, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(o_flag, flag, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(n_val, val, sizeof(int)*text_size, cudaMemcpyDeviceToDevice);
	int blockNums = ceil(newSize / MAX_THREAD_PER_BLOCK/2);
	int threadNums = blockNums <= 1 ? newSize : MAX_THREAD_PER_BLOCK*2;
	int length = 1;
	int tempThreadNum = threadNums/2;
	for (int i = 0; i < loopTimes; ++i){
		Reduce_Kernel << <blockNums, tempThreadNum >> >(n_flag, n_val, text_size, newSize, threadNums, length);
		length *= 2;
		if (blockNums <= 1){
			tempThreadNum = tempThreadNum / 2 < 1 ? 1 : tempThreadNum / 2;
		}
		blockNums = blockNums / 2 < 1 ? 1 : blockNums / 2;
		threadNums *= 2;
	}
	cudaMemset(&n_val[newSize - 1], 0, 1);
	threadNums /= 2;
	length /= 2;
	tempThreadNum = 1;
	for (int i = 0; i < loopTimes; ++i){
		DownSweep_Kernel << <blockNums, tempThreadNum >> >(n_flag, o_flag, n_val, text_size, newSize, threadNums, length);
		length /= 2;
		if (tempThreadNum >= MAX_THREAD_PER_BLOCK){
			blockNums = blockNums * 2;
		}
		tempThreadNum = tempThreadNum * 2 >= MAX_THREAD_PER_BLOCK ? MAX_THREAD_PER_BLOCK : tempThreadNum * 2;
		threadNums /= 2;
	}
	blockNums = ceil(text_size / MAX_THREAD_PER_BLOCK);
	threadNums = MAX_THREAD_PER_BLOCK;
	inclusive_Kernel << <blockNums, threadNums >> >(o_flag, val, n_val, text_size, threadNums);
}
void CountPosition2(const char *text, int *pos, int text_size)
{
	int blockNums = ceil(text_size / MAX_THREAD_PER_BLOCK);
	int threadNums = MAX_THREAD_PER_BLOCK;
	Transform << <blockNums, threadNums >> >(text, pos, text_size, threadNums);
	SegmentedScan(pos, pos,text_size);
}

