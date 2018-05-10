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
#define  MAX_THREAD_PER_BLOCK 1024.0
#define WARPSIZE 32
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
__global__ void lastbreak(const char * src, int *val, int size, int blockSize){
	int index = blockIdx.x*blockSize + threadIdx.x;
	if (index >= size)return;
	for (int i = index; i >= 0; --i){
		if (src[i] == '\n'){
			val[index] = index-i;
			return;
		}
	}
	val[index] = index + 1;
	return;	
}
void Scan(const char *text,int *val, int text_size){
	int blockNums = ceil(text_size / MAX_THREAD_PER_BLOCK);
	int threadNums = MAX_THREAD_PER_BLOCK;
	lastbreak << <blockNums, threadNums >> >(text, val, text_size, threadNums);
}
void CountPosition2(const char *text, int *pos, int text_size)
{
	int blockNums = ceil(text_size / MAX_THREAD_PER_BLOCK);
	int threadNums = MAX_THREAD_PER_BLOCK;
	Scan(text, pos, text_size);
}

