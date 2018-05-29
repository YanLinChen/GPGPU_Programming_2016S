#include "lab3.h"
#include <cstdio>
#include<cuda_runtime_api.h>
#include<cuda_device_runtime_api.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#define ITERTIMES 20000

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}
__global__ void PoissonInit(
	const float *background,
	const float *target,
	const float *mask,
	float *mask2,
	int *mask3,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	const int y_negt = yt - 1, x_negt = xt - 1, y_post = yt + 1, x_post = xt + 1;
	const float tval[3] = { target[curt * 3 + 0], target[curt * 3 + 1], target[curt * 3 + 2] };
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		
		const int yb = oy + yt, xb = ox + xt;
		//make sure in background
		if (yb >= hb || yb < 0 || xb >= wb || xb < 0)return;
		const int y_negb = yb - 1, x_negb = xb - 1;
		const int y_posb = yb + 1, x_posb = xb + 1;
		const int curb = wb*yb + xb;

		int counts = 0;
		int direction = 0;
		float Cts[3] = { 0, 0, 0 };
		if (y_negb >= 0){
			direction = direction | 1;
			++counts;
		}
		if (y_posb < hb){
			direction = direction | 2;
			++counts;
		}
		if (x_negb >= 0){
			direction = direction | 4;
			++counts;
		}
		if (x_posb < wb){
			direction = direction | 8;
			++counts;
		}
		if (y_negt >= 0&&direction&1){
			Cts[0] += tval[0] - target[(curt - wt) * 3 + 0];
			Cts[1] += tval[1] - target[(curt - wt) * 3 + 1];
			Cts[2] += tval[2] - target[(curt - wt) * 3 + 2];
			if (mask[curt - wt]<=127.0){
				Cts[0] += background[(curb-wb) * 3 + 0];
				Cts[1] += background[(curb-wb) * 3 + 1];
				Cts[2] += background[(curb-wb) * 3 + 2];
				direction = direction & 0xE;
			}

		}

		
		if (y_post  < ht &&direction & 2){
			Cts[0] += tval[0] - target[(curt + wt) * 3 + 0];
			Cts[1] += tval[1] - target[(curt + wt) * 3 + 1];
			Cts[2] += tval[2] - target[(curt + wt) * 3 + 2];
			if (mask[curt + wt] <= 127.0){
				Cts[0] += background[(curb + wb) * 3 + 0];
				Cts[1] += background[(curb + wb) * 3 + 1];
				Cts[2] += background[(curb + wb) * 3 + 2];
				direction = direction & 0xD;
			}

		}

		if (x_negt >= 0 && direction & 4){
			Cts[0] += tval[0] - target[(curt - 1) * 3 + 0];
			Cts[1] += tval[1] - target[(curt - 1) * 3 + 1];
			Cts[2] += tval[2] - target[(curt - 1) * 3 + 2];
			if (mask[curt - 1] <= 127.0){
				Cts[0] += background[(curb - 1) * 3 + 0];
				Cts[1] += background[(curb - 1) * 3 + 1];
				Cts[2] += background[(curb - 1) * 3 + 2];
				direction = direction & 0xB;
			}


		}

		if (x_post < wt&&direction & 8){
			Cts[0] += tval[0] - target[(curt + 1) * 3 + 0];
			Cts[1] += tval[1] - target[(curt + 1) * 3 + 1];
			Cts[2] += tval[2] - target[(curt + 1) * 3 + 2];
			if (mask[curt + 1] <= 127.0){
				Cts[0] += background[(curb + 1) * 3 + 0];
				Cts[1] += background[(curb + 1) * 3 + 1];
				Cts[2] += background[(curb + 1) * 3 + 2];
				direction = direction & 0x7;
			}
		}

		
		if (counts == 0){
			//this is impossible when mask and tex is not empty 
		}
		else{
			mask2[curt * 3 + 0] = Cts[0];
			mask2[curt * 3 + 1] = Cts[1];
			mask2[curt * 3 + 2] = Cts[2];
		}
		mask3[curt] = (counts << 4) + direction;
	}

}
__global__ void PoissonBlending(
	const float *target,
	const float *mask,
	float *mask2,
	int *mask3,
	float *output,
	float *outputbuf,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		if (yb >= hb || yb < 0 || xb >= wb || xb < 0)return;
		const int y_negb = yb - 1, x_negb = xb - 1;
		const int y_posb = yb + 1, x_posb = xb + 1;
		const int curb = wb*yb + xb;
		
		
		int counts = mask3[curt] >> 4;
		int direction = mask3[curt] % 16;
		int direction2 = 0;
		float vals[3] = { mask2[curt*3 + 0], mask2[curt*3 + 1], mask2[curt *3+ 2] };
		
		if ((direction & 1)){
			vals[0] += outputbuf[(curb -wb) * 3 + 0];
			vals[1] += outputbuf[(curb -wb) * 3 + 1];
			vals[2] += outputbuf[(curb -wb) * 3 + 2];
		}
		if ((direction & 2)){
			vals[0] += outputbuf[(curb + wb) * 3 + 0];
			vals[1] += outputbuf[(curb + wb) * 3 + 1];
			vals[2] += outputbuf[(curb + wb) * 3 + 2];
		}
		if ((direction & 4)){
			vals[0] += outputbuf[(curb - 1) * 3 + 0];
			vals[1] += outputbuf[(curb - 1) * 3 + 1];
			vals[2] += outputbuf[(curb - 1) * 3 + 2];
		}
		if ((direction & 8)){
			vals[0] += outputbuf[(curb + 1) * 3 + 0];
			vals[1] += outputbuf[(curb + 1) * 3 + 1];
			vals[2] += outputbuf[(curb + 1) * 3 + 2];
		}
		vals[0] /= counts;
		vals[1] /= counts;
		vals[2] /= counts;
		output[curb * 3 + 0] = (vals[0]) ;
		output[curb * 3 + 1] = (vals[1]) ;
		output[curb * 3 + 2] = (vals[2]) ;

	}
}



void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);

	float *mask2;
	int *mask3;
	cudaError_t e;
	e=cudaMalloc((void**)&mask2,sizeof(float)*wt*ht*3);
	if (e != cudaSuccess){
		printf("CUDA Memory allocate fail\n");
	}
	e = cudaMalloc((void**)&mask3, sizeof(int)*wt*ht);
	if (e != cudaSuccess){
		printf("CUDA Memory allocate fail\n");
	}
	PoissonInit << <dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >> >(
		background, target, mask, mask2, mask3,
		wb, hb, wt, ht, oy, ox
		);
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess){
		printf("CUDA Kernel fail : %s\n",cudaGetErrorString(e));
	}
	float *outputbuf;
	e=cudaMalloc((void**)&outputbuf, sizeof(float)*wb*hb * 3);
	if (e != cudaSuccess){
		printf("CUDA Memory allocate fail\n");
	}
	e=cudaMemcpy(outputbuf,output, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	if (e != cudaSuccess){
		printf("CUDA Memory copy fail\n");
	}
	for (int i = 0; i < ITERTIMES; ++i){
		PoissonBlending << <dim3(CeilDiv(wt, 32), CeilDiv(ht, 16)), dim3(32, 16) >> >(
			target, mask, mask2, mask3,
			output, outputbuf,
			wb, hb, wt, ht, oy, ox
		);
		cudaMemcpy(outputbuf, output, wb*hb*sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	}
	e = cudaDeviceSynchronize();
	if (e != cudaSuccess){
		printf("CUDA Kernel fail : %s\n", cudaGetErrorString(e));
	}
	cudaFree(mask2);
	cudaFree(mask3);
	cudaFree(outputbuf);
}
