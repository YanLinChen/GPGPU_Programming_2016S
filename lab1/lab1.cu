#include "lab1.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
static const unsigned W = 640;
static const unsigned H = 640;
static const unsigned NFRAME = 1200;



struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
	min.x = -1.81632441553;
	min.y = -1.17321289392;
	max.x = 0.18367558446;
	max.y = 0.82678710607;
	len.x = max.x - min.x;
	len.y = max.y - min.y;
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


__device__ int3 RGB2YUV(int3 RGB){
	int3 YUV;
	YUV.x = (0.299*RGB.x + 0.587*RGB.y + 0.114*RGB.z);
	YUV.y = (-0.169*RGB.x - 0.331*RGB.y + 0.5*RGB.z) + 128;
	YUV.z = (0.5*RGB.x - 0.419*RGB.y - 0.081*RGB.z) + 128;
	return YUV;
}
__device__ double Mandelbrot(double2 Pos, int maxIteration){
	int iteration = 0;
	double2 z;
	z.x = z.y = 0;
	while (iteration < maxIteration && (z.x*z.x + z.y*z.y) < 4.0)
	{
		double temp = (z.x*z.x) - (z.y*z.y) + Pos.x;
		z.y = (z.y*z.x*2.0) + Pos.y;
		z.x = temp;
		++iteration;
	}
	if (iteration < maxIteration){
		double log_zn = log(z.x*z.x + z.y*z.y) / 2;
		double nu = log(log_zn / log(2.0)) / log(2.0);
		iteration = iteration + 1 - nu;
	}
	return (double)(iteration*1.0)/ maxIteration;
}
__device__ int3 falseColor(float gray){
	/*
	const float Colors[18] = {
		0, 0, 0.56,
		0, 0, 1,
		0,1, 1,
		1, 1, 0,
		1, 0, 0,
		0, 0, 0
	};
	const int mapID[10] = {
		0,1,1,2,2,3,3,4,4,4
	};
	const int mapID2[5] = {
		0,1,3,5,7
	};
	const float scalar[5] = {
		1,2,2,2,1
	};
	*/
	const float Colors[12] = {
	0, 0, 0.56,
	0, 0, 1,
	1, 1, 1,
	0, 0, 0
	};
	const int mapID[10] = {
		0, 0, 1, 1, 1, 1, 1, 2, 2, 2
	};
	const int mapID2[3] = {
		0, 2, 7
	};
	const float scalar[3] = {
		2, 5, 1
	};

	int3 ans;
	int step = floor(gray * 8);
	step = mapID[step];
	float v = (gray * 8.0 - mapID2[step]) / scalar[step];
	ans.x = (Colors[step * 3] * (1.0 - v) + Colors[step * 3 + 3] * v) * 255 * gray;
	ans.y = (Colors[step * 3 + 1] * (1.0 - v) + Colors[step * 3 + 3 + 1] * v) * 255 * gray;
	ans.z = (Colors[step * 3 + 2] * (1.0 - v) + Colors[step * 3 + 3 + 2] * v) * 255 * gray;
	//ans.x = ans.y = ans.z = gray * 255;
	return ans;
}
__global__ void GenerateFrame(uint8_t *yuv, int W, int H, double scale, int maxIteration, float2 min, float2 len,int sampleScale){
	int2 Pos;
	Pos.x = blockIdx.x*sampleScale;
	Pos.y = threadIdx.x*sampleScale;
	double mandelbrot = 0;
	for (int i = 0; i < sampleScale; ++i){
		for (int j = 0; j < sampleScale; ++j){
			double2 Pos2;
			Pos2.x = (Pos.x - W*sampleScale * 0.5)*1.0 / W / sampleScale / scale + 0.5;
			Pos2.y = (Pos.y - H*sampleScale * 0.5)*1.0 / H / sampleScale / scale + 0.5;
			Pos2.x = min.x + len.x*Pos2.x;
			Pos2.y = min.y + len.y*Pos2.y;
			double  mandelbrot_temp = Mandelbrot(Pos2, maxIteration);
			mandelbrot += mandelbrot_temp;
		}
	}
	mandelbrot = mandelbrot / sampleScale / sampleScale;
	int3 RGB = falseColor(mandelbrot);
	int3 YUV = RGB2YUV(RGB);
	int offsetY = blockIdx.x + threadIdx.x*W;
	int offsetU = blockIdx.x / 2 + threadIdx.x / 2 * W / 2 + W*H;
	int offsetV = blockIdx.x / 2 + threadIdx.x / 2 * W / 2 + W*H*1.25;
	yuv[offsetY] = YUV.x;
	yuv[offsetU] = YUV.y;
	yuv[offsetV] = YUV.z;
}
void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	printf("\r%f %% processing\t\t\t\t", impl->t * 100.0 / NFRAME);
	cudaError_t cudaStatus;
	double scale = exp((impl->t)*0.015)+1.0;
	int maxIteration = 50 * pow(log10(scale * 2), 1.25);

	
	//do not let W and H bigger than block size limit
	GenerateFrame << <W, H >> >(yuv, W, H, scale, maxIteration,min,len,3);
	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();
	++(impl->t);

}
