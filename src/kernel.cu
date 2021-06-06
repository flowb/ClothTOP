#include "ClothTOP.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

surface<void, cudaSurfaceType2D> outputSurface;
surface<void, cudaSurfaceType2D> outputSurface2;


__device__ float4 operator+(const float4 & a, const float4 & b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


__global__ void
PosNormKernel(int npoints, int ntris, Vec4* pos, int* indices, Vec4* normals, Vec3* uvs, int imgw, int imgh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = x + y * imgw;

	int offset1 = 4 * idx;
	int offset2 = 4 * idx + 1;
	int offset3 = 4 * idx + 2;

	if (x < imgw && y < imgh ) 
	{
		if (idx < npoints)
		{
			surf2Dwrite(make_float4(pos[idx].x, pos[idx].y, pos[idx].z, 1), outputSurface, (int)sizeof(float4) * (offset1 % imgw), (offset1 / imgw));
			surf2Dwrite(make_float4(normals[idx].x, normals[idx].y, normals[idx].z, 0.0f), outputSurface, (int)sizeof(float4) * (offset3 % imgw), (offset3 / imgw));
		}

		if (idx < ntris) 
		{
			surf2Dwrite(make_float4((float)indices[3*idx + 2], (float)indices[3 * idx + 1], (float)indices[3 * idx + 0], 0.0f), outputSurface, (int)sizeof(float4) * (offset2 % imgw), (offset2 / imgw));
		}
	}
}

__global__ void
PosNormKernelUV(int npoints, int ntris, Vec4* pos, int* indices, Vec4* normals, Vec3* uvs, int imgw, int imgh)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = x + y * imgw;

	int uv_idx_x = sizeof(float4) * (int)(uvs[idx].x * imgw);
	int uv_idx_y = (int)((uvs[idx].y) * imgh);

	if (x < imgw && y < imgh && idx < npoints) 
	{
		surf2Dwrite(make_float4(pos[idx].x, pos[idx].y, pos[idx].z, 1), outputSurface, uv_idx_x, uv_idx_y);
		surf2Dwrite(make_float4(normals[idx].x, normals[idx].y, normals[idx].z, 0.0f), outputSurface2, uv_idx_x, uv_idx_y);
	}
}

void launch_PosNormKernel(int outputmode, dim3 grid, dim3 block,int npoints, int ntris, void** mapped_g_buff, cudaArray* output1, cudaArray* output2, int imgw, int imgh)
{
	Vec4* pos = (Vec4*)mapped_g_buff[0];
	int* indices = (int*)mapped_g_buff[1];
	Vec4* normals = (Vec4*)mapped_g_buff[2];
	Vec3* uvs = (Vec3*)mapped_g_buff[3];

	cudaCheck(cudaBindSurfaceToArray(outputSurface, output1));
	cudaCheck(cudaBindSurfaceToArray(outputSurface2, output2));

	if(outputmode == 0)
		PosNormKernel << < grid, block >> > (npoints, ntris, pos, indices, normals, uvs, imgw, imgh);
	else
		PosNormKernelUV << < grid, block >> > (npoints, ntris, pos, indices, normals, uvs, imgw, imgh);
}
