
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define a 0.0
#define b 1.0
#define n 1000

//float K(float x, float y)
//{
//	return x * y;
//}
//
//__device__  float f(float x)
//{
//	return 2 * x;
//}
//
//__device__ float pi(float x)
//{
//	return 4.0 / (1.0 + x * x);
//}
//
//__device__ float phi(float x, int i)
//{
//	return __powf(x, i);
//}

float ut(float x)
{
	return 3 * x;
}


//__device__ float integral(float *ar, int m, float sum)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	//float val = ar[idx];
//	if (idx < m)
//	{
//		ar[idx] = phi(ar[idx], 3);
//		atomicAdd(&sum, ar[idx]);
//	}
//	printf("hello from %d, sum= &f\n", idx, sum);
//	return sum;
//}

__device__ void hello()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("hello from %d\n", idx);
}


//__global__ void calc(float *ar, int m, float *sum, int i, int j)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	//printf("idx=%d\n", idx);
//	//float val = ar[idx];
//	//hello();
//	if (idx < m)
//	{
//		//hello();
//		//ar[idx] = phi(ar[idx], 3);
//		ar[idx] = __powf(ar[idx], i + j);
//		atomicAdd(sum, ar[idx]);
//		//printf("hello from %d, sum= &f\n", idx, sum);
//	}
//}


__global__ void create_F(float *arr, float *f, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		float ff = 0;
		for (int i = 0; i < size; i++)
			ff += __powf(arr[i], blockIdx.x * blockDim.x + threadIdx.x + 1);
		ff *= 2;
		ff /= size;
		f[idx] = ff;
	}
}

__global__ void createMatrix(float *A, float *arr, int size)
{
	int ind = size * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	//printf("index = %d", ind);
	if (ind < size * size)
	{
		float alpha = 0;
		float beta = 0;
		for (int i = 0; i < size; i++)
		{
			alpha += __powf(arr[i], blockDim.y * blockIdx.y + threadIdx.y + blockDim.x * blockIdx.x + threadIdx.x);
			beta += __powf(arr[i], blockDim.y * blockIdx.y + threadIdx.y + blockDim.x * blockIdx.x + threadIdx.x + 2);
		}

		alpha /= size;
		beta /= size;
		A[ind] = alpha - beta;
		//sum = 0;
	}
	//A[threadIdx.y*size + threadIdx.x] = integral(arr, size, sum);//10 * threadIdx.y + threadIdx.x;
	//printf("A[%d]= %f\n", A[threadIdx.y*size + threadIdx.x]);
}
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


int main()
{
	float h = (b - a) / n;
	float sum_h = 0;
	float *x_h = new float[n];
	float *f_h = new float[n];
	float *x_d, *sum_d, *d_A, *f_d;
	int m = n;
	size_t size = m * m * sizeof(float);
	int blockSize = 256;
	int blocks = n / blockSize + (n % blockSize == 0 ? 0 : 1);
	cudaMalloc((void **)&x_d, sizeof(float)*n);
	cudaMalloc((void **)&f_d, sizeof(float)*n);
	cudaMalloc((void **)&sum_d, sizeof(float));
	cudaMalloc((void **)&d_A, size);
	cudaMemcpy(sum_d, &sum_h, sizeof(float), cudaMemcpyHostToDevice);
	int i = 0;
	dim3 threadsPerBlock = dim3(32, 32);
	dim3 blocksPerGrid = dim3(3, 3);
	float *h_A = new float[size];
	for (float x = a + 0.5*h; x < b; x += h)
	{
		x_h[i] = x;
		i++;
	}
	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
			h_A[j * m + i] = 0;
	cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice);
	//Kernel launch
	createMatrix << <blocksPerGrid, threadsPerBlock >> > (d_A, x_d, n);
	create_F << <blocks, blockSize >> > (x_d, f_d, n);
	cudaMemcpy(f_h, f_d, sizeof(float)*n, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	float err = 0, err_f = 0;
	for (int i = 0; i < m; i++)
	{
		err_f += 2.f / (2 + i) - f_h[i];
		for (int j = 0; j < m; j++)
		{
			//calc << <blocks, blockSize >> > (x_d, n, sum_d, i, j);
			//cudaMemcpy(&sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
			//h_A[j * m + i] = sum_h / n;
			err += (1.f / (1 + i + j) - 1.f / (3 + i + j)) - h_A[j * m + i];
			//printf("h_A[%d][%d]= %f\n", i, j, h_A[j*m + i]);
			//sum_h = 0.0;
			//cudaMemcpy(sum_d, &sum_h, sizeof(float), cudaMemcpyHostToDevice);
			//cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice);
			//calc << <blocks, blockSize >> > (x_d, n, sum_d, i, j + 2);
			//cudaMemcpy(&sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
		}
		//printf("\n");
	}
	printf("error= %f\nerror_f= %f", err / n / n, err_f / n);
	//printf("result=%f\n", sum_h / n);
	cudaFree(sum_d);
	cudaFree(x_d);
	cudaFree(d_A);
	delete[] h_A;
	delete[] x_h;
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
