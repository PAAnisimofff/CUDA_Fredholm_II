
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include "cublas_v2.h"
#include "cusolverDn.h"


#define a 0.0
#define b 1.0
#define n 100
//double K(double x, double y)
//{
//	return x * y;
//}
//
//__device__  double f(double x)
//{
//	return 2 * x;
//}
//
//__device__ double pi(double x)
//{
//	return 4.0 / (1.0 + x * x);
//}
//
//__device__ double phi(double x, int i)
//{
//	return __powf(x, i);
//}

double ut(double x)
{
	return 3 * x;
}



__device__ void hello()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("hello from %d\n", idx);
}


__global__ void Solve(double *dA, double *dF, double *dX0, double *dX1, int N)
{
	double aa, sum = 0;
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t < N)
	{
		for (int j = 0; j < N; j++)
		{
			sum += dA[j + t * N] * dX0[j];
			//if (t == 0 && j == 0) printf("dA[%d]= %f\tdX0[%d]= %f\n", j + t * N, dA[j + t * N], j, dX0[j]);
			if (j == t)
				aa = dA[j + t * N];
		}
		dX1[t] = dX0[t] + (dF[t] - sum) / aa;
		printf("%f %f %f %f %f\n", dX1[t], dX0[t], dF[t], sum, aa);
	}
	}

__global__ void _resolution(double *x, double *c, double *y, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		double sum = 0;
		for (int i = 0; i < size; i++)
		{
			sum += c[i] * pow(x[idx], i);
			//printf("%f   %f\n", c[i], x[idx]);
		}

		y[idx] = 3 * x[idx] + c[idx];
		//printf("\n\n%f   ", y[idx]);
	}
}


__global__ void KernelJacobi(double* deviceA, double* deviceF, double* deviceX0, double* deviceX1, int N)

{
	double temp;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		deviceX1[i] = deviceF[i];
		for (int j = 0; j < N; j++)
		{
			if (i != j)
				deviceX1[i] -= deviceA[j + i * N] * deviceX0[j];
			else
				temp = deviceA[j + i * N];
		}
		deviceX1[i] /= temp;
	}
}

//Raschetdeltidlyausloviaostanovki

__global__ void EpsJacobi(double* deviceX0, double* deviceX1, double* delta, int N)

{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		delta[i] += abs(deviceX0[i] - deviceX1[i]);
		deviceX0[i] = deviceX1[i];
	}
}

__global__ void kernel(double *A, double *f, double par, double *x0, double *x1, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		int ia = n * idx;
		double sum = 0;
		for (int i = 0; i < size; i++)
			sum += A[ia + i] * x0[i];
		x1[idx] = x0[idx] + par * (sum - f[idx]);
		//printf("%f\n", x1[idx]);
	}
}

__global__ void Eps(double *dX0, double *dX1, double *delta, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		//if (i == 0) printf("d0[%d]= %f\tdX1[%d]= %f\n", i, dX0[i], i, dX1[i]);
		delta[i] = abs(dX0[i] - dX1[i]);
		dX0[i] = dX1[i];
	}
}

__global__ void create_F(double *arr, double *f, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		double ff = 0;
		for (int i = 0; i < size; i++)
			ff += pow(arr[i], idx + 2);
		ff *= 2;
		ff /= size;
		f[idx] = ff;
	}
}

__global__ void createMatrix(double *A, double *arr, int size)
{
	int ind = size * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
	//printf("index = %d", ind);
	if (ind < size * size)
	{
		double alpha = 0;
		double beta = 0;
		for (int i = 0; i < size; i++)
		{
			alpha += __powf(arr[i], blockDim.y * blockIdx.y + threadIdx.y + blockDim.x * blockIdx.x + threadIdx.x + 2);
			beta += (__powf(arr[i], blockDim.x * blockIdx.x + threadIdx.x + 2) / (blockDim.y * blockIdx.y + threadIdx.y + 3));
		/*	if (i == 0 && ind == 0)
				printf("alpha=%f\tbeta=%f\n", alpha, beta);*/
		}

		alpha /= size;
		beta /= size;
		A[ind] = alpha - beta;
		//sum = 0;
	}
	//A[threadIdx.y*size + threadIdx.x] = integral(arr, size, sum);//10 * threadIdx.y + threadIdx.x;
	//printf("A[%d]= %f\n", A[threadIdx.y*size + threadIdx.x]);
}




__global__ void resolution(double *x, double *c, double *y, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		double sum = 0;
		for (int i = 0; i < size; i++)
			sum += c[i] * pow(x[idx], i);
		y[idx] = sum;
	}
}

int main()
{
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	int m = n;
	const int lda = m;
	const int ldb = m;
	const int nrhs = 1; // number of right hand side vectors
	size_t size = m * m * sizeof(double);
	//double h = (b - a) / (n - 1); //шаг
	//double sum_h = 0;
	//double EPS = 1.e-5;
	double *h_A = new double[size]; // матрица
	double *x_h = new double[n]; //сетка
	double *hx_int = new double[n]; //сетка для интеграла
	double *f_h = new double[n]; //правая часть
	//double *h_x0 = new double[n]; //приближение x(n)
	//double *h_x1 = new double[n]; //приближение x(n+1)
	//double *h_delta = new double[n]; //разница |x(n+1)-x(n)|
	double *xc = new double[n];
	double *dx_int, *d_A, *f_d, *d_tau, *d_work, *x_d;
	int lwork = 0, *devInfo;
	int info_gpu = 0;
	const double one = 1;


	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	cublas_status = cublasCreate(&cublasH);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	cudaStat2 = cudaMalloc((void**)&d_tau, sizeof(double)*n);

	cudaEvent_t start, stop;
	double time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int N_threads = 32;
	int blockSize = N_threads * N_threads;
	int blocks = n / blockSize + (n % blockSize == 0 ? 0 : 1);
	cudaMalloc((void **)&x_d, sizeof(double)*n);


	cudaMalloc((void **)&dx_int, sizeof(double)*n); //сетка для интеграла
	//cudaMalloc((void **)&d_x0, sizeof(double)*n); //приближение x(n)
	//cudaMalloc((void **)&d_x1, sizeof(double)*n); //приближение x(n+1)
	//cudaMalloc((void **)&d_delta, sizeof(double)*n);//разница |x(n+1)-x(n)|



	cudaStat1 = cudaMalloc((void **)&d_A, size); //матрица
	cudaStat2 = cudaMalloc((void **)&d_tau, sizeof(double)*n);
	cudaStat3 = cudaMalloc((void **)&f_d, sizeof(double)*n); //правая часть
	cudaStat4 = cudaMalloc((void **)&devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);
	//cudaMemcpy(sum_d, &sum_h, sizeof(double), cudaMemcpyHostToDevice);
	int i = 0, j = 0;
	dim3 threadsPerBlock = dim3(N_threads, N_threads);
	dim3 blocksPerGrid = dim3(blocks * 5, blocks * 5);
	/*for (double x = a + i*h; x <= b; x += h)
	{
		x_h[i] = x;
		i++;
	}
	i = 0;*/
	double h = (b - a) / n; //шаг
	for (double x = a + 0.5 * h; x < b; x += h)
	{
		hx_int[i] = x;
		i++;
	}
	//printf("x[%d]= %f  x[%d]= %f\n", 0, x_h[0], n - 1, x_h[n - 1]);
	/*for (i = 0; i < n; i++)
	{
		printf("x[%d]= %f\n ", i, x_h[i]);
	}
	printf("\n");*/
	//for (i = 0; i < n; i++)
	//	h_x0[i] = 1;
	/*for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
			h_A[j * m + i] = 0;*/
	/*double eps = 1;
	int k = 0;*/
	//cudaMemcpy(x_d, x_h, sizeof(double)*n, cudaMemcpyHostToDevice); //сетка
	cudaMemcpy(dx_int, hx_int, sizeof(double)*n, cudaMemcpyHostToDevice); //сетка интеграла
	//cudaMemcpy(d_x0, h_x0, sizeof(double)*n, cudaMemcpyHostToDevice); //начальное приближение
	//Kernel launch
	createMatrix << <blocksPerGrid, threadsPerBlock >> > (d_A, dx_int, n);
	create_F << <blocks, blockSize >> > (dx_int, f_d, n);
	cudaStat1 = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(f_h, f_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);

	cusolver_status = cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);

	printf("after geqrf: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);


	cusolver_status = cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda, d_tau, f_d, ldb, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	// check if QR is good or not
	cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);

	printf("after ormqr: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

	cublas_status = cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, f_d, ldb);
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(xc, f_d, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);

	cudaMemcpy(f_d, f_h, sizeof(double)*n, cudaMemcpyHostToDevice);
	double err = 0, err_f = 0;
	double sum = 0;
	//printf("\t Matrix A: \n");
	for (i = 0; i < m; i++)
	{
		err_f += 2.0 / (3 + i) - f_h[i];
		for (j = 0; j < m; j++)
		{
			//calc << <blocks, blockSize >> > (x_d, n, sum_d, i, j);
			//cudaMemcpy(&sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
			//h_A[j * m + i] = sum_h / n;
			err += (1.0 / (3 + i + j) - 1.0 / ((3 + i)*(3 + j))) - h_A[j * m + i];
			if(j!=i) sum += abs(h_A[j*m + i] / h_A[i*m + i]);
		//	printf("%f   ", h_A[j*m + i]);
			//sum_h = 0.0;
			//cudaMemcpy(sum_d, &sum_h, sizeof(double), cudaMemcpyHostToDevice);
			//cudaMemcpy(x_d, x_h, sizeof(double)*n, cudaMemcpyHostToDevice);
			//calc << <blocks, blockSize >> > (x_d, n, sum_d, i, j + 2);
			//cudaMemcpy(&sum_h, sum_d, sizeof(double), cudaMemcpyDeviceToHost);
		}
	//	printf("\n");
	}
	//printf("norm= %f\n", sum);
	//printf("\t Right part: \n");
	//for (i = 0; i < n; i++)
		//printf("%f  ", f_h[i]);
	printf("\t Error\n");
	//printf("error= %f\nerror_f= %f\n", err / n / n, err_f / n);
	printf("Error= %f\n", err_f / n);
	//cudaEventRecord(start, 0);
	//while (eps > EPS)
	//{
	//	k++;
	//	cudaMemcpy(d_delta, h_delta, sizeof(double)*n, cudaMemcpyHostToDevice);
	//	//Solve << <blocks, blockSize >> > (d_A, f_d, d_x0, d_x1, n);
	//	KernelJacobi << <blocks, blockSize >> > (d_A, f_d, d_x0, d_x1, n);
	//	EpsJacobi << <blocks, blockSize >> > (d_x0, d_x1, d_delta, n);
	//	cudaMemcpy(h_delta, d_delta, sizeof(double)*n, cudaMemcpyDeviceToHost);
	//	eps = 0;
	//	for (j = 0; j < n; j++)
	//	{
	//		eps += h_delta[j];
	//		h_delta[j] = 0;
	//	}

	//	eps /= n;
	//	//printf("\n Eps[%d]=%f\n ", k, eps);
	//}
	//cudaMemcpy(h_x1, d_x0, sizeof(double)*n, cudaMemcpyDeviceToHost);
	for (i = 0; i < n; i++)
		//printf("x[%d]= %f\n", i, xc[i]);
	_resolution << <blocks, blockSize >> > (dx_int, f_d, x_d, n);
	cudaMemcpy(x_h, x_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
	/*cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);*/	for (i = 0; i < n; i++)
		printf("%f\n", x_h[i]);
	//printf("result=%f\n", sum_h / n);
	//cudaFree(sum_d);
	cudaFree(x_d);
	cudaFree(d_A);
	cudaFree(dx_int);
	if (d_tau) cudaFree(d_tau);
	if (f_d) cudaFree(f_d);
	if (devInfo) cudaFree(devInfo);
	if (d_work) cudaFree(d_work);
	//cudaFree(d_delta);
	//cudaFree(d_x0);
	//cudaFree(d_x1);
	delete[] h_A;
	//delete[] x_h;
	delete[] f_h;
	//delete[] h_delta;
	//delete[] h_x0;
	//delete[] h_x1;
	if (cublasH) cublasDestroy(cublasH);
	if (cusolverH) cusolverDnDestroy(cusolverH);
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
