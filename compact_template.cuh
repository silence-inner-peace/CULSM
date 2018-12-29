/*
input : d_Val, d_isValid, totalNum
output: d_ValValid, d_numValid
思想：先对d_isValid做scan操作，得到每个有效元素应该存放的index

example
input
	d_Val      : 0 0 0 0 0 1 2 3 4 5
	d_isValid  : 1 0 0 0 0 1 1 1 1 1 
	totalNum   : 10
mid 
	d_scanValid: 0 1 1 1 1 1 2 3 4 5		得到有效元素compact后的位置
output
	d_ValValid : 0         1 2 3 4 5
	d_numValid : 6


if(d_isValid[gid] == 1)
	d_ValValid[d_scanValid[gid]] = d_Val[gid];


 */

// #include <stdio.h>
// #include <iostream>
// #include <time.h>
// using namespace std;



// #include <stdlib.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
 
#ifndef __COMPACT_TEMPLATE_CUH__
#define __COMPACT_TEMPLATE_CUH__
#include "reduction_template.cuh"
int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)


int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	
	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}
	

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) { 
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	} 
	
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length);
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length);

void scanLargeDeviceArray(int *d_out, int *d_in, int length) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length) 
{
	int powerOfTwo = nextPowerOfTwo(length);
	prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length) 
{
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);


	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

void scan_host(int* h_out, int* h_in, int length)
{
	int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	cudaMalloc((void **)&d_out, arraySize);
	cudaMalloc((void **)&d_in, arraySize);
	cudaMemset(d_out, 0, arraySize);
	cudaMemcpy(d_in, h_in, arraySize, cudaMemcpyHostToDevice);
	
	if (length > ELEMENTS_PER_BLOCK) 
	{
		scanLargeDeviceArray(d_out, d_in, length);
	}
	else 
	{
		scanSmallDeviceArray(d_out, d_in, length);
	}

	cudaMemcpy(h_out, d_out, arraySize, cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	cudaFree(d_in);
}

void scan_device(int* d_out, int* d_in, int length)
{
	// int *d_out, *d_in;
	const int arraySize = length * sizeof(int);

	// cudaMalloc((void **)&d_out, arraySize);
	// cudaMalloc((void **)&d_in, arraySize);
	cudaMemset(d_out, 0, arraySize);
	// cudaMemcpy(d_in, h_in, arraySize, cudaMemcpyHostToDevice);
	
	if (length > ELEMENTS_PER_BLOCK) 
	{
		scanLargeDeviceArray(d_out, d_in, length);
	}
	else 
	{
		scanSmallDeviceArray(d_out, d_in, length);
	}

	// cudaMemcpy(h_out, d_out, arraySize, cudaMemcpyDeviceToHost);
	// cudaFree(d_out);
	// cudaFree(d_in);
}

template <class T1, class T2>
__global__ void compact_kernel(int* d_scanValid, int* d_Val, T1* d_isValid, T2 totalNum, 
							 int* d_ValValid)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid < totalNum && d_isValid[gid] == 1)
	{
		d_ValValid[d_scanValid[gid]] = d_Val[gid];
	}
}

void compact_t_device( int** h_ValValid, int *h_numValid, 
				int* d_Val, int* d_isValid, int totalNum)
{
	// int* d_Val;
	// int* d_isValid;
	// const int arraySize = totalNum * sizeof(int);
	// cudaMalloc((void **)&d_Val, arraySize);
	// cudaMalloc((void **)&d_isValid, arraySize);
	// cudaMemcpy(d_Val, h_Val, arraySize, cudaMemcpyHostToDevice);
	// cudaMemcpy(d_isValid, h_isValid, arraySize, cudaMemcpyHostToDevice);
	
	const int arraySize = totalNum * sizeof(int);
	int* d_reduceSum = NULL;

	if (*h_numValid == -1)
	{
		cudaMalloc((void**)&d_reduceSum, sizeof(int));
		cudaMemset(d_reduceSum, 0, sizeof(int));
		reduction_t(d_reduceSum, d_isValid, totalNum, (totalNum + 1024 - 1) / 1024, 1024);
		cudaMemcpy(h_numValid, d_reduceSum, sizeof(int), cudaMemcpyDeviceToHost);
	}

	const int validArraySize = (*h_numValid) * sizeof(int);
	*h_ValValid = (int*)malloc(validArraySize);
	int* d_ValValid;
	cudaMalloc((void **)&d_ValValid, validArraySize);
	
	// int* h_scanValid = (int*)malloc(arraySize);
	int* d_scanValid;
	cudaMalloc((void **)&d_scanValid, arraySize);
	scan_device(d_scanValid, d_isValid, totalNum);

	// cudaMemcpy(d_scanValid, h_scanValid, arraySize, cudaMemcpyHostToDevice);
	compact_kernel<< <(totalNum + 1024 - 1) / 1024, 1024>> >(d_scanValid, d_Val, d_isValid, totalNum, d_ValValid);
	cudaMemcpy(*h_ValValid, d_ValValid, sizeof(int)* (*h_numValid), cudaMemcpyDeviceToHost);


	// cudaFree(d_Val);
	// cudaFree(d_isValid);
	if (d_reduceSum!=NULL)
		cudaFree(d_reduceSum);
	cudaFree(d_ValValid);
	cudaFree(d_scanValid);
	// free(h_scanValid);

}

void compact_t_host( int** h_ValValid, int *h_numValid, 
				int* h_Val, int* h_isValid, int totalNum)
{
	int* d_Val;
	int* d_isValid;
	const int arraySize = totalNum * sizeof(int);
	cudaMalloc((void **)&d_Val, arraySize);
	cudaMalloc((void **)&d_isValid, arraySize);
	cudaMemcpy(d_Val, h_Val, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_isValid, h_isValid, arraySize, cudaMemcpyHostToDevice);
	
	int* d_reduceSum = NULL;
	cudaMalloc((void**)&d_reduceSum, sizeof(int));
	cudaMemset(d_reduceSum, 0, sizeof(int));
	reduction_t(d_reduceSum, d_isValid, totalNum, (totalNum + 1024 - 1) / 1024, 1024);
	cudaMemcpy(h_numValid, d_reduceSum, sizeof(int), cudaMemcpyDeviceToHost);


	const int validArraySize = (*h_numValid) * sizeof(int);
	*h_ValValid = (int*)malloc(validArraySize);
	int* d_ValValid;
	cudaMalloc((void **)&d_ValValid, validArraySize);
	
	int* h_scanValid = (int*)malloc(arraySize);
	scan_host(h_scanValid, h_isValid, totalNum);
	int* d_scanValid;
	cudaMalloc((void **)&d_scanValid, arraySize);
	cudaMemcpy(d_scanValid, h_scanValid, arraySize, cudaMemcpyHostToDevice);
	compact_kernel<< <(totalNum + 1024 - 1) / 1024, 1024>> >(d_scanValid, d_Val, d_isValid, totalNum, d_ValValid);
	cudaMemcpy(*h_ValValid, d_ValValid, sizeof(int)* (*h_numValid), cudaMemcpyDeviceToHost);


	cudaFree(d_Val);
	cudaFree(d_isValid);
	cudaFree(d_reduceSum);
	cudaFree(d_ValValid);
	cudaFree(d_scanValid);
	free(h_scanValid);

}


// int main()
// {
// 	const int N = 10;
// 	time_t t;
// 	srand((unsigned)time(&t));
// 	int *h_isValid = new int[N];
// 	for (int i = 0; i < N; i++) 
// 	{
// 		h_isValid[i] = rand() % 2;
// 	}
// 	int *h_Val = new int[N];
// 	for (int i = 0; i < N; i++) 
// 	{
// 		h_Val[i] = rand() % 10;
// 	}

// 	int* h_ValValid = NULL;
// 	int h_numValid = 0;
// 	compact_t_host(&h_ValValid, &h_numValid, h_Val, h_isValid, N);

// 	// int *outGPU = new int[N]();
// 	// scan(outGPU, in, N);
// 	cout<<"h_Val"<<endl;
// 	for (int i = 0; i < N; i++) 
// 	{
// 		cout<<h_Val[i]<<"\t";
// 	}
// 	cout<<endl;
// 	cout<<"h_isValid"<<endl;
// 	for (int i = 0; i < N; i++) 
// 	{
// 		cout<<h_isValid[i]<<"\t";
// 	}
// 	cout<<endl;
// 	cout<<"h_ValValid"<<endl;
// 	for (int i = 0; i < h_numValid; i++) 
// 	{
// 		cout<<h_ValValid[i]<<"\t";
// 	}
// 	cout<<endl;


// 	delete []h_isValid;
// 	delete []h_Val;
// 	free(h_ValValid);

// 	return 0;
// }


#endif