#ifndef LOCALMID_H
#define LOCALMID_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "utils.h"

template<class DataType1,class Oper>
__global__ void G_Reclass(DataType1 *input, int width, int height, DataType1 *oldValueSet, DataType1 *newValueSet, int length,Oper op);


template<class DataType1,class OperType>
void ReClass(DataType1* input, int width, int height, DataType1* oldValueSet, DataType1* newValueSet, int length)
{
	DataType1* d_input;

	DataType1* d_oldValueSet;
	DataType1* d_newValueSet;

	checkCudaErrors(cudaMalloc(&d_input, sizeof(DataType1)*width*height));

	checkCudaErrors(cudaMalloc(&d_oldValueSet, sizeof(DataType1)*length));
	checkCudaErrors(cudaMalloc(&d_newValueSet, sizeof(DataType1)*length));

	checkCudaErrors(cudaMemcpy(d_input, input, sizeof(DataType1)*width*height, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_oldValueSet, oldValueSet, sizeof(DataType1)*length, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_newValueSet, newValueSet, sizeof(DataType1)*length, cudaMemcpyHostToDevice));

	// dim3 block = CuEnvControl::getBlock2D();
	// dim3 grid = CuEnvControl::getGrid(width, height);

	dim3 block(512,1);
	dim3 grid((width+511)/512, (height+511/512));
	G_Reclass<DataType1,OperType> << <grid, block>> >(d_input, width, height, d_oldValueSet, d_newValueSet, length, OperType());

	checkCudaErrors(cudaMemcpy(input, d_input, sizeof(DataType1)*width*height, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_oldValueSet));
	checkCudaErrors(cudaFree(d_newValueSet));

}















#endif