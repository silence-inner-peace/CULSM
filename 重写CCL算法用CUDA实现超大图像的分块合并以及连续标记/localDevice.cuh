#ifndef LOCALDEVICE_H
#define LOCALDEVICE_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

template<class DataType1>
class RelabelValueUpdate
{
public:
	__device__ DataType1 operator()(DataType1 value, DataType1* oldValueSet, DataType1* newValueSet, int length)
	{
		for (int idx = 0; idx < length; idx++)
		{
			if (abs(oldValueSet[idx] - value) < 1e-6)
			{
				return newValueSet[idx];
			}
		}
		return value;
		// return 1;
	}
};

template<class DataType1,class Oper>
__global__ void G_Reclass(DataType1 *input, int width, int height, DataType1 *oldValueSet, DataType1 *newValueSet, int length, Oper op)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (x_idx >= width)
		return;
	int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (y_idx >= height)
		return;
	int gid = y_idx * width + x_idx;
	int val = input[gid];
	input[gid] = op(val, oldValueSet, newValueSet, length);

}










#endif