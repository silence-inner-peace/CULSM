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





//针对于求面积、最小外包矩形、斑块中心这一类已知每个删格单元值的删格操作
template<class DataType1, class DataTypeLabel, class DataTypeEachPixel>
class PlusVal	//get Area
{
public:
	__device__ DataType1 operator()(DataType1* dOut, DataTypeLabel regLabel, DataTypeEachPixel val)
	{
		atomicAdd(dOut + regLabel, val);//get area
	}
};


template<class DataType1, class DataTypeLabel, class DataTypeEachPixel>
class GetMax	//得到斑块的最小外包矩形
{
public:
	__device__ DataType1 operator()(DataType1* dOut, DataTypeLabel regLabel, DataTypeEachPixel val)
	{
		atomicMax(dOut + regLabel, val);//get boundBox
	}
};


template<class DataType1, class DataTypeLabel, class DataTypeEachPixel>
class GetMin	//得到斑块的最小外包矩形
{
public:
	__device__ DataType1 operator()(DataType1* dOut, DataTypeLabel regLabel, DataTypeEachPixel val)
	{
		atomicMin(dOut + regLabel, val);//get boundBox
	}
};


//example: getArea
//G_OnePixel(dOutPixNum, dev_labelMap, width, task_height, PlusVal, 1);
template<class DataTypeOut, class DataTypeLabel, class Oper, class DataTypeVal>
__global__ void G_OnePixel(DataTypeOut* dOut, DataTypeLabel* dev_labelMap, int width, int task_height, Oper op, DataTypeVal val)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		DataTypeLabel regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			op(dOut, regLabel, val);
		}

	}
}

//example: getPeri
//G_OnePixel(dOutPeri, dev_labelMap, dev_pixelPerimeter, width, task_height, PlusVal);
template<class DataTypeOut, class DataTypeLabel, class DataTypeEachPixel, class Oper>
__global__ void G_OnePixel(DataTypeOut* dOut, DataTypeLabel* dev_labelMap, DataTypeEachPixel *dev_pixelPerimeter, int width, int task_height, Oper op)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		DataTypeLabel regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		DataTypeEachPixel val = dev_pixelPerimeter[gid];
		if (regLabel >= 0)
		{
			op(dOut, regLabel, val);
		}

	}
}


//example: getBound_X
//G_Oper_X(dOutBound_XMin, dev_labelMap, width, task_height, GetMin);
//G_Oper_X(dOutBound_XMax, dev_labelMap, width, task_height, GetMax);
//
//example: getCenterX
//G_Oper_X(dOutXsum, dev_labelMap, width, task_height, PlusVal);
template<class DataType1, class DataTypeLabel, class Oper>
__global__ void G_Oper_X(DataType1* dOut, DataTypeLabel* dev_labelMap, int width, int task_height, Oper op)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		DataTypeLabel regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			op(dOut, regLabel, x);
		}
	}
}


//example: getBound_Y
//G_Oper_Y(dOutBound_YMin, dev_labelMap, width, task_height, GetMin);
//G_Oper_Y(dOutBound_YMax, dev_labelMap, width, task_height, GetMax);
//example: getCenterY
//G_Oper_Y<int,int, PlusVal<int,int,int>>(dOutYsum, dev_labelMap, width, task_height, PlusVal());
template<class DataType1, class DataTypeLabel, class Oper>
__global__ void G_Oper_Y(DataType1* dOut, DataTypeLabel* dev_labelMap, int width, int task_height, Oper op)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		DataTypeLabel regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			op(dOut, regLabel, y);
		}
	}
}










#endif