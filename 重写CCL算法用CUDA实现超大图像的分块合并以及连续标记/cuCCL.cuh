#ifndef __CUCCL_CUH__
#define __CUCCL_CUH__
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "timer.h"

#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "AbstractBlockClass.h"
#include "Array2D.h"
#include "Array2D_CUDA.h"
#include "GlobalConfiguration.h"
#include "singleton.h"

// #define _DEBUG

#define NO_USE_CLASS -1

__device__ int find(int * localLabel, int p)
{
	if (localLabel[p] != -1)
	{
		while (p != localLabel[p])
		{
			p = localLabel[p];
		}
		return p;
	}
	else
		return -1;
}
__device__ void findAndUnion(int* buf, int g1, int g2) {
	bool done;
	do {

		g1 = find(buf, g1);
		g2 = find(buf, g2);

		// it should hold that g1 == buf[g1] and g2 == buf[g2] now

		if (g1 < g2) {
			int old = atomicMin(&buf[g2], g1);
			done = (old == g2);
			g2 = old;
		}
		else if (g2 < g1) {
			int old = atomicMin(&buf[g1], g2);
			done = (old == g1);
			g1 = old;
		}
		else {
			done = true;
		}

	} while (!done);
}

__global__ void setUsefulClass(int* devSrcData, int * devLabelMap, int width, int task_height, bool* d_use)//
{
	int tid = threadIdx.x;

	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;

	bool limits = x < width && y < task_height;
	int id = x + y * width;
/////////////////////////////////////////////////////////////////////////////////////////////
	__shared__ bool shared_USE[NUM_CLASSES];

	// // 初始化共享内存数组
	if(id < NUM_CLASSES)
	{
		shared_USE[id] = d_use[id];
	}
	__syncthreads();
/////////////////////////////////////////////////////////////////////////////////////////////
	if (limits)
	{
		int focusP = devSrcData[id];
		if(!d_use[focusP])
		{
			devSrcData[id] = NO_USE_CLASS;//如果这个类型不参与计算，将其赋值为-1
			devLabelMap[id] = NO_USE_CLASS;
		}
	}
}

__global__ void gpuLineLocal(int* devSrcData, int * devLabelMap, int width, int task_height, int nodata)
{
	int tid = threadIdx.x;

	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;

	bool limits = x < width && y < task_height;
	int id = x + y * width;


	__shared__ int localLabel[32 * 16];


	if (limits)
	{
		localLabel[tid] = tid;
		__syncthreads();

		int focusP = devSrcData[x + y * width];
		if(focusP != NO_USE_CLASS)
		{
			if (focusP != nodata && threadIdx.x > 0 && focusP == devSrcData[x - 1 + y * width])
				localLabel[tid] = localLabel[tid - 1];
			__syncthreads();

			int buf = tid;

			while (buf != localLabel[buf])
			{
				buf = localLabel[buf];
				localLabel[tid] = buf;
			}

			int globalL = (blockIdx.x * blockDim.x + buf) + (blockIdx.y) * width;
			devLabelMap[id] = globalL;

			if (focusP == nodata)
				devLabelMap[id] = -1;
		}
	}

}

__global__ void gpuLineUfGlobal(int* devSrcData, int * devLabelMap, int width, int task_height, int nodata, bool use_diags)//
{
	int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;

	if (in_limits)
	{
		int center = devSrcData[gid];
		if (center != nodata && center != NO_USE_CLASS)
		{

			if(use_diags)	//8
			{
				if (x > 0 && threadIdx.x == 0)//&& center == left
				{
					if (center == devSrcData[x - 1 + y * width])
						findAndUnion(devLabelMap, gid, x - 1 + y * width); // left
				}
				if (y > 0 && threadIdx.y == 0)//&& center == up 
				{
					if (center == devSrcData[x + (y - 1) * width])
						findAndUnion(devLabelMap, gid, x + (y - 1) * width); // up
				}
				if (y > 0 && x > 0 && threadIdx.y == 0)// && center == leftup
				{
					if (center == devSrcData[x - 1 + (y - 1) * width])
						findAndUnion(devLabelMap, gid, x - 1 + (y - 1) * width); // up-left
				}
				if (y > 0 && x < width - 1 && threadIdx.y == 0)// && center == rightup
				{
					if (center == devSrcData[x + 1 + (y - 1) * width])
						findAndUnion(devLabelMap, gid, x + 1 + (y - 1) * width); // up-right
				}
			}
			else	//4
			{
				// search neighbour, left and up
				if (x > 0 && threadIdx.x == 0 && center == devSrcData[x - 1 + y * width])
					findAndUnion(devLabelMap, gid, x - 1 + y * width); // left
				if (y > 0 && threadIdx.y == 0 && center == devSrcData[x + (y - 1) * width])
					findAndUnion(devLabelMap, gid, x + (y - 1) * width); // up	
			}
		}
	}
}

__global__ void gpuLineUfFinal(int * labelMap, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int gid = x + y * width;
	
	bool limits = x < width && y < task_height;
	
	if (limits)
	{
		int center = labelMap[gid];
		if(center != NO_USE_CLASS)
		{
			labelMap[gid] = find(labelMap, gid);
		}
	}
}


//这个就是用来生成连通域，并将结果保存在设备端，供指数计算类调用，不应该有其他功能
//调用gpuLineUF之后就得到标记数组devLabelMap
class cuCCLClass
{
public:
	Array2D< Cutype<int> >* devSrcData;		//source
	Array2D< Cutype<int> >* devLabelMap;	//label
	GlobalConfiguration G_Config;
	
	int width;
	int task_height;
	int nodata;	

public:
	cuCCLClass(	int* hostSrcData, 
				int width,
				int task_height,
				int nodata);
	void gpuLineUF(dim3 blockSize, dim3 gridSize);
	~cuCCLClass();
};

cuCCLClass::cuCCLClass(	int* hostSrcData, 
						int _width,
						int _task_height,
						int _nodata)
{
	devSrcData = new Array2D< Cutype<int> >(hostSrcData, _task_height, _width);
	devLabelMap = new Array2D< Cutype<int> >(_task_height,_width);
	
	G_Config = Singleton<GlobalConfiguration>::Instance();

	width = _width;
	task_height = _task_height;
	nodata = _nodata;
}

void cuCCLClass::gpuLineUF(dim3 blockSize, dim3 gridSize)
{
	const int blockSizeX = blockSize.x * blockSize.y;
	const int blockSizeY = 1;
	dim3 blockSizeLine(blockSizeX, blockSizeY, 1);
	dim3 gridSizeLine((width + blockSizeX - 1) / blockSizeX, (task_height + blockSizeY - 1) / blockSizeY, 1);
	
	setUsefulClass<< < gridSizeLine, blockSizeLine >> >(devSrcData->getDevData(), devLabelMap->getDevData(), width, task_height, G_Config.d_USE->getDevData());//
	
	#ifdef _DEBUG
	devSrcData->show();
	#endif
	
	gpuLineLocal<< < gridSizeLine, blockSizeLine >> > (devSrcData->getDevData(), devLabelMap->getDevData(), width, task_height, nodata);
	
	#ifdef _DEBUG
	devLabelMap->show();
	#endif
	
	gpuLineUfGlobal<< < gridSizeLine, blockSizeLine >> > (devSrcData->getDevData(), devLabelMap->getDevData(), width, task_height, nodata,G_Config.USE_DIAGS);//
	
	#ifdef _DEBUG
	devLabelMap->show();
	#endif
	
	gpuLineUfFinal<< < gridSizeLine, blockSizeLine >> > (devLabelMap->getDevData(), width, task_height);
	
	#ifdef _DEBUG
	devLabelMap->show();
	#endif
	
	//进行完以上三个步骤，计算后的连通域就存储在devLabelMap中
}

cuCCLClass::~cuCCLClass()
{
	if(NULL != devSrcData)
	{
		delete devSrcData;
		devSrcData = NULL;
	}
	if(NULL != devLabelMap)
	{
		delete devLabelMap;
		devLabelMap = NULL;
	}
}
#endif	//__CUCCL_CUH__


// int main(int argc, char const *argv[])
// {
// 	int array[25] = { 1, 3, 3, 3, 3,
// 					  1, 3, 3, 1, 3,
// 					  1, 2, 1, 3, 2,
// 					  2, 1, 3, 2, 3,
// 					  1, 2, 2, 3, 2 };

// 	int* srcTest = new int[25];
// 	for (int i = 0; i < 25; i++)
// 	{
// 		srcTest[i] = array[i];
// 	}
// 	PRead *pread = new PRead(5, 5, 0);

// 	bool useful_class[10] = {1,1,1,1,1,1,1,1,1,1};


// 	//初始化CUDA
// 	int gpuIdx = 1;//设置计算能力大于3.5的GPU
// 	initCUDA(gpuIdx);
	
// 	//所有关于GPU显存的初始化都要放在initCUDA之后进行，否则会出现随机值
// 	GlobalConfiguration& config = Singleton<GlobalConfiguration>::Instance();
// 	config.set_USE(useful_class);
// 	config.set_USE_DIAGS(true);

// 	int width = pread->cols();
// 	int height = pread->rows();
// 	int nodata = (int)pread->invalidValue();
// 	// int blockNum = getDevideInfo(width, height, nodata, dataBlockArray);
// 	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
// 	dim3 blockDim1(blockSize.x, blockSize.y, 1);
// 	dim3 gridDim1((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);

// 	cuCCLClass* ccl = new cuCCLClass(srcTest, width, height, nodata);
// 	ccl->gpuLineUF(blockDim1,gridDim1);
// 	ccl->devLabelMap->show();
// 	delete ccl;
// 	return 0;
// }