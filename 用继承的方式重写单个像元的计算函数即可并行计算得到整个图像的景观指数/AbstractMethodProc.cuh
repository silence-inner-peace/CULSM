#ifndef __ABSTRACT_METHOD_PROC__
#define __ABSTRACT_METHOD_PROC__


// #define _DEBUG
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "timer.h"

#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
// #include "utils.h"

// #include "basestruct.h"
#include "derivedPeriBlockClass.h"

#include "reduction_template.cuh"
#include "compact_template.cuh"
#include "Array2D.h"
#include "Array2D_CUDA.h"
#include "cuCCL.cuh"
#include "localDevice.cuh"
#include "calPeriForEachPixel.cuh"
#include "calPeriMetrics.cuh"



__global__ void getIsValid(int* d_label, int* d_isValid, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		if (d_label[gid] == gid)
		{
			d_isValid[gid] = 1;
		}
	}
}
__global__ void updateDevLabel(int * dev_labelMap, int labelStart, int task_height, int width)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;
	bool limits = x < width && y < task_height;
	if (limits)
	{
		dev_labelMap[gid] += labelStart;
	}
}

/*--------------------------------------------------------------------------------------
this class define the routine of calculate metrics;
step1: load data, save the source data in -----cuCCLClass: devSrcData;
step2: run ccl algorithm, save label result in -----cuCCLClass: devLabelMap;
step3: calculate for each pixel, save the result in ------AbstractCalforEachPixel: d_outputOnePixel;
step4: statistic to get sparse metrics matrics, save in -----AbstractCalMetrics: d_outputSparse;
step5: compress the sparse matrics for output, save in -----TemplateMethodProc: mh_valValid,mnCurPatchNum

--------------------------------------------------------------------------------------*/


template<class PatchType, class valType ,class CalforEachPixelOper, class CalMetricsOper>
class TemplateMethodProc
{
public:	//mid variabl
	cuCCLClass *mpCclObj;
	AbstractCalforEachPixel *mpAbsForEachPix;
	AbstractCalMetrics *mpAbsCalMetrics;

public:
	TemplateMethodProc( CuLSM::dataBlock* _curBlock,
						CGDALRead* pread,
						// PRead* pread,int* src,
						dim3 grid,dim3 block);
	
	void templateMethod(CuLSM::dataBlock* _curBlock,
						map<int,PatchType>& QufRootMap,
						dim3 grid, dim3 block,
						int** h_curBlockMetricsResult);
	
	//step 5
	void compactMethod(CuLSM::dataBlock* _curBlock,
						int* d_outputSparseMetrics,
						map<int,PatchType>& QufRootMap,
						dim3 grid, dim3 block,
						int** h_curBlockMetricsResult);
	
	// virtual void output2Vector(PatchType& patch, valType val) = 0;//这个函数需要用户重写

	~TemplateMethodProc();
	
};

template<class PatchType, class valType ,class CalforEachPixelOper, class CalMetricsOper>
TemplateMethodProc<PatchType,valType,CalforEachPixelOper,CalMetricsOper>::TemplateMethodProc
										( CuLSM::dataBlock* _curBlock,
										CGDALRead* pread,
										// PRead* pread,int* src,
										dim3 grid,dim3 block)
{
	//step1
	_curBlock->loadBlockData(pread);
	// _curBlock->loadBlockData(pread,src);
	
	int* hostSrcData = _curBlock->mh_SubData;
	int _width = _curBlock->mnWidth;
	int _task_height = _curBlock->mnSubTaskHeight;
	int _nodata = _curBlock->mnNodata;

	//step2
	mpCclObj =  new cuCCLClass(hostSrcData, _width, _task_height, _nodata);
	mpCclObj->gpuLineUF(block,grid);

	mpAbsForEachPix = NULL;
	mpAbsCalMetrics = NULL;
}

template<class PatchType, class valType ,class CalforEachPixelOper, class CalMetricsOper>
TemplateMethodProc<PatchType,valType,CalforEachPixelOper,CalMetricsOper>::~TemplateMethodProc()
{
	delete mpCclObj;
	mpCclObj = NULL;
	delete mpAbsForEachPix;
	mpAbsForEachPix = NULL;
	delete mpAbsCalMetrics;
	mpAbsCalMetrics = NULL;
}


template<class PatchType, class valType ,class CalforEachPixelOper, class CalMetricsOper>
void TemplateMethodProc<PatchType,valType,CalforEachPixelOper,CalMetricsOper>::
						compactMethod(CuLSM::dataBlock* _curBlock,
										int* d_outputSparseMetrics,
										map<int,PatchType>& QufRootMap,
										dim3 grid, dim3 block,
										int** h_curBlockMetricsResult)
{
	int _width = mpCclObj->width;
	int _taskHeight = mpCclObj->task_height;
	int _nBytes_task = sizeof(int) * _width * _taskHeight;
	const int numElements = _width * _taskHeight;
	int* d_outputLabelOfSubData = mpCclObj->devLabelMap->getDevData();
	int* d_inputSrcSubData = mpCclObj->devSrcData->getDevData();
	
	Array2D< Cutype<int> >* d_IsValid = new Array2D< Cutype<int> >(_taskHeight,_width);
	int* d_IsValidData = d_IsValid->getDevData();
	getIsValid << <grid,block >> >(d_outputLabelOfSubData, d_IsValidData, _width, _taskHeight);
	updateDevLabel << <grid,block >> > (d_outputLabelOfSubData, _curBlock->mnStartTag, _taskHeight, _width);
	
	#ifdef _DEBUG
	mpCclObj->devLabelMap->show();
	d_IsValid->show();
	#endif

	_curBlock->mh_LabelVal= (int*)malloc(_nBytes_task);
	checkCudaErrors(cudaMemcpy(_curBlock->mh_LabelVal, d_outputLabelOfSubData, _nBytes_task, cudaMemcpyDeviceToHost));
	_curBlock->mh_curPatchNum = -1;
	compact_t_device(&(_curBlock->mh_compactLabel), &(_curBlock->mh_curPatchNum), d_outputLabelOfSubData, d_IsValidData, numElements);
	compact_t_device(&(_curBlock->mh_compactSrc), &(_curBlock->mh_curPatchNum), d_inputSrcSubData, d_IsValidData, numElements);
	compact_t_device(h_curBlockMetricsResult, &(_curBlock->mh_curPatchNum), d_outputSparseMetrics, d_IsValidData, numElements);
	
	cout << "h_outputNumOfValidElements: " << _curBlock->mh_curPatchNum << endl;

	for (int i = 0; i < _curBlock->mh_curPatchNum; i++)
	{	
		PatchType temp;
		temp.nLabel = _curBlock->mh_compactLabel[i];
		temp.nType = _curBlock->mh_compactSrc[i];
		int tempVal = (*h_curBlockMetricsResult)[i];
		
		//这一行需要用户写，怎么样通过传字符串动态的选择变量进行赋值还不知道,需要实现如下功能
		// temp.nPerimeterByPixel = (*h_curBlockMetricsResult)[i];
		
		// output2Vector(temp, tempVal);
		temp.nMetricsByPixel = tempVal;


		temp.isUseful = false;
		QufRootMap.insert(make_pair(temp.nLabel, temp));
	}

	if (d_IsValid!=NULL)
	{
		delete d_IsValid;
		d_IsValid = NULL;
	}	
}


template<class PatchType, class valType ,class CalforEachPixelOper, class CalMetricsOper>
void TemplateMethodProc<PatchType,valType,CalforEachPixelOper,CalMetricsOper>::templateMethod(
										CuLSM::dataBlock* _curBlock,
										map<int,PatchType>& QufRootMap,
										dim3 grid, dim3 block,
										int** h_curBlockMetricsResult)
{
	// local variable
	int* h_inputHoloUp = _curBlock->mh_holoUp;
	int* h_inputHoloDown = _curBlock->mh_holoDown;
	int* d_outputLabelOfSubData = mpCclObj->devLabelMap->getDevData();
	int* d_inputSrcSubData = mpCclObj->devSrcData->getDevData();
	
	#ifdef _DEBUG
	mpCclObj->devSrcData->show();
	mpCclObj->devLabelMap->show();
	#endif
	//step3
	mpAbsForEachPix = new CalforEachPixelOper(h_inputHoloUp,h_inputHoloDown,mpCclObj);
	mpAbsForEachPix->calEachPixel(grid,block);
	#ifdef _DEBUG
	mpAbsForEachPix->d_outputOnePixel->show();
	#endif

	//step4
	mpAbsCalMetrics = new CalMetricsOper(mpAbsForEachPix);
	mpAbsCalMetrics->calMetrics(grid,block);
	#ifdef _DEBUG
	mpAbsCalMetrics->d_outputSparse->show();
	#endif

	int *d_outputSparseMetrics = mpAbsCalMetrics->d_outputSparse->getDevData();

	//step5
	compactMethod(_curBlock, d_outputSparseMetrics, QufRootMap,grid,block, h_curBlockMetricsResult);

}

#endif //__ABSTRACT_METHOD_PROC__
