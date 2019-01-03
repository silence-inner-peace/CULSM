#ifndef __ABSTRACT_CAL_FOR_EACH_PIXEL__
#define __ABSTRACT_CAL_FOR_EACH_PIXEL__
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
// #include "basestruct.h"
#include "AbstractBlockClass.h"

#include "reduction_template.cuh"
#include "compact_template.cuh"
#include "Array2D.h"
#include "Array2D_CUDA.h"
#include "cuCCL.cuh"
#include "localDevice.cuh"


/*--------------------------------------------------------------------------------------
计算单个像元的值，以便于以后计算整个连通域
比如单个像元对整个连通域的周长贡献，计算方法由calEachPixel控制,AbstractCalforEachPixel是基类，继承得到具体的计算
input:
	int* h_inputHoloUp		//提供当前分块的 上边界
	int* h_inputHoloDown	//提供当前分块的 下边界
	cuCCLClass *ccl 		//提供当前分块的 原始值，标记值，width,height

output:
	Array2D< Cutype<int> >* d_outputOnePixel;	//保存当前分块中单个像元中的metrics值
--------------------------------------------------------------------------------------*/

class AbstractCalforEachPixel
{
public:
	AbstractCalforEachPixel(int* h_inputHoloUp,
							int* h_inputHoloDown,
							cuCCLClass *ccl);
	virtual void calEachPixel(dim3 grid,dim3 block) = 0;
	~AbstractCalforEachPixel();
public:
	Array2D< Cutype<int> >* d_inputHoloUp;	
	Array2D< Cutype<int> >* d_inputHoloDown;	
	cuCCLClass *m_cclObj;	

	Array2D< Cutype<int> >* d_outputOnePixel;	
};

AbstractCalforEachPixel::AbstractCalforEachPixel(int* h_inputHoloUp,
						 int* h_inputHoloDown,
						 cuCCLClass* cclObj)
{
	int width = cclObj->width;
	int height = cclObj->task_height;
	// d_inputHoloUp = new Array2D< Cutype<int> >(h_inputHoloUp, 1, width);
	// d_inputHoloDown = new Array2D< Cutype<int> >(h_inputHoloDown, 1, width);
	if(h_inputHoloUp!=NULL)
	{
		d_inputHoloUp = new Array2D< Cutype<int> >(h_inputHoloUp, 1, width);
	}
	else
	{
		d_inputHoloUp = NULL;
	}
	if(h_inputHoloDown!=NULL)
	{
		d_inputHoloDown = new Array2D< Cutype<int> >(h_inputHoloDown, 1, width);
	}
	else
	{
		d_inputHoloDown = NULL;
	}

	m_cclObj = cclObj;
    d_outputOnePixel = new Array2D< Cutype<int> >(height,width);
}
AbstractCalforEachPixel::~AbstractCalforEachPixel()
{
	if(NULL != d_inputHoloUp)
	{
		delete d_inputHoloUp;
		d_inputHoloUp = NULL;
	}
	if(NULL != d_inputHoloDown)
	{
		delete d_inputHoloDown;
		d_inputHoloDown = NULL;
	}
	if (d_outputOnePixel!=NULL)
	{
		delete d_outputOnePixel;
		d_outputOnePixel = NULL;
	}
}
#endif //__ABSTRACT_CAL_FOR_EACH_PIXEL__

