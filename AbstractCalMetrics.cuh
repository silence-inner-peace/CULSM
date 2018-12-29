#ifndef __ABSTRACT_CAL_METRICS__
#define __ABSTRACT_CAL_METRICS__

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
#include "calPeriForEachPixel.cuh"

class AbstractCalMetrics
{
public:
	AbstractCalforEachPixel* pAbsForEachPix;
	Array2D< Cutype<int> >* d_outputSparse;	
public:
	AbstractCalMetrics( AbstractCalforEachPixel* _AbsForEachPix)
	{
		pAbsForEachPix = _AbsForEachPix;
	   	d_outputSparse = new Array2D< Cutype<int> >(pAbsForEachPix->m_cclObj->task_height,
    												pAbsForEachPix->m_cclObj->width);
	}

	virtual void calMetrics(dim3 grid,dim3 block) = 0;

	~AbstractCalMetrics()
	{
		if(d_outputSparse!=NULL)
		{
			delete d_outputSparse;
			d_outputSparse = NULL;
		}
	}
	
};




#endif //__ABSTRACT_CAL_METRICS__