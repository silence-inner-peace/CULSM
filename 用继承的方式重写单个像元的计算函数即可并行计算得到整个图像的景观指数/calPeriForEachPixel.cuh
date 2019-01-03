#ifndef __CAL_PERI_FOR_EACH_PIXEL_CUH_
#define __CAL_PERI_FOR_EACH_PIXEL_CUH_
#include "AbstractCalforEachPixel.cuh"


__global__ void getEachPixelPerimeter(int *d_taskData, int *d_holoUp, int *d_holoDown, int* d_PixelPerimeter, int task_height, int width)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	// int x = threadIdx.x + blockDim.y * blockDim.x * blockIdx.x;
	// int y = blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;
	if (in_limits)
	{
		int center = d_taskData[gid];
		if (x == 0)
		{
			d_PixelPerimeter[gid] += 1;
		}
		if (x == width - 1)
		{
			d_PixelPerimeter[gid] += 1;
		}
		if (x>0)
		{
			if (center != d_taskData[gid - 1])
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		if (x < width - 1)
		{
			if (center != d_taskData[gid + 1])
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		if (y>0)
		{
			if (center != d_taskData[gid - width])
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		if (y < task_height - 1)
		{
			if (center != d_taskData[gid + width])//down
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		if (d_holoUp == NULL && d_holoDown == NULL)		//不用分块
		{
			if (y == 0)
			{
				d_PixelPerimeter[gid] += 1;
			}
			if (y == task_height - 1)
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		else if (d_holoUp == NULL && d_holoDown != NULL)//第一个分块
		{
			if (y == 0)
			{
				d_PixelPerimeter[gid] += 1;
			}
			if (y == task_height - 1)
			{
				if (center != d_holoDown[x])
				{
					d_PixelPerimeter[gid] += 1;
				}
			}
		}
		else if (d_holoUp != NULL && d_holoDown == NULL)//最后一个分块
		{
			if (y == 0)
			{
				if (center != d_holoUp[x])
				{
					d_PixelPerimeter[gid] += 1;
				}
			}
			if (y == task_height - 1)
			{
				d_PixelPerimeter[gid] += 1;
			}
		}
		else	//中间的分块
		{
			if (y == 0)
			{
				if (center != d_holoUp[x])
				{
					d_PixelPerimeter[gid] += 1;
				}
			}
			if (y == task_height - 1)
			{
				if (center != d_holoDown[x])
				{
					d_PixelPerimeter[gid] += 1;
				}
			}
		}
	}
}
__global__ void getEachPixelArea(int* d_PixelArea, int task_height, int width)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int gid = x + y * width;
	bool in_limits = x < width && y < task_height;
	if (in_limits)
	{
		d_PixelArea[gid] = 1;
	}
}
class calPeriForEachPixel : public AbstractCalforEachPixel
{
public:
	calPeriForEachPixel(int* h_inputHoloUp,
					int* h_inputHoloDown,
					cuCCLClass *ccl):AbstractCalforEachPixel(h_inputHoloUp,h_inputHoloDown,ccl){}
	void calEachPixel(dim3 grid,dim3 block)
	{
		//kernel ----calculate perimeter for each pixel
		int _width = m_cclObj->width;
		int _taskHeight = m_cclObj->task_height;

		if(d_inputHoloUp!=NULL&&d_inputHoloDown!=NULL)
		{			
		getEachPixelPerimeter << <grid, block >> >(m_cclObj->devSrcData->getDevData(),
												   d_inputHoloUp->getDevData(), 
												   d_inputHoloDown->getDevData(), 
												   d_outputOnePixel->getDevData(),
												   _taskHeight, 
												   _width);
		}
		else if(d_inputHoloUp==NULL&&d_inputHoloDown!=NULL)
		{
			getEachPixelPerimeter << <grid, block >> >(m_cclObj->devSrcData->getDevData(),
												   NULL, 
												   d_inputHoloDown->getDevData(), 
												   d_outputOnePixel->getDevData(),
												   _taskHeight, 
												   _width);		
		}
		else if(d_inputHoloUp!=NULL&&d_inputHoloDown==NULL)
		{
					getEachPixelPerimeter << <grid, block >> >(m_cclObj->devSrcData->getDevData(),
												   d_inputHoloUp->getDevData(), 
												   NULL, 
												   d_outputOnePixel->getDevData(),
												   _taskHeight, 
												   _width);
		}
		else
		{
		getEachPixelPerimeter << <grid, block >> >(m_cclObj->devSrcData->getDevData(),
												   NULL, 
												   NULL, 
												   d_outputOnePixel->getDevData(),
												   _taskHeight, 
												   _width);			
		}
	}
	
};

class calAreaForEachPixel : public AbstractCalforEachPixel
{
public:
	calAreaForEachPixel(int* h_inputHoloUp,
					int* h_inputHoloDown,
					cuCCLClass *ccl):AbstractCalforEachPixel(h_inputHoloUp,h_inputHoloDown,ccl){}
	void calEachPixel(dim3 grid,dim3 block)
	{
		//kernel ----calculate perimeter for each pixel
		int _width = m_cclObj->width;
		int _taskHeight = m_cclObj->task_height;

		getEachPixelArea << <grid, block >> >(d_outputOnePixel->getDevData(), 
												   _taskHeight, 
												   _width);
	}
	
};
#endif //__CAL_PERI_FOR_EACH_PIXEL_CUH_
// AbstractCalforEachPixel::calEachPixel()
// {
// 	int _width = m_cclObj->width;
// 	int _taskHeight = m_cclObj->task_height;

// 	G_OnePixel<int,int,PlusVal<int,int,int>,int>(d_outputOnePixel->getDevData(), m_cclObj->devLabelMap, _width, _taskHeight, PlusVal(), 1);
// }




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
// 	//初始化CUDA
// 	int gpuIdx = 1;//设置计算能力大于3.5的GPU
// 	initCUDA(gpuIdx);

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

// 	int* h_holoup=NULL;
// 	int* h_holodown=NULL;
// 	calPeriForEachPixel* calPeriProc = new calPeriForEachPixel(h_holoup,h_holodown,ccl);
// 	calPeriProc->calEachPixel(blockDim1,gridDim1);
// 	calPeriProc->d_outputOnePixel->show();

// 	delete ccl;
// 	delete calPeriProc;
// 	return 0;
// }