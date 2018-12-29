#ifndef __CAL_PERI_METRICS_CUH__
#define __CAL_PERI_METRICS_CUH__
#include "AbstractCalMetrics.cuh"


__global__ void getPeri(int* dOutPeri, int* dev_labelMap, int *dev_pixelPerimeter, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		int regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			atomicAdd(dOutPeri + regLabel, dev_pixelPerimeter[gid]);
		}
	}
}
__global__ void getArea(int* dOutArea, int* dev_labelMap, int *dev_pixelArea, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		int regLabel = dev_labelMap[gid];//get labeled val,if the labled value != -1 than calculate its area and primeter ;
		if (regLabel >= 0)
		{
			atomicAdd(dOutArea + regLabel, dev_pixelArea[gid]);
		}
	}
}
class calPeriMetrics:public AbstractCalMetrics
{
public:
	calPeriMetrics(AbstractCalforEachPixel* _AbsForEachPix):AbstractCalMetrics(_AbsForEachPix){}
	void calMetrics(dim3 grid,dim3 block)
	{
		int _width = pAbsForEachPix->m_cclObj->width;
		int _taskHeight = pAbsForEachPix->m_cclObj->task_height;	
		int* d_label = pAbsForEachPix->m_cclObj->devLabelMap->getDevData();
		int* d_src = pAbsForEachPix->m_cclObj->devSrcData->getDevData();
		int* d_onePixel = pAbsForEachPix->d_outputOnePixel->getDevData();

		int* d_outputSparsePeri = d_outputSparse->getDevData();

		getPeri << <grid, block >> >(d_outputSparsePeri, d_label, d_onePixel, _width, _taskHeight);

	}
};

class calAreaMetrics:public AbstractCalMetrics
{
public:
	calAreaMetrics(AbstractCalforEachPixel* _AbsForEachPix):AbstractCalMetrics(_AbsForEachPix){}
	void calMetrics(dim3 grid,dim3 block)
	{
		int _width = pAbsForEachPix->m_cclObj->width;
		int _taskHeight = pAbsForEachPix->m_cclObj->task_height;	
		int* d_label = pAbsForEachPix->m_cclObj->devLabelMap->getDevData();
		int* d_src = pAbsForEachPix->m_cclObj->devSrcData->getDevData();
		int* d_onePixel = pAbsForEachPix->d_outputOnePixel->getDevData();

		int* d_outputSparsePeri = d_outputSparse->getDevData();

		getArea << <grid, block >> >(d_outputSparsePeri, d_label, d_onePixel, _width, _taskHeight);

	}
};

#endif //__CAL_PERI_METRICS_CUH__
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
// 	calPeriForEachPixel* calPeriOnePixelProc = new calPeriForEachPixel(h_holoup,h_holodown,ccl);
// 	calPeriOnePixelProc->calEachPixel(blockDim1,gridDim1);
// 	calPeriOnePixelProc->d_outputOnePixel->show();

// 	calPeriMetrics *calPeriProc = new calPeriMetrics(calPeriOnePixelProc);
// 	calPeriProc->calMetrics(blockDim1,gridDim1);
// 	calPeriProc->d_outputSparse->show();

// 	delete ccl;
// 	delete calPeriOnePixelProc;
// 	delete calPeriProc;
// 	return 0;
// }