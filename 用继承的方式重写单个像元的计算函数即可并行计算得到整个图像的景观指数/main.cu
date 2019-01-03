// #include "PeriMethodProc.cuh"
#include "AbstractMethodProc.cuh"
#include "AbstractMerge.cuh"
#include "timer.h"


// #define _DEBUG
	
void initCUDA(int& devIdx)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  int dev = 0;
  if (devIdx < 0) dev = 0;
  if (devIdx > deviceCount - 1) dev = deviceCount - 1;
  else dev = devIdx;
  cudaSetDevice(dev);

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
  {
    printf("Using device %d:\n", dev);
    printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
      prop.name, (int)prop.totalGlobalMem, (int)prop.major,
      (int)prop.minor, (int)prop.clockRate);
  }
  if (prop.major < 2)
  {
    fprintf(stderr, "ERROR: CUDPP hash tables are only supported on "
      "devices with compute\n  capability 2.0 or greater; "
      "exiting.\n");
    exit(1);
  }
}


int main(int argc, char const *argv[])
{
	//初始化CUDA

	CpuTimer alltime;
	alltime.start();	
	int gpuIdx = 1;//设置计算能力大于3.5的GPU
	initCUDA(gpuIdx);

	AbstractMerge<  CuLSM::myPatch, 
					CuLSM::derivedMetricsBlockClass, 
					int,
					calPeriForEachPixel,
					calPeriMetrics, 
					TemplateMethodProc, 
					ADD<int>> *process 
	= new AbstractMerge<CuLSM::myPatch, 
						CuLSM::derivedMetricsBlockClass, 
						int, 
						calPeriForEachPixel,
						calPeriMetrics, 
						TemplateMethodProc, 
						ADD<int>>(argv[1]);

	// AbstractMerge<  CuLSM::myPatch, 
	// 				CuLSM::derivedMetricsBlockClass, 
	// 				int,
	// 				calAreaForEachPixel,
	// 				calAreaMetrics, 
	// 				TemplateMethodProc, ADD<int>> *process 
	// = new AbstractMerge<CuLSM::myPatch, 
	// 					CuLSM::derivedMetricsBlockClass, 
	// 					int, 
	// 					calAreaForEachPixel,
	// 					calAreaMetrics, 
	// 					TemplateMethodProc, 
	// 					ADD<int>>(argv[1]);
	process->calInEachBlock();
	delete process;

	alltime.stop();
	printf("%s = %f\n", VNAME(alltime), alltime.elapsed());

	return 0;
}










// GDAL读数据
// template<class dataBlockType>
// int getDevideInfo(int width, int height, int nodata, dataBlockType** dataBlockArray)
// {
// 	int maxnum;		//可以读入的像元的个数
// 	size_t freeGPU, totalGPU;
// 	cudaMemGetInfo(&freeGPU, &totalGPU);//size_t* free, size_t* total
// 	cout << "(free,total)" << freeGPU << "," << totalGPU << endl;

// 	maxnum = (freeGPU) / (sizeof(int)* 10);//每个pixel基本上要开辟6个中间变量，变量类型都是int
// 	// maxnum = (freeGPU) / (sizeof(int)* 6 * 2);//每个pixel基本上要开辟6个中间变量，变量类型都是int
// 	int sub_height = maxnum / width - 5;	//每个分块的高度sub_height
// 	sub_height = 2500;
// 	int blockNum = height / sub_height + 1;	//总的分块个数

// 	//*dataBlockArray = new CuLSM::dataBlock[blockNum];
// 	*dataBlockArray = (dataBlockType*)malloc(blockNum*sizeof(dataBlockType));

// 	int subIdx = 0;
// 	for (int height_all = 0; height_all < height; height_all += sub_height)
// 	{
// 		int task_start = subIdx*sub_height;
// 		int task_end;
// 		if ((subIdx + 1)*sub_height - height <= 0)
// 			task_end = (subIdx + 1)*sub_height - 1;
// 		else
// 			task_end = height - 1;
// 		int data_start, data_end;
// 		if (task_start - 1 <= 0)
// 			data_start = 0;
// 		else
// 			data_start = task_start - 1;
// 		if (task_end + 1 >= height - 1)
// 			data_end = height - 1;
// 		else
// 			data_end = task_end + 1;
// 		int data_height = data_end - data_start + 1;
// 		int task_height = task_end - task_start + 1;

// 		(*dataBlockArray)[subIdx].mnDataStart = data_start;
// 		(*dataBlockArray)[subIdx].mnDataEnd = data_end;
// 		(*dataBlockArray)[subIdx].mnTaskStart = task_start;
// 		(*dataBlockArray)[subIdx].mnTaskEnd = task_end;
// 		(*dataBlockArray)[subIdx].mnSubTaskHeight = task_height;
// 		(*dataBlockArray)[subIdx].mnSubDataHeight = data_height;
// 		(*dataBlockArray)[subIdx].mnStartTag = task_start*width;//当前分块的起始标记值，也就是该分块的第一个栅格的一维索引值
// 		(*dataBlockArray)[subIdx].mnWidth = width;
// 		(*dataBlockArray)[subIdx].mnNodata = nodata;

// 		subIdx++;
// 	}
// 	return blockNum;
// }

// template<class dataBlockType>
// void recordBoundary(dataBlockType &curBlock, int iBlock, int width,
// 	int** vecOriginValRow1, int** vecOriginValRow2, int** vecLabelValRow1, int** vecLabelValRow2)
// {
// 	int nBytePerLine = sizeof(int)*width;
// 	if (curBlock.isFirstBlock())
// 	{
// 		memcpy(*vecOriginValRow1 + iBlock*width, curBlock.mh_SubData + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
// 		memcpy(*vecLabelValRow1 + iBlock*width, curBlock.mh_LabelVal + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
// 	}
// 	else if (curBlock.isLastBlock())
// 	{
// 		memcpy(*vecOriginValRow2 + (iBlock - 1)*width, curBlock.mh_SubData, nBytePerLine);
// 		memcpy(*vecLabelValRow2 + (iBlock - 1)*width, curBlock.mh_LabelVal, nBytePerLine);

// 	}
// 	else
// 	{
// 		memcpy(*vecOriginValRow2 + (iBlock - 1)*width, curBlock.mh_SubData, nBytePerLine);
// 		memcpy(*vecLabelValRow2 + (iBlock - 1)*width, curBlock.mh_LabelVal, nBytePerLine);

// 		memcpy(*vecOriginValRow1 + iBlock*width, curBlock.mh_SubData + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
// 		memcpy(*vecLabelValRow1 + iBlock*width, curBlock.mh_LabelVal + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
// 	}
// }

// void showArr(int* arr, int size)
// {
// 	for (int i = 0; i < size; ++i)
// 	{
// 		cout<<arr[i]<<"\t";
// 	}
// 	cout<<endl;
// }



// int main(int argc, char const *argv[])
// {
// 	//初始化CUDA
// 	int gpuIdx = 1;//设置计算能力大于3.5的GPU
// 	initCUDA(gpuIdx);


// 	//初始化GDAL
// 	GDALAllRegister();
// 	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
// 	CGDALRead* pread = new CGDALRead;

// 	if (!pread->loadMetaData(argv[1]))
// 	{
// 		cout << "load error!" << endl;
// 	}
// 	cout << "rows:" << pread->rows() << endl;
// 	cout << "cols:" << pread->cols() << endl;
// 	cout << "bandnum:" << pread->bandnum() << endl;
// 	cout << "datalength:" << pread->datalength() << endl;
// 	cout << "invalidValue:" << pread->invalidValue() << endl;
// 	cout << "datatype:" << GDALGetDataTypeName(pread->datatype()) << endl;
// 	cout << "projectionRef:" << pread->projectionRef() << endl;
// 	cout << "perPixelSize:" << pread->perPixelSize() << endl;


// 	int width = pread->cols();
// 	int height = pread->rows();
// 	int nodata = (int)pread->invalidValue();
	
// 	CuLSM::derivedPeriBlockClass* dataBlockArray = NULL;		//blockInfo
// 	int blockNum = getDevideInfo(width, height, nodata, &dataBlockArray);


// 	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
// 	dim3 blockDim1(blockSize.x, blockSize.y, 1);
// 	dim3 gridDim1((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);


// 	vector<CuLSM::myPatch> vecAllLabel;//用于记录不重复的label
// 	CuLSM::UnionFind<CuLSM::myPatch, ADD<int>> *Quf = new CuLSM::UnionFind<CuLSM::myPatch, ADD<int>>();


// 	if(blockNum > 1)
// 	{
// 		int* vecOriginValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecOriginValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecLabelValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecLabelValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		checkMemAlloc(vecOriginValRow1);
// 		checkMemAlloc(vecOriginValRow2);
// 		checkMemAlloc(vecLabelValRow1);
// 		checkMemAlloc(vecLabelValRow2);

// 		for (int iBlock = 0; iBlock < blockNum; iBlock++)
// 		{
// 			CuLSM::derivedPeriBlockClass *curBlock = &(dataBlockArray[iBlock]);
// 			PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics> *method 
// 			= new PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics>
// 			(curBlock,pread, gridDim1,blockDim1);
// 			// (curBlock,pread,srcTest, gridDim1,blockDim1);
			
// 			method->templateMethod(curBlock, vecAllLabel,gridDim1,blockDim1,
// 								   	 &(curBlock->mh_compactPerimeterByPixel));
// 			recordBoundary(dataBlockArray[iBlock], iBlock, width, &vecOriginValRow1, &vecOriginValRow2, &vecLabelValRow1, &vecLabelValRow2);
// 	#ifdef _DEBUG
// 			cout<<"mergeArray-------------------------------------------------------------"<<endl;
// 			showArr(vecOriginValRow1,width*(blockNum-1));
// 			showArr(vecOriginValRow2,width*(blockNum-1));
// 			showArr(vecLabelValRow1,width*(blockNum-1));
// 			showArr(vecLabelValRow2,width*(blockNum-1));
// 			cout<<"mergeArray-------------------------------------------------------------"<<endl;
// 	#endif

// 			delete method;
// 			method = NULL;
// 		}
		
// 	#ifdef _DEBUG
// 		for (int i = 0; i < vecAllLabel.size(); ++i)
// 		{
// 			CuLSM::myPatch temp = vecAllLabel[i];
// 			cout<< temp.nLabel <<"\t"
// 				<< temp.nType <<"\t"
// 				<< temp.nPerimeterByPixel << endl;
// 		}
// 	#endif
		

// 		Quf->initUF(vecAllLabel);
// 		mergePatch(width, blockNum, (int)pread->invalidValue(), vecOriginValRow1, vecOriginValRow2, vecLabelValRow1, vecLabelValRow2, Quf);
// 		free(vecOriginValRow1);
// 		free(vecOriginValRow2);
// 		free(vecLabelValRow1);
// 		free(vecLabelValRow2);

// 		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
// 		Quf->qOutputRootMap("patchLevelResult");

// 		delete Quf;
// 		free(dataBlockArray);

// 	}
// 	else
// 	{
// 		CuLSM::derivedPeriBlockClass *curBlock = &(dataBlockArray[0]);
		
// 		PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics> *method 
// 		= new PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics>
// 		(curBlock,pread, gridDim1,blockDim1);
		
// 		method->templateMethod(curBlock, vecAllLabel,gridDim1,blockDim1,
// 								   	 &(curBlock->mh_compactPerimeterByPixel));
// 	}

// 	return 0;
// }

















//测试数据
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
	
// 	CuLSM::derivedPeriBlockClass* dataBlockArray = NULL;		//blockInfo
// 	int blockNum = getDevideInfo(width, height, nodata, &dataBlockArray);


// 	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
// 	dim3 blockDim1(blockSize.x, blockSize.y, 1);
// 	dim3 gridDim1((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);


// 	vector<CuLSM::myPatch> vecAllLabel;//用于记录不重复的label
// 	CuLSM::UnionFind<CuLSM::myPatch, ADD<int>> *Quf = new CuLSM::UnionFind<CuLSM::myPatch, ADD<int>>();


// 	if(blockNum > 1)
// 	{
// 		int* vecOriginValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecOriginValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecLabelValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		int* vecLabelValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
// 		checkMemAlloc(vecOriginValRow1);
// 		checkMemAlloc(vecOriginValRow2);
// 		checkMemAlloc(vecLabelValRow1);
// 		checkMemAlloc(vecLabelValRow2);

// 		for (int iBlock = 0; iBlock < blockNum; iBlock++)
// 		{
// 			CuLSM::derivedPeriBlockClass *curBlock = &(dataBlockArray[iBlock]);
// 			PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics> *method 
// 			= new PeriMethodProc<CuLSM::myPatch,int,calPeriForEachPixel,calPeriMetrics>
// 			(curBlock,pread,srcTest, gridDim1,blockDim1);
			
// 			method->templateMethod(curBlock, vecAllLabel,gridDim1,blockDim1,
// 								   	 &(curBlock->mh_compactPerimeterByPixel));
// 			recordBoundary(dataBlockArray[iBlock], iBlock, width, &vecOriginValRow1, &vecOriginValRow2, &vecLabelValRow1, &vecLabelValRow2);
// 			cout<<"mergeArray-------------------------------------------------------------"<<endl;
// 			showArr(vecOriginValRow1,width*(blockNum-1));
// 			showArr(vecOriginValRow2,width*(blockNum-1));
// 			showArr(vecLabelValRow1,width*(blockNum-1));
// 			showArr(vecLabelValRow2,width*(blockNum-1));
// 			cout<<"mergeArray-------------------------------------------------------------"<<endl;

// 			delete method;
// 			method = NULL;
// 		}
		
// 		for (int i = 0; i < vecAllLabel.size(); ++i)
// 		{
// 			CuLSM::myPatch temp = vecAllLabel[i];
// 			cout<< temp.nLabel <<"\t"
// 				<< temp.nType <<"\t"
// 				<< temp.nPerimeterByPixel << endl;
// 		}
		

// 		Quf->initUF(vecAllLabel);
// 		mergePatch(width, blockNum, (int)pread->invalidValue(), vecOriginValRow1, vecOriginValRow2, vecLabelValRow1, vecLabelValRow2, Quf);
// 		free(vecOriginValRow1);
// 		free(vecOriginValRow2);
// 		free(vecLabelValRow1);
// 		free(vecLabelValRow2);

// 		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
// 		Quf->qOutputRootMap("patchLevelResult");

// 		delete Quf;
// 		free(dataBlockArray);

// 	}

// 	return 0;
// }



