#ifndef __CMERGE_CUH__
#define __CMERGE_CUH__
#include "cuCCL.cuh"
#include "GDALRead.h"
#include "compact_template.cuh"

// #define _DEBUG
/*
这个类实现要分块计算的大图像的封装
 */
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

__global__ void getIsValid(int* d_label, int* d_isValid, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;

	bool limits = x < width && y < task_height;
	if (limits)
	{
		int center = d_label[gid];
		if(center != NO_USE_CLASS)
		{
			if (d_label[gid] == gid)
			{
				d_isValid[gid] = 1;
			}			
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
		int center = dev_labelMap[gid];
		if( center != NO_USE_CLASS)
		{
			dev_labelMap[gid] += labelStart;
		}	
	}
}

int findMerge4(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel)
{
	int Meg_count = 0;//开始计数Meg
	int center;
	for (int i = 0; i < width; i++)
	{
		int	LastLabel = -1;//上次标记序号
		int	CurLabel = -1;
		center = h_subDataFirst[i];//以上一行中每个pixel为中心，构造模板遍历
		if (center == BGvalue)
			continue;
		if (center == h_subDataSecond[i])//同一列中上一行数据与下一行图像数据一致
		{
			LastLabel = lastRowLabel[i];//上次标记序号
			CurLabel = firstRowLabel[i];
			int	repetition = 0;//是否重复
			for (int i = 0; i < Meg_count; i++)
			{
				if ((Meg[2 * i] == LastLabel) && (Meg[2 * i + 1] == CurLabel))
				{
					repetition = 1;
					break;
				}
			}
			if (!repetition)
			{
				Meg[Meg_count * 2] = LastLabel;
				Meg[Meg_count * 2 + 1] = CurLabel;
				Meg_count++;
			}
		}
	}
	return Meg_count;
}

int findMerge8(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel)
{
	int Meg_count = 0;//开始计数Meg
	int center;
	for (int i = 0; i < width; i++)
	{
		int	LastLabel = -1;//上次标记序号
		int	CurLabel = -1;
		center = h_subDataFirst[i];//以上一行中每个pixel为中心，构造模板遍历
		if (center == BGvalue)
			continue;
		if (center == h_subDataSecond[i])//同一列中上一行数据与下一行图像数据一致
		{
			LastLabel = lastRowLabel[i];//上次标记序号
			CurLabel = firstRowLabel[i];
			int	repetition = 0;//是否重复
			for (int i = 0; i < Meg_count; i++)
			{
				if ((Meg[2 * i] == LastLabel) && (Meg[2 * i + 1] == CurLabel))
				{
					repetition = 1;
					break;
				}
			}
			if (!repetition)
			{
				Meg[Meg_count * 2] = LastLabel;
				Meg[Meg_count * 2 + 1] = CurLabel;
				Meg_count++;
			}
		}
		if ((i - 1 >= 0) && (center == h_subDataSecond[i - 1]))//上一行数据与左边下一行图像数据一致
		{
			LastLabel = lastRowLabel[i];//上次标记序号
			CurLabel = firstRowLabel[i - 1];

			int	repetition = 0;//是否重复
			for (int i = 0; i < Meg_count; i++)
			{
				if ((Meg[2 * i] == LastLabel) && (Meg[2 * i + 1] == CurLabel))
				{
					repetition = 1;
					break;
				}
			}
			if (!repetition)
			{
				Meg[Meg_count * 2] = LastLabel;
				Meg[Meg_count * 2 + 1] = CurLabel;
				Meg_count++;
			}
		}
		if ((i + 1 < width) && (center == h_subDataSecond[i + 1]))//上一行数据与右边下一行图像数据一致
		{
			LastLabel = lastRowLabel[i];//上次标记序号
			CurLabel = firstRowLabel[i + 1];

			int	repetition = 0;//是否重复
			for (int i = 0; i < Meg_count; i++)
			{
				if ((Meg[2 * i] == LastLabel) && (Meg[2 * i + 1] == CurLabel))
				{
					repetition = 1;
					break;
				}
			}
			if (!repetition)
			{
				Meg[Meg_count * 2] = LastLabel;
				Meg[Meg_count * 2 + 1] = CurLabel;
				Meg_count++;
			}
		}
	}
	return Meg_count;
}

template<class PatchType, class MergeType>
void mergePatchUsingUF(int* mergeArr, int mergeCount, CuLSM::UnionFind<PatchType, MergeType> *Quf)
{
	for (int i = 0; i < mergeCount; i++)
	{
		if (mergeArr[2 * i] != -1)
		{
			int	cur_index = mergeArr[2 * i + 1];
			int	last_index = Quf->qFind(mergeArr[2 * i]);
			for (int j = i + 1; j < mergeCount; j++)//遍历后面的合并数组是否有和当前的cur_index一样的（连通U型）
			{
				if (mergeArr[j * 2 + 1] == cur_index)
				{
					//merge
					int	cur_lastindex = Quf->qFind(mergeArr[j * 2]);
					Quf->qUnion(cur_lastindex, cur_index);//合并序号
					mergeArr[j * 2] = mergeArr[j * 2 + 1] = -1;//标记无效
				}
			}
			//merge 
			Quf->qUnion(last_index, cur_index);
			mergeArr[i * 2] = mergeArr[i * 2 + 1] = -1;//标记已合并
		}
	}
}


class CBigImgCCL
{
private:
	cuCCLClass *mpCclObj;
	
	#ifdef _DEBUG
	PRead* pread;
	int* m_src;
	#else
	CGDALRead* pread;
	#endif
	
	CuLSM::UnionFind<CuLSM::CPatchLabel, ADD<int>> *Quf;
	CuLSM::dataBlock* dataBlockArray;
	int blockNum;
	GlobalConfiguration G_Config;

private:	
	int* vecOriginValRow1;
	int* vecOriginValRow2;
	int* vecLabelValRow1;
	int* vecLabelValRow2;
	dim3 blockDim1;
	dim3 gridDim1;


public:
	#ifdef _DEBUG
	CBigImgCCL(PRead* _pread, int *_src);
	#else
	CBigImgCCL(const char* _filename);
	#endif
	void calInEachBlock();
	~CBigImgCCL()
	{
		delete Quf;
		delete pread;
		free(dataBlockArray);
		if(vecOriginValRow1!=NULL)
				free(vecOriginValRow1);
		if(vecOriginValRow2!=NULL)
				free(vecOriginValRow2);
		if(vecLabelValRow1!=NULL)
				free(vecLabelValRow1);
		if(vecLabelValRow2!=NULL)
				free(vecLabelValRow2);
	}
private:
	int getDevideInfo();
	void compactMethod(CuLSM::dataBlock *curBlock);
	void recordBoundary(CuLSM::dataBlock &curBlock, int iBlock, int width);
	void mergePatch();	
};

#ifdef _DEBUG
CBigImgCCL::CBigImgCCL(PRead* _pread, int* _src)
{
	pread = _pread;
	m_src = _src;
	int width = pread->cols();
	int height = pread->rows();
	int nodata = (int)pread->invalidValue();	
	blockNum = getDevideInfo();
	if(blockNum > 1)
	{
		vecOriginValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecOriginValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecLabelValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecLabelValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		checkMemAlloc(vecOriginValRow1);
		checkMemAlloc(vecOriginValRow2);
		checkMemAlloc(vecLabelValRow1);
		checkMemAlloc(vecLabelValRow2);
	}
	else
	{
		vecOriginValRow1 = NULL;
		vecOriginValRow2 = NULL;
		vecLabelValRow1 = NULL;
		vecLabelValRow2	= NULL;
	}
	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
	blockDim1 = dim3(blockSize.x, blockSize.y, 1);
	gridDim1 = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

	Quf = new CuLSM::UnionFind<CuLSM::CPatchLabel, ADD<int>>();
	G_Config = Singleton<GlobalConfiguration>::Instance();

}
#else
CBigImgCCL::CBigImgCCL(const char* _filename)
{
	pread = new CGDALRead;
	if (!pread->loadMetaData(_filename))
	{
		cout << "load error!" << endl;
	}
	cout << "rows:" << pread->rows() << endl;
	cout << "cols:" << pread->cols() << endl;
	cout << "bandnum:" << pread->bandnum() << endl;
	cout << "datalength:" << pread->datalength() << endl;
	cout << "invalidValue:" << pread->invalidValue() << endl;
	cout << "datatype:" << GDALGetDataTypeName(pread->datatype()) << endl;
	cout << "projectionRef:" << pread->projectionRef() << endl;
	cout << "perPixelSize:" << pread->perPixelSize() << endl;	
	
	int width = pread->cols();
	int height = pread->rows();
	int nodata = (int)pread->invalidValue();	
	blockNum = getDevideInfo();
	if(blockNum > 1)
	{
		vecOriginValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecOriginValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecLabelValRow1 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		vecLabelValRow2 = (int*)malloc(sizeof(int)* width * (blockNum - 1));
		checkMemAlloc(vecOriginValRow1);
		checkMemAlloc(vecOriginValRow2);
		checkMemAlloc(vecLabelValRow1);
		checkMemAlloc(vecLabelValRow2);
	}
	else
	{
		vecOriginValRow1 = NULL;
		vecOriginValRow2 = NULL;
		vecLabelValRow1 = NULL;
		vecLabelValRow2	= NULL;
	}
	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
	blockDim1 = dim3(blockSize.x, blockSize.y, 1);
	gridDim1 = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

	Quf = new CuLSM::UnionFind<CuLSM::CPatchLabel, ADD<int>>();
	G_Config = Singleton<GlobalConfiguration>::Instance();
	
}
#endif
int CBigImgCCL::getDevideInfo()
{
	int width = pread->cols();
	int height = pread->rows();
	int nodata = (int)pread->invalidValue();	
	int maxnum;		//可以读入的像元的个数
	size_t freeGPU, totalGPU;
	cudaMemGetInfo(&freeGPU, &totalGPU);//size_t* free, size_t* total
	cout << "(free,total)" << freeGPU << "," << totalGPU << endl;

	maxnum = (freeGPU) / (sizeof(int)* 10);//每个pixel基本上要开辟6个中间变量，变量类型都是int
	// maxnum = (freeGPU) / (sizeof(int)* 6 * 2);//每个pixel基本上要开辟6个中间变量，变量类型都是int
	int sub_height = maxnum / width - 5;	//每个分块的高度sub_height

	// sub_height = 1000;
	#ifdef _DEBUG
	sub_height = 2;
	#endif
	int blockNum = height / sub_height + 1;	//总的分块个数

	//*dataBlockArray = new CuLSM::dataBlock[blockNum];
	dataBlockArray = (CuLSM::dataBlock*)malloc(blockNum*sizeof(CuLSM::dataBlock));

	int subIdx = 0;
	for (int height_all = 0; height_all < height; height_all += sub_height)
	{
		int task_start = subIdx*sub_height;
		int task_end;
		if ((subIdx + 1)*sub_height - height <= 0)
			task_end = (subIdx + 1)*sub_height - 1;
		else
			task_end = height - 1;
		int data_start, data_end;
		if (task_start - 1 <= 0)
			data_start = 0;
		else
			data_start = task_start - 1;
		if (task_end + 1 >= height - 1)
			data_end = height - 1;
		else
			data_end = task_end + 1;
		int data_height = data_end - data_start + 1;
		int task_height = task_end - task_start + 1;

		dataBlockArray[subIdx].mnDataStart = data_start;
		dataBlockArray[subIdx].mnDataEnd = data_end;
		dataBlockArray[subIdx].mnTaskStart = task_start;
		dataBlockArray[subIdx].mnTaskEnd = task_end;
		dataBlockArray[subIdx].mnSubTaskHeight = task_height;
		dataBlockArray[subIdx].mnSubDataHeight = data_height;
		dataBlockArray[subIdx].mnStartTag = task_start*width;//当前分块的起始标记值，也就是该分块的第一个栅格的一维索引值
		dataBlockArray[subIdx].mnWidth = width;
		dataBlockArray[subIdx].mnNodata = nodata;

		subIdx++;
	}
	return blockNum;
}

void CBigImgCCL::recordBoundary(CuLSM::dataBlock &curBlock, int iBlock, int width)
{
	int nBytePerLine = sizeof(int)*width;
	if (curBlock.isFirstBlock())
	{
		memcpy(vecOriginValRow1 + iBlock*width, curBlock.mh_SubData + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
		memcpy(vecLabelValRow1 + iBlock*width, curBlock.mh_LabelVal + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
	}
	else if (curBlock.isLastBlock())
	{
		memcpy(vecOriginValRow2 + (iBlock - 1)*width, curBlock.mh_SubData, nBytePerLine);
		memcpy(vecLabelValRow2 + (iBlock - 1)*width, curBlock.mh_LabelVal, nBytePerLine);

	}
	else
	{
		memcpy(vecOriginValRow2 + (iBlock - 1)*width, curBlock.mh_SubData, nBytePerLine);
		memcpy(vecLabelValRow2 + (iBlock - 1)*width, curBlock.mh_LabelVal, nBytePerLine);

		memcpy(vecOriginValRow1 + iBlock*width, curBlock.mh_SubData + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
		memcpy(vecLabelValRow1 + iBlock*width, curBlock.mh_LabelVal + width*(curBlock.mnSubTaskHeight - 1), nBytePerLine);
	}
}


void CBigImgCCL::compactMethod(CuLSM::dataBlock *_curBlock)
{
	int _width = mpCclObj->width;
	int _taskHeight = mpCclObj->task_height;
	int _nBytes_task = sizeof(int) * _width * _taskHeight;
	const int numElements = _width * _taskHeight;
	int* d_outputLabelOfSubData = mpCclObj->devLabelMap->getDevData();
	int* d_inputSrcSubData = mpCclObj->devSrcData->getDevData();
	
	Array2D< Cutype<int> >* d_IsValid = new Array2D< Cutype<int> >(_taskHeight,_width);
	_curBlock->mh_curPatchNum = -1;	//置为-1表示还没有经过规约，不知道当前分块有多少个patch


	int* d_IsValidData = d_IsValid->getDevData();
	getIsValid << <gridDim1,blockDim1 >> >(d_outputLabelOfSubData, d_IsValidData, _width, _taskHeight);

	//记录rootMap的一维位置，即d_label[gid]==gid的位置
	compact_t_device(&(_curBlock->mh_RootPos), &(_curBlock->mh_curPatchNum), 
					   d_outputLabelOfSubData, d_IsValidData, numElements);

	updateDevLabel << <gridDim1,blockDim1 >> > (d_outputLabelOfSubData, _curBlock->mnStartTag, _taskHeight, _width);

	_curBlock->mh_LabelVal= (int*)malloc(_nBytes_task);
	checkCudaErrors(cudaMemcpy(_curBlock->mh_LabelVal, d_outputLabelOfSubData, _nBytes_task, cudaMemcpyDeviceToHost));
	
	compact_t_device(&(_curBlock->mh_compactLabel), &(_curBlock->mh_curPatchNum), 
						d_outputLabelOfSubData, d_IsValidData, numElements);
	compact_t_device(&(_curBlock->mh_compactSrc), &(_curBlock->mh_curPatchNum), 
						d_inputSrcSubData, d_IsValidData, numElements);
	
	cout << "h_outputNumOfValidElements: " << _curBlock->mh_curPatchNum << endl;


	for (int i = 0; i < _curBlock->mh_curPatchNum; i++)
	{	
		CuLSM::CPatchLabel temp;
		temp.nLabel = _curBlock->mh_compactLabel[i];
		temp.nType = _curBlock->mh_compactSrc[i];
		temp.isUseful = false;
		Quf->rootMap.insert(make_pair(temp.nLabel, temp));
	}
	if (d_IsValid!=NULL)
	{
		delete d_IsValid;
		d_IsValid = NULL;
	}
}
void CBigImgCCL::mergePatch()
{
	int*h_rowOneValue = vecOriginValRow1;
	int*h_rowTwoValue = vecOriginValRow2;
	int*h_rowOneLabel = vecLabelValRow1;
	int*h_rowTwoLabel = vecLabelValRow2;

	int width = pread->cols();
	int BGvalue = (int)pread->invalidValue();	

	clock_t start1, end1;
	start1 = clock();
	int *mergeArr = NULL;	//合并数组
	int mergeCount = 0;	//合并计数
	int i;
	for (i = 0; i< blockNum - 1; i++)	//mergeStructArraySize = blockNum-1
	{
		mergeArr = (int *)malloc(sizeof(int)* width * 2);
		if (mergeArr == NULL)
		{
			printf("\nERROR! Can not allocate space for mergeArr!");
			exit(-1);
		}
		
		if (G_Config.USE_DIAGS)
		{
			mergeCount = findMerge8(width, BGvalue, mergeArr, h_rowOneValue + i*width, h_rowTwoValue + i*width, h_rowOneLabel + i*width, h_rowTwoLabel + i*width);
		}
		else
		{
			mergeCount = findMerge4(width, BGvalue, mergeArr, h_rowOneValue + i*width, h_rowTwoValue + i*width, h_rowOneLabel + i*width, h_rowTwoLabel + i*width);
		}
		mergePatchUsingUF(mergeArr, mergeCount, Quf);
		free(mergeArr);
		mergeArr = NULL;
		mergeCount = 0;
	}
	end1 = clock();
	double dur = (double)(end1 - start1);
	printf("LineCCLNoSplit Use Time:%f\n", (dur / CLOCKS_PER_SEC));
}
void CBigImgCCL::calInEachBlock()
{
	int width = pread->cols();
	if(blockNum > 1)
	{	
		for (int iBlock = 0; iBlock < blockNum; iBlock++)
		{
			CuLSM::dataBlock *curBlock = &(dataBlockArray[iBlock]);
			//step 1 GDAL READ
			#ifdef _DEBUG
			curBlock->loadBlockData(pread,m_src);
			#else
			curBlock->loadBlockData(pread);
			#endif
			//step 2 run CCL save label result in -----cuCCLClass: devLabelMap;
			mpCclObj =  new cuCCLClass(curBlock->mh_SubData, curBlock->mnWidth, curBlock->mnSubTaskHeight, curBlock->mnNodata);
			mpCclObj->gpuLineUF(blockDim1,gridDim1);

			//step 3 compress the sparse matrics for output
			//将所有可能的根节点的位置保存到mh_RootPos中，将标记值存在mh_LabelVal中
			//构造节点放在Quf->rootMap中，为合并分块做准备
			compactMethod(curBlock);

			//step 4 record boundary between two blocks for UnionFind merge
			recordBoundary(dataBlockArray[iBlock], iBlock, width);

			if(mpCclObj!=NULL)
			{	
				delete mpCclObj;
				mpCclObj = NULL;
			}
		}
		// Quf->initUF(vecAllLabel);
		mergePatch();
		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
		Quf->qOutputRootMap("patchLevelResult");
		
		//将连通域标记为连续值,修改每个pixel,覆盖存储到mh_LabelVal中
		cout<<"curContinueLabel======================================" << endl;
		for (int iBlock = 0; iBlock < blockNum; iBlock++)
		{
			CuLSM::dataBlock *curBlock = &(dataBlockArray[iBlock]);
			curBlock->getContinueLabelVal(Quf);
		}
#ifdef _DEBUG
		for (int iBlock = 0; iBlock < blockNum; iBlock++)
		{
			CuLSM::dataBlock *curBlock = &(dataBlockArray[iBlock]);
			int *curContinueLabel = curBlock->mh_LabelVal;
			for (int i = 0; i < curBlock->mnSubTaskHeight; ++i)
			{
				for (int j = 0; j < curBlock->mnWidth; ++j)
				{
					cout << curContinueLabel[i*curBlock->mnWidth+j]<<"\t";
				}
				cout<<endl;
			}
		}
#endif
	}
	else
	{
		CuLSM::dataBlock *curBlock = &dataBlockArray[0];

		//step 1 GDAL READ
		#ifdef _DEBUG
		curBlock->loadBlockData(pread,m_src);
		#else
		curBlock->loadBlockData(pread);
		#endif
		
		//step 2 run CCL save label result in -----cuCCLClass: devLabelMap;
		mpCclObj =  new cuCCLClass(curBlock->mh_SubData, curBlock->mnWidth, curBlock->mnSubTaskHeight, curBlock->mnNodata);
		mpCclObj->gpuLineUF(blockDim1,gridDim1);

		//step 3 compress the sparse matrics for output
		compactMethod(curBlock);

		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
		Quf->qOutputRootMap("patchLevelResult");

		if(mpCclObj!=NULL)
		{	
			delete mpCclObj;
			mpCclObj = NULL;
		}

		//将连通域标记为连续值
		cout<<"curContinueLabel======================================" << endl;
		curBlock->getContinueLabelVal(Quf);
		#ifdef _DEBUG
		int *curContinueLabel = curBlock->mh_LabelVal;
		for (int i = 0; i < curBlock->mnSubTaskHeight; ++i)
		{
			for (int j = 0; j < curBlock->mnWidth; ++j)
			{
				cout << curContinueLabel[i*curBlock->mnWidth+j]<<"\t";
			}
			cout<<endl;
		}
		#endif
	}
}
#endif

int main(int argc, char const *argv[])
{
	int array[25] = { 1, 3, 3, 3, 3,
					  1, 3, 3, 1, 3,
					  1, 2, 1, 3, 2,
					  2, 1, 3, 2, 3,
					  1, 2, 2, 3, 2 };

	int* srcTest = new int[25];
	for (int i = 0; i < 25; i++)
	{
		srcTest[i] = array[i];
	}
	PRead *pread = new PRead(5, 5, 0);

	bool useful_class[10] = {1,1,1,1,1,1,1,1,1,1};
	// bool useful_class[10] = {0,0,1,1,1,1,1,0,1,1};
	//3,4,5,7
	// bool useful_class[10] = {0,0,0,0,1,0,0,0,0,0};

	//初始化CUDA
	int gpuIdx = 1;//设置计算能力大于3.5的GPU
	initCUDA(gpuIdx);
	
	//所有关于GPU显存的初始化都要放在initCUDA之后进行，否则会出现随机值
	GlobalConfiguration& config = Singleton<GlobalConfiguration>::Instance();
	config.set_USE(useful_class);
	config.set_USE_DIAGS(true);
	
	#ifdef _DEBUG
	CBigImgCCL *ccl = new CBigImgCCL(pread, srcTest);
	#else
	CBigImgCCL *ccl = new CBigImgCCL(argv[1]);
	#endif

	ccl->calInEachBlock();
	delete ccl;


	return 0;
}