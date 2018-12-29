#ifndef __ABSTRACT_MERGE_CUH__
#define __ABSTRACT_MERGE_CUH__

int findMerge(int width, int BGvalue, int* Meg, int* h_subDataFirst, int *h_subDataSecond, int* lastRowLabel, int* firstRowLabel)
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


template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template < typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper > class TemplateMethodType, 
		class MergeType>
class AbstractMerge
{
public:
	CGDALRead* pread;
	TemplateMethodType<PatchType,valType,CalforEachPixelOper,CalMetricsOper> *mpTmpMethodProc;
	// vector<PatchType> vecAllLabel;//用于记录不重复的label
public:
	CuLSM::UnionFind<PatchType, MergeType> *Quf;
	DataBlockType* dataBlockArray;		//blockInfo
	int blockNum;
public:
	int* vecOriginValRow1;
	int* vecOriginValRow2;
	int* vecLabelValRow1;
	int* vecLabelValRow2;
	dim3 blockDim1;
	dim3 gridDim1;
public:
	AbstractMerge(const char* _filename);
	int getDevideInfo();
	void recordBoundary(DataBlockType &curBlock, int iBlock, int width);
	void calInEachBlock();
	// virtual void calInEachBlock() = 0;
	void mergePatch();
	~AbstractMerge()
	{
		delete Quf;
		delete pread;
		delete mpTmpMethodProc;
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
};

template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template < typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper> class TemplateMethodType, 
		class MergeType>
		AbstractMerge<  PatchType, 
						DataBlockType, 
						valType, 
						CalforEachPixelOper, 
						CalMetricsOper, 
		 				TemplateMethodType, 
		 				MergeType>
::AbstractMerge(const char* _filename)
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
	gridDim1 = dim3((pread->cols() + blockSize.x - 1) / blockSize.x, (pread->rows() + blockSize.y - 1) / blockSize.y, 1);

	Quf = new CuLSM::UnionFind<PatchType, MergeType>();
	
}
template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template <typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper> class TemplateMethodType, 
		class MergeType>
int AbstractMerge<  PatchType, 
					DataBlockType, 
					valType, 
					CalforEachPixelOper, 
					CalMetricsOper, 
	 				TemplateMethodType, 
	 				MergeType>
::getDevideInfo()
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
	// sub_height = 2500;
	int blockNum = height / sub_height + 1;	//总的分块个数

	//*dataBlockArray = new CuLSM::dataBlock[blockNum];
	dataBlockArray = (DataBlockType*)malloc(blockNum*sizeof(DataBlockType));

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


template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template <typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper> class TemplateMethodType, 
		class MergeType>
void AbstractMerge< PatchType, 
					DataBlockType, 
					valType, 
					CalforEachPixelOper, 
					CalMetricsOper, 
	 				TemplateMethodType, 
	 				MergeType>
::recordBoundary(DataBlockType &curBlock, int iBlock, int width)
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

template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template <typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper> class TemplateMethodType, 
		class MergeType>
void AbstractMerge< PatchType, 
					DataBlockType, 
					valType, 
					CalforEachPixelOper, 
					CalMetricsOper, 
	 				TemplateMethodType, 
	 				MergeType>
::calInEachBlock()
{
	int width = pread->cols();
	if(blockNum > 1)
	{	
		for (int iBlock = 0; iBlock < blockNum; iBlock++)
		{
			DataBlockType *curBlock = &(dataBlockArray[iBlock]);
			mpTmpMethodProc = new TemplateMethodType<PatchType,valType,CalforEachPixelOper,CalMetricsOper>
			(curBlock,pread, gridDim1,blockDim1);
			// (curBlock,pread,srcTest, gridDim1,blockDim1);

			mpTmpMethodProc->templateMethod(curBlock, Quf->rootMap, gridDim1,blockDim1,
								   	 // &(curBlock->mh_compactPerimeterByPixel));
								   	 // &(curBlock->mh_compactAreaByPixel));
								   	 &(curBlock->mh_compactMetricsByPixel));
			recordBoundary(dataBlockArray[iBlock], iBlock, width);

			delete mpTmpMethodProc;
			mpTmpMethodProc = NULL;
		}
		// Quf->initUF(vecAllLabel);
		mergePatch();
		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
		Quf->qOutputRootMap("patchLevelResult");
	}
	else
	{
		DataBlockType *curBlock = &dataBlockArray[0];
		mpTmpMethodProc = new TemplateMethodType<PatchType,valType,CalforEachPixelOper,CalMetricsOper>
			(curBlock,pread, gridDim1,blockDim1);
		mpTmpMethodProc->templateMethod(curBlock, Quf->rootMap,gridDim1,blockDim1,
								   	 // &(curBlock->mh_compactPerimeterByPixel));
								   	 // &(curBlock->mh_compactAreaByPixel));
								   	 &(curBlock->mh_compactMetricsByPixel));
		Quf->qRelabel();		//对rootMap进行重标记，现在生成的是连续的序号。
		Quf->qOutputRootMap("patchLevelResult");
	}
}


template<class PatchType, 
		class DataBlockType, 
		class valType, 
		class CalforEachPixelOper, 
		class CalMetricsOper, 
		template <typename PatchType,
				   typename valType,
				   typename CalforEachPixelOper,
				   typename CalMetricsOper> class TemplateMethodType, 
		class MergeType>
void AbstractMerge< PatchType, 
					DataBlockType, 
					valType, 
					CalforEachPixelOper, 
					CalMetricsOper, 
	 				TemplateMethodType, 
	 				MergeType>
::mergePatch()
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
		mergeCount = findMerge(width, BGvalue, mergeArr, h_rowOneValue + i*width, h_rowTwoValue + i*width, h_rowOneLabel + i*width, h_rowTwoLabel + i*width);
		mergePatchUsingUF(mergeArr, mergeCount, Quf);
		free(mergeArr);
		mergeArr = NULL;
		mergeCount = 0;
	}
	end1 = clock();
	double dur = (double)(end1 - start1);
	printf("LineCCLNoSplit Use Time:%f\n", (dur / CLOCKS_PER_SEC));
}


#endif