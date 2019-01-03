#include "AbstractBlockClass.h"

#include <gdal_priv.h>
#include <cpl_conv.h>
#include "GDALRead.h"
#include "timer.h"
#include "Array2D_CUDA.h"
#include "localMidOper.cuh"
#include "localDevice.cuh"
#define VNAME(name) (#name)
// #define _TEST
// template<class DataType1,class OperType>
// void ReClass(DataType1* input, int width, int height, DataType1* oldValueSet, DataType1* newValueSet, int length);

#define NO_USE_CLASS -1

CuLSM::dataBlock::dataBlock()
{
	mnDataStart = -1;
	mnDataEnd = -1;
	mnTaskStart = -1;
	mnTaskEnd = -1;
	mnSubDataHeight = -1;
	mnSubTaskHeight = -1;
	mnStartTag = -1;
	mnWidth = -1;
	mnNodata = -1;



	mh_holoUp = NULL;					//当前分块的上一行，若为第一行则一直保持NULL
	mh_holoDown = NULL;				//当前分块的下一行，若为最后一行则一直保持NULL
	mh_SubData = NULL;					//保存当前分块的src信息
	
	mh_LabelVal = NULL;				//保存当前分块的标记值，从GPU传回的标记值
	mh_RelabelVal = NULL;				//保存当前分块的标记值,经过处理后的连续标记值
	mh_curPatchNum = -1;					//当前分块中斑块的数量
	
	mh_compactSrc = NULL;				//记录原始类型
	mh_compactLabel = NULL;				//标记值集合
	mh_compactRelabel = NULL;			//重标记值的集合

	// mh_compactMetricsByPixel = NULL;	//当前分块中要计算的指数（比如面积、周长）
	// mh_compactAreaByPixel = NULL;		//当前分块中每个斑块的面积（用像元数量表示）
	// mh_compactPerimeterByPixel = NULL;	//当前分块中每个斑块的周长（用像元为单位长度表示）
}



void CuLSM::dataBlock::loadBlockData(PRead* pread, int* src)
{
	int width = pread->cols();
	size_t nBytes_data = mnSubTaskHeight * width * sizeof(int);
	mh_SubData = (int*)malloc(nBytes_data);
	if (mh_SubData == NULL)
	{
		cout << "\nERROR! Can not allocate space for this block!!!" << endl;
		exit(-1);
	}
	memset(mh_SubData, 0, mnSubTaskHeight * width);
	memcpy(mh_SubData, src + mnTaskStart*width, nBytes_data);

	size_t nBytes_OneLine = width * sizeof(int);
	if (isSplit())
	{
		if (isFirstBlock())
		{
			mh_holoUp = NULL;
			mh_holoDown = (int*)malloc(nBytes_OneLine);
			memset(mh_holoDown, 0, width);
			memcpy(mh_holoDown, src + mnDataEnd * width, nBytes_OneLine);
		}
		else if (isLastBlock())
		{
			mh_holoUp = (int*)malloc(nBytes_OneLine);
			memset(mh_holoUp, 0, width);
			memcpy(mh_holoUp, src + mnDataStart * width, nBytes_OneLine);
			mh_holoDown = NULL;
		}
		else
		{
			mh_holoDown = (int*)malloc(nBytes_OneLine);
			memset(mh_holoDown, 0, width);
			memcpy(mh_holoDown, src + mnDataEnd * width, nBytes_OneLine);
			mh_holoUp = (int*)malloc(nBytes_OneLine);
			memset(mh_holoUp, 0, width);
			memcpy(mh_holoUp, src + mnDataStart * width, nBytes_OneLine);
		}
	}
	else
	{
		mh_holoUp = NULL;
		mh_holoDown = NULL;
	}

}

void callGDALLoadBlock(CGDALRead* pread, int width, int height, int nXOffset, int nYOffset, int* data)
{
	switch (pread->datatype())
	{
		case GDT_Byte:
		{
						 pread->readDataBlock<unsigned char>(width, height, nXOffset, nYOffset, data);
						 break;
		}
		case GDT_UInt16:
		{
						   pread->readDataBlock<unsigned short>(width, height, nXOffset, nYOffset, data);
						   break;
		}
		case GDT_Int16:
		{
						  pread->readDataBlock<short>(width, height, nXOffset, nYOffset, data);
						  break;
		}
		case GDT_UInt32:
		{
						   pread->readDataBlock<unsigned int>(width, height, nXOffset, nYOffset, data);
						   break;
		}
		case GDT_Int32:
		{
						  pread->readDataBlock<int>(width, height, nXOffset, nYOffset, data);
						  break;
		}
		case GDT_Float32:
		{
							float* allData = pread->transforData<float>();
							break;
		}
		case GDT_Float64:
		{
							double* allData = pread->transforData<double>();
							break;
		}
		default:
		{
				   cout << "transfor data type false!" << endl;
		}
	}
}


void CuLSM::dataBlock::loadBlockData(CGDALRead* pread)//int width, int data_height, int data_start, int** h_subData, CGDALRead* pread
{
	CpuTimer loadBlockData;
	loadBlockData.start();

	int width = pread->cols();
	size_t nBytes_data = mnSubTaskHeight * width * sizeof(int);
	mh_SubData = (int*)malloc(nBytes_data);
	if (mh_SubData == NULL)
	{
		cout << "\nERROR! Can not allocate space for this block!!!" << endl;
		exit(-1);
	}
	memset(mh_SubData, 0, mnSubTaskHeight * width);
	callGDALLoadBlock(pread, width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);

	size_t nBytes_OneLine = width * sizeof(int);
	if (isSplit())
	{
		if (isFirstBlock())
		{
			mh_holoUp = NULL;
			mh_holoDown = (int*)malloc(nBytes_OneLine);
			callGDALLoadBlock(pread, width, 1, 0, mnDataEnd, mh_holoDown);
		}
		else if (isLastBlock())
		{
			mh_holoUp = (int*)malloc(nBytes_OneLine);
			callGDALLoadBlock(pread, width, 1, 0, mnDataStart, mh_holoUp);
			mh_holoDown = NULL;
		}
		else
		{
			mh_holoUp = (int*)malloc(nBytes_OneLine);
			mh_holoDown = (int*)malloc(nBytes_OneLine);
			callGDALLoadBlock(pread, width, 1, 0, mnDataEnd, mh_holoDown);
			callGDALLoadBlock(pread, width, 1, 0, mnDataStart, mh_holoUp);
		}
	}
	//switch (pread->datatype())
	//{
	//	case GDT_Byte:
	//	{
	//					 pread->readDataBlock<unsigned char>(width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);
	//					 break;
	//	}
	//	case GDT_UInt16:
	//	{
	//					   pread->readDataBlock<unsigned short>(width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);
	//					   break;
	//	}
	//	case GDT_Int16:
	//	{
	//					  pread->readDataBlock<short>(width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);
	//					  break;
	//	}
	//	case GDT_UInt32:
	//	{
	//					   pread->readDataBlock<unsigned int>(width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);
	//					   break;
	//	}
	//	case GDT_Int32:
	//	{
	//					  pread->readDataBlock<int>(width, mnSubTaskHeight, 0, mnTaskStart, mh_SubData);
	//					  break;
	//	}
	//	case GDT_Float32:
	//	{
	//						float* allData = pread->transforData<float>();
	//						break;
	//	}
	//	case GDT_Float64:
	//	{
	//						double* allData = pread->transforData<double>();
	//						break;
	//	}
	//	default:
	//	{
	//			   cout << "transfor data type false!" << endl;
	//	}
	//}
	loadBlockData.stop();
	printf("%s = %f\n", VNAME(loadBlockData), loadBlockData.elapsed());

}

void CuLSM::dataBlock::freeSubData()
{
	if(this->mh_SubData!= NULL)
	{
		printf("%s\n", "freeSubData");
		free(this->mh_SubData);
		mh_SubData = NULL;
	}
}

void CuLSM::dataBlock::freeLabel()
{
	if(this->mh_LabelVal!= NULL)
	{
		printf("%s\n", "freeLabel");
		free(this->mh_LabelVal);
		mh_LabelVal = NULL;
	}
}
void CuLSM::dataBlock::freeRelabelVal()
{
	if (this->mh_RelabelVal != NULL)
	{
		printf("%s\n", "freeRelabelVal");
		free(this->mh_RelabelVal);
		mh_RelabelVal = NULL;
	}
}
void CuLSM::dataBlock::freeCompactLabel()
{
	if(this->mh_compactLabel!= NULL)
	{
		printf("%s\n", "freeCompactLabel");
		free(this->mh_compactLabel);
		mh_compactLabel = NULL;
	}
}
void CuLSM::dataBlock::freeCompactRelabel()
{
	if(this->mh_compactRelabel!= NULL)
	{
		printf("%s\n", "freeCompactRelabel");
		free(this->mh_compactRelabel);
		mh_compactRelabel = NULL;
	}
}

// void CuLSM::dataBlock::freeCompactMetricsByPixel()
// {
// 	if (this->mh_compactMetricsByPixel != NULL)
// 	{
// 		printf("%s\n", "freeCompactMetricsByPixel");
// 		free(this->mh_compactMetricsByPixel);
// 		mh_compactMetricsByPixel = NULL;
// 	}
// }
// void CuLSM::dataBlock::freeCompactAreaByPixel()
// {
// 	if (this->mh_compactAreaByPixel != NULL)
// 	{
// 		printf("%s\n", "freeCompactAreaByPixel");
// 		free(this->mh_compactAreaByPixel);
// 		mh_compactAreaByPixel = NULL;
// 	}
// }
// void CuLSM::dataBlock::freeCompactPerimeterByPixel()
// {
// 	if (this->mh_compactPerimeterByPixel != NULL)
// 	{
// 		printf("%s\n", "freeCompactPerimeterByPixel");
// 		free(this->mh_compactPerimeterByPixel);
// 		mh_compactPerimeterByPixel = NULL;
// 	}
// }
void CuLSM::dataBlock::freeHoloUp()
{
	if (this->mh_holoUp != NULL)
	{
		printf("%s\n", "freeHoloUp");
		free(this->mh_holoUp);
		mh_holoUp = NULL;
	}
}
void CuLSM::dataBlock::freeHoloDown()
{
	if (this->mh_holoDown != NULL)
	{
		printf("%s\n", "freeHoloDown");
		free(this->mh_holoDown);
		mh_holoDown = NULL;
	}
}

void CuLSM::dataBlock::freeRootPos()
{
	if (this->mh_RootPos != NULL)
	{
		printf("%s\n", "freeHoloDown");
		free(this->mh_RootPos);
		mh_RootPos = NULL;
	}
}
void CuLSM::dataBlock::freeCompactSrc()
{
	if (this->mh_compactSrc != NULL)
	{
		printf("%s\n", "freeCompactSrc");
		free(this->mh_compactSrc);
		mh_compactSrc = NULL;
	}
}
void CuLSM::dataBlock::getCompactRelabel(CuLSM::UnionFind<Patch,ADD<int>>* Quf)
{
	mh_compactRelabel = (int*)malloc(sizeof(int)*mh_curPatchNum);
	for (int i = 0; i < mh_curPatchNum; ++i)
	{
		mh_compactRelabel[i] = Quf->rootMap[mh_compactLabel[i]].nLabel;
	}

#ifdef _TEST
	cout<< "mh_compactLabel"<<endl;
	for (int i = 0; i < mh_curPatchNum; ++i)
	{
		cout<< mh_compactLabel[i]<<"\t";
	}
	cout<<endl;
	cout<< "mh_compactRelabel"<<endl;
	for (int i = 0; i < mh_curPatchNum; ++i)
	{
		cout<< mh_compactRelabel[i]<<"\t";
	}
	cout<<endl;
#endif

	// for (int i = 0; i < mnSubTaskHeight*mnWidth; i++)
	// {
	// 	mh_LabelVal[i] = Quf.qFind(mh_LabelVal[i]);
	// }
}

void CuLSM::dataBlock::getContinueLabelVal()
{
/* 将mh_LabelVal中的值修改为连续的标记值
input:mh_LabelVal
oldValueSet:mh_compactLabel
newValueSet:mh_compactRelabel
length:mh_curPatchNum
 */
	//relabel start 
	CpuTimer relabeltime;
	relabeltime.start();
#ifdef _TEST
	cout<<endl;
	cout<<"mh_LabelVal before relabel"<<endl;
	for (int i = 0; i < mnSubTaskHeight*mnWidth; ++i)
	{
		cout<< mh_LabelVal[i]<<"\t";
	}
	cout<<endl;
#endif

	//parallel
	ReClass<int, RelabelValueUpdate<int>>
	(mh_LabelVal, mnWidth, mnSubTaskHeight, mh_compactLabel, mh_compactRelabel, (int)mh_curPatchNum);


#ifdef _TEST
	cout<<"parallelResult with template"<<endl;
	for (int i = 0; i < mnSubTaskHeight*mnWidth; ++i)
	{
		cout<< mh_LabelVal[i]<<"\t";
	}
	cout<<endl;
#endif
	relabeltime.stop();
	printf("%s = %f\n", VNAME(relabeltime), relabeltime.elapsed());

}
__global__ void Set_d_IsRoot(int* d_RootPos, int* d_IsRoot, int curPatchNum, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;
	//判断哪些节点是根节点，是根节点的pixel不需要重标记
	if (gid < curPatchNum)
	{
		d_IsRoot[d_RootPos[gid]] = 1;
	}	
}
__global__ void ReLabelEachPixel(int* d_label, int* d_IsRoot, int labelStart, int width, int task_height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = x + y * width;//global 1D index;
	//判断哪些节点是根节点，是根节点的pixel不需要重标记

	bool limits = x < width && y < task_height;
	if (limits)
	{
		int center = d_label[gid];
		if(center!= NO_USE_CLASS)
		{
			if(!d_IsRoot[gid])//如果当前pixel不是根，更新其为根节点的值
			{
				d_label[gid] -= labelStart;
				d_label[gid] = d_label[d_label[gid]];
			}
		}
	}
}
void CuLSM::dataBlock::getContinueLabelVal(CuLSM::UnionFind<CuLSM::CPatchLabel, ADD<int>>* Quf)
{
/* 将mh_LabelVal中的值修改为连续的标记值
input:mh_LabelVal
oldValueSet:mh_compactLabel
newValueSet:mh_compactRelabel
length:mh_curPatchNum
 */
	//relabel start 
	CpuTimer relabeltime;
	relabeltime.start();

	// mh_RelabelVal = (int*)malloc(sizeof(int) * mnWidth * mnSubTaskHeight);

	// // sequencial
	// cout<<"sequencial relabel"<<endl;
	// for(int i = 0; i < mnWidth * mnSubTaskHeight; i++)
	// {
	// 	mh_RelabelVal[i] = Quf->rootMap[mh_LabelVal[i]].nLabel;
	// }
	// relabeltime.stop();
	// printf("%s = %f\n", VNAME(relabeltime), relabeltime.elapsed());


	// 先修改根节点的标记，再用CUDA更新每个pixel的标记

	for (int i = 0; i < mh_curPatchNum; ++i)
	{
		int tempPos = mh_RootPos[i];		//	根节点的一维位置索引
		int tempVal = mh_LabelVal[tempPos];	//这些位置原先的标记值，要找到这个值的root重新赋值
		mh_LabelVal[tempPos] = Quf->rootMap[tempVal].nLabel;
	}

	int width = mnWidth;
	int height = mnSubTaskHeight;
	int _nBytes_task = sizeof(int) * width * height;
	int* h_Label = mh_LabelVal;
	int* h_RootPos = mh_RootPos;
#ifdef _DEBUG
	int2 blockSize; 	blockSize.x = 2;	blockSize.y = 2;
#else
	int2 blockSize; 	blockSize.x = 32;	blockSize.y = 16;
#endif
	dim3 blockDim1 = dim3(blockSize.x, blockSize.y, 1);
	dim3 gridDim1 = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, 1);

	Array2D< Cutype<int> >* d_Label = new Array2D< Cutype<int> >(h_Label, height, width);
	Array2D< Cutype<int> >* d_IsRoot = new Array2D< Cutype<int> >(height, width);
	Array2D< Cutype<int> >* d_RootPos = new Array2D< Cutype<int> >(h_RootPos, 1, mh_curPatchNum);

	// d_RootPos->show();
 
	Set_d_IsRoot<< <gridDim1,blockDim1>> >(d_RootPos->getDevData(), d_IsRoot->getDevData(), mh_curPatchNum, width);
  	ReLabelEachPixel<< <gridDim1,blockDim1>> >(d_Label->getDevData(), d_IsRoot->getDevData(), mnStartTag, width, height);
	checkCudaErrors(cudaMemcpy(mh_LabelVal, d_Label->getDevData(), _nBytes_task, cudaMemcpyDeviceToHost));
	
	delete d_Label;
	delete d_IsRoot;
	delete d_RootPos;
}

int CuLSM::dataBlock::getDataStart()
{
	return mnDataStart;
}

int CuLSM::dataBlock::getDataEnd()
{
	return mnDataEnd;
}

int CuLSM::dataBlock::getTaskStart()
{
	return mnTaskStart;
}

int CuLSM::dataBlock::getTaskEnd()
{
	return mnTaskEnd;
}

int CuLSM::dataBlock::getDataHeight()
{
	return mnSubDataHeight;
}

int CuLSM::dataBlock::getTaskHeight()
{
	return mnSubTaskHeight;
}

int CuLSM::dataBlock::getStartTag()
{
	return mnStartTag;
}

int CuLSM::dataBlock::getWidth()
{
	return mnWidth;
}

int CuLSM::dataBlock::getNoData()
{
	return mnNodata;
}
int* CuLSM::dataBlock::getSrcData()
{
	return mh_SubData;
}
int* CuLSM::dataBlock::getLabelData()
{
	return mh_LabelVal;
}

void CuLSM::dataBlock::printLabelVal()
{
	for (int i = 0; i < mnSubTaskHeight * mnWidth; ++i)
	{
		cout<<mh_LabelVal[i]<<"\t";
	}
}
void CuLSM::dataBlock::printRelabelVal()
{
	for (int i = 0; i < mnSubTaskHeight * mnWidth; ++i)
	{
		cout << mh_RelabelVal[i] << "\t";
	}
}