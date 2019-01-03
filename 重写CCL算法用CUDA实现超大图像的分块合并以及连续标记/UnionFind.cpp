// #include "UnionFind.h"
#include "AbstractBlockClass.h"
// #include "derivedPeriBlockClass.h"
#include "timer.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <fstream>
using namespace std;
//参考https://jiayi797.github.io/2017/11/05/%E7%AE%97%E6%B3%95-%E5%B9%B6%E6%9F%A5%E9%9B%86/

//显式实例化类，为了避免编译不通过.给模板类型参数，模板创建实例对象
// template class CuLSM::UnionFind<CuLSM::myPatch,ADD<int>>;
template class CuLSM::UnionFind<CuLSM::CPatchLabel,ADD<int>>;


// template void mergePatch(int ,int, int, int*, int*, int*, int*, 
// 						CuLSM::UnionFind<CuLSM::myPatch,ADD<int>>*);

// template UnionFind<CuLSM::myPatch,ADD<int>>::initUF(vector<PatchType>& vecAllLabel);
template<class PatchType, class MergeType>
CuLSM::UnionFind<PatchType,MergeType>::UnionFind()
{
	keys = NULL;
	continueLabels = NULL;
	count = 0;
};


template<class PatchType, class MergeType>
CuLSM::UnionFind<PatchType,MergeType>::UnionFind(vector<PatchType>& vecAllLabel)
{
	count = vecAllLabel.size();
	for (int elem = 0; elem < count; elem++)
	{
		PatchType curEelem = vecAllLabel[elem];
		rootMap.insert(make_pair(curEelem.nLabel, curEelem));
	}
}


template<class PatchType, class MergeType>
CuLSM::UnionFind<PatchType,MergeType>::~UnionFind()
{
	rootMap.clear();
	if (keys!=NULL)
	{
		// free(keys);
		delete []keys;
		keys = NULL;
	}
	if (continueLabels!=NULL)
	{
		// free(continueLabels);
		delete []continueLabels;
		continueLabels=NULL;
	}
}


template<class PatchType, class MergeType>
void CuLSM::UnionFind<PatchType,MergeType>::initUF(vector<PatchType>& vecAllLabel)
{
	count = vecAllLabel.size();
	for (int elem = 0; elem < count; elem++)
	{
		PatchType curEelem = vecAllLabel[elem];
		rootMap.insert(make_pair(curEelem.nLabel, curEelem));
	}
}

template<class PatchType, class MergeType>
int CuLSM::UnionFind<PatchType,MergeType>::qFind(int key)
{
	while (key != rootMap[key].nLabel)
	{
		rootMap[key].nLabel = rootMap[rootMap[key].nLabel].nLabel;
		key = rootMap[key].nLabel;
	}
	return key;
}

template<class PatchType, class MergeType>
void CuLSM::UnionFind<PatchType,MergeType>::qUnion(int p, int q)
{
	int pRoot = qFind(p);
	int qRoot = qFind(q);
	if (pRoot == qRoot)
		return;
	//将较大的序号合并到较小根序号中
	if (pRoot > qRoot)
		rootMap[pRoot].nLabel = qRoot;
	else
		rootMap[qRoot].nLabel = pRoot;
	count--;

}


template<class PatchType, class MergeType>
int CuLSM::UnionFind<PatchType,MergeType>::qCount()
{
	return count;
}

// template<class PatchType, class MergeType>
// void CuLSM::UnionFind<PatchType,MergeType>::qMergeMetrics(PatchType& curPatch, int root)
// {
// 	// rootMap[root].nPerimeterByPixel = 
// 	// MergeType()(rootMap[root].nPerimeterByPixel,curPatch.nPerimeterByPixel);//通过产生临时对象调用仿函数
// 	// rootMap[root].nAreaByPixel = 
// 	// MergeType()(rootMap[root].nAreaByPixel,curPatch.nAreaByPixel);//通过产生临时对象调用仿函数
// 	rootMap[root].nMetricsByPixel = 
// 	MergeType()(rootMap[root].nMetricsByPixel,curPatch.nMetricsByPixel);//通过产生临时对象调用仿函数
// }

template<class PatchType, class MergeType>
void CuLSM::UnionFind<PatchType,MergeType>::qRelabel ()
{

	CpuTimer relabelTime;
	relabelTime.start();


	typename map<int, PatchType>::iterator  iter;

	// //这里是否需要先全部存为根节点还需要实验验证。
	// for (iter = rootMap.begin(); iter != rootMap.end(); iter++)
	// {
	// 	int curKey = iter->first;
	// 	PatchType& curPatch = iter->second;
	// 	if (curKey != curPatch.nLabel)//说明不是跟节点
	// 	{
	// 		int root = qFind(curKey);
	// 		curPatch.nLabel = root;
	// 		// qMergeMetrics(curPatch,root);

	// 		rootMap[root].nMetricsByPixel = 
	// 		MergeType()(rootMap[root].nMetricsByPixel,curPatch.nMetricsByPixel);//通过产生临时对象调用仿函数

	// 		// rootMap[root].nAreaByPixel += curPatch.nAreaByPixel;
	// 		// rootMap[root].nPerimeterByPixel += curPatch.nPerimeterByPixel;
	// 	}
	// }

	int i = 0;
	for (iter = rootMap.begin(); iter != rootMap.end(); iter++)
	{
		int curKey = iter->first;
		PatchType& curPatch = iter->second; 
		int curVal = curPatch.nLabel;
		if (curKey == curVal)//说明是根节点
		{
			curPatch.isUseful = TRUE;
			curPatch.nLabel = i;
			i++;
		}
		else
		{
			curPatch.nLabel = rootMap[curVal].nLabel;
		}
	}
	cout << "qRelabel finished!" << endl;
	relabelTime.stop();
	printf("%s = %f\n", VNAME(relabelTime), relabelTime.elapsed());

}

template<class PatchType, class MergeType>
void* CuLSM::UnionFind<PatchType,MergeType>::setVecKeysAndContinueLabels()
{
	// cout<<"patch count:"<<count<<endl;
	// cout<<"sum patch count:"<<rootMap.size()<<endl;
	int rootMapSize = rootMap.size();
	keys = new int[rootMapSize];
	// keys = (int*)malloc(sizeof(int)*count);
	if (keys == NULL)
	{
		cout << "allocate memory false！" << endl;
		exit(-1);
	}
	continueLabels = new int[rootMapSize];
	// continueLabels = (int*)malloc(sizeof(int)*count);
	if (continueLabels == NULL)
	{
		cout << "allocate memory false！" << endl;
		exit(-1);
	}
	int j = 0;
	typename map<int, PatchType>::iterator iter;
	for (iter = rootMap.begin(); iter != rootMap.end(); j++, iter++)
	{
		int curKey = iter->first;
		int curVal = iter->second.nLabel;
		keys[j] = curKey;
		continueLabels[j] = curVal;
		// cout << keys[j] << "\t" << continueLabels[j] << endl;
	}
	cout << "setVecKeysAndContinueLabels finished!" << endl;
}


template<class PatchType, class MergeType>
int* CuLSM::UnionFind<PatchType,MergeType>::getKeys()
{
	return keys;
}

template<class PatchType, class MergeType>
int* CuLSM::UnionFind<PatchType,MergeType>::getContinueLabels()
{
	return continueLabels;
}


template<class PatchType, class MergeType>
void CuLSM::UnionFind<PatchType,MergeType>::qOutputRootMap(std::string Name)
{
	CpuTimer outputResult;
	outputResult.start();
	cout << "outputResult start" << endl;
	std::ofstream f;
	char filename[50];//此处给出可能出现的最大长度，否则会出现堆栈溢出；
	strcpy(filename, (Name + "_patch.csv").c_str());
	f.open(filename, std::ios::out);
	f << "ID" << "," << "\t"
		<< "TYPE" << "," << "\t"
		<< "AREA" << "," << "\t"
		<< "PERIMETER" << "," << "\t"
		<< "PARA" << "," << "\t"
		<< "SHAPE" << "," << "\t"
		<< "FRAC" << "," << "\t"
		<< std::endl;

	typename map<int, PatchType>::iterator  iter;
	int i = 0;
	for (iter = rootMap.begin(); iter != rootMap.end(); iter++)
	{

		int curKey = iter->first;
		PatchType& curPatch = iter->second;
		if (curPatch.isUseful)
		{
			int curLabel = curPatch.nLabel;
			int curType = curPatch.nType;
			//int curArea = curPatch.nAreaByPixel;
			//int curPeri = curPatch.nPerimeterByPixel;
			
			
			//暂时将指数结果直接计算输出
			// int curPeri = curPatch.nPerimeterByPixel;
			// int curPeri = curPatch.nAreaByPixel;
			// int curPeri = curPatch.nMetricsByPixel;
			// f << curPeri << "," << "\t"
			//   << endl;

			// double curArea = ((double)curPatch.nAreaByPixel)*30*30/10000;//areaByPixel*cellSizex*cellSizey
			// int curPeri = curPatch.nPerimeterByPixel * 30 ;
			// double para = (double)curPeri / curArea;
			// double shapeIndex = 0.25 * (double)curPeri / sqrt(curArea)/100;
			// double frac = 2 * log(0.25 * (double)curPeri) / log(curArea*10000);
			f << curLabel << "," << "\t"
			  << curType << "," << "\t"
			//   << curArea << "," << "\t"
			//   << curPeri << "," << "\t"
			//   << para << "," << "\t"
			//   << shapeIndex << "," << "\t"
			//   << frac << "," << "\t"
			  << endl;
		}
	}
	f.close();
	outputResult.stop();
	printf("%s = %f\n", VNAME(outputResult), outputResult.elapsed());
}






/*
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

template<class PatchType, class MergeType>
void mergePatch(int width, int blockNum, int BGvalue, 
	int* h_rowOneValue, int* h_rowTwoValue, int* h_rowOneLabel, int* h_rowTwoLabel, 
	CuLSM::UnionFind<PatchType, MergeType> *Quf)
{
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
*/