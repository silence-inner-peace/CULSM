#ifndef __BASE_BLOCK_CLASS_H__
#define __BASE_BLOCK_CLASS_H__
#include "GDALRead.h"
#include <unordered_map>
#include <map>
class PRead
{
public:
	PRead(int col1, int row1, int invalid1){
		col = col1;
		row = row1;
		invalid = invalid1;
	}
public:
	int col;
	int row;
	int invalid;
public:
	int cols(){ return col; }
	int rows(){ return row; }
	int invalidValue(){ return invalid; }
};
template<class T>
class ADD
{
public:
	T operator()(const T in, const T out)
	{
		return in + out;
	}
};

namespace CuLSM
{
	// typedef struct
	// {
	// 	int nLabel;			//斑块序号，标记后的值
	// 	int nType;			//斑块类型（class）从原始数据中获得
	// 	// int nAreaByPixel;	//斑块数量
	// 	int nPerimeterByPixel;//斑块轮廓
	// 	bool isUseful;		//标记是否是根节点，是否有效
	// }Patch;
	
	class Patch
	{
	public:
		int nLabel;			//斑块序号，标记后的值
		int nType;			//斑块类型（class）从原始数据中获得
		// int nAreaByPixel;	//斑块数量
		// int nPerimeterByPixel;//斑块轮廓
		bool isUseful;		//标记是否是根节点，是否有效
	};

	class CPatchLabel
	{
	public:
		int nLabel;			//斑块序号，标记后的值
		int nType;			//斑块类型（class）从原始数据中获得
		bool isUseful;		//标记是否是根节点，是否有效
	};

	template<class PatchType, class MergeType>
	class UnionFind
	{
	private:
		int count;						//连通域的数量
		int *keys;						//存储原始标记值的数组
		int *continueLabels;			//存储重标记后的连续标记值的数组
		
	public:
		map<int, PatchType> rootMap;//存储根节点
	public:
		UnionFind();
		~UnionFind();
		UnionFind(vector<PatchType>& vecAllLabel);//用现在所有的标记初始化map
		void initUF(vector<PatchType>& vecAllLabel);
	public:
		int qFind(int key);//找根节点
		void qUnion(int x, int y);//合并根节点
		int qCount();
		void qRelabel();
		// void qMergeMetrics(PatchType& curPatch, int root);//这个函数需要用户自己定义
		void* setVecKeysAndContinueLabels();
		int* getKeys();
		int* getContinueLabels();
		void qOutputRootMap(std::string Name);
	};



	//描述每个分块
	class dataBlock
	{
	public:
		int mnDataStart;
		int mnDataEnd;
		int mnTaskStart;
		int mnTaskEnd;
		int mnSubDataHeight;
		int mnSubTaskHeight;
		int mnStartTag;
		int mnWidth;
		int mnNodata;
	public:
		int getDataStart();
		int getDataEnd();
		int getTaskStart();
		int getTaskEnd();
		int getDataHeight();
		int getTaskHeight();
		int getStartTag();
		int getWidth();
		int getNoData();
	public:
		bool isFirstBlock()
		{
			if (mnDataStart == mnTaskStart)
				return true;
			else
				return false;
		}
		bool isLastBlock()
		{
			if (mnDataEnd == mnTaskEnd)
				return true;
			else
				return false;
		}
		bool isSplit()
		{
			if ((mnDataStart == mnTaskStart) && (mnDataEnd == mnTaskEnd))
				return false;
			else
				return true;
		}

		//读当前分块的信息，将信息存储于mh_SubData中
	public:
		// read using GDAL
		int* mh_holoUp = NULL;					//当前分块的上一行，若为第一行则一直保持NULL
		int* mh_holoDown = NULL;				//当前分块的下一行，若为最后一行则一直保持NULL
		int* mh_SubData = NULL;					//保存当前分块的src信息
		
		int* mh_LabelVal = NULL;				//保存当前分块的标记值，从GPU传回的标记值
		int* mh_RelabelVal = NULL;				//保存当前分块的标记值,经过处理后的连续标记值
		int  mh_curPatchNum;					//当前分块中斑块的数量
		int* mh_RootPos = NULL;					//记录根节点的一维位置
		
		int* mh_compactSrc = NULL;				//记录原始类型
		int* mh_compactLabel = NULL;			//标记值集合
		int* mh_compactRelabel = NULL;			//重标记值的集合
		// int* mh_compactAreaByPixel = NULL;		//当前分块中每个斑块的面积（用像元数量表示）
		// int* mh_compactPerimeterByPixel = NULL;	//当前分块中每个斑块的周长（用像元为单位长度表示）
	public:
		dataBlock();
		int* getSrcData();
		int* getLabelData();
		void loadBlockData(CGDALRead* pread);//int width, int data_height, int data_start, int** h_subData, CGDALRead* pread
		void loadBlockData(PRead* pread, int* src);//int width, int data_height, int data_start, int** h_subData, CGDALRead* pread
		void freeSubData();		//释放当前分块
		void freeLabel();		//释放当前分块
		void freeRelabelVal();	//释放重标记数组
		void freeCompactLabel();//释放标记集合
		void freeCompactRelabel();
		// void freeCompactMetricsByPixel();
		// void freeCompactAreaByPixel();
		// void freeCompactPerimeterByPixel();
		void freeHoloUp();
		void freeHoloDown();
		void freeRootPos();
		void freeCompactSrc();
		void getCompactRelabel(CuLSM::UnionFind<Patch,ADD<int>>* Quf);	//得到原始标记值对应的连续标记值
		void getContinueLabelVal();		//将mh_LabelVal中的值修改为连续的标记值,并行
		void getContinueLabelVal(CuLSM::UnionFind<CuLSM::CPatchLabel,ADD<int>>* Quf);		//串行，用union-find查找，快很多
		void printLabelVal();	//标记值打印
		void printRelabelVal();	//打印连续标记值
		void freeDataBlock(){
			freeSubData();
			freeLabel();
			freeRelabelVal();
			freeCompactLabel();
			freeCompactRelabel();
			// freeCompactMetricsByPixel();
			// freeCompactAreaByPixel();
			// freeCompactPerimeterByPixel();
			freeCompactSrc();
			freeHoloUp();
			freeHoloDown();
			freeRootPos();
		}

	};
}

template<class PatchType, class MergeType>
void mergePatch(int width, int blockNum, int BGvalue, 
	int* h_rowOneValue, int* h_rowTwoValue, int* h_rowOneLabel, int* h_rowTwoLabel, 
	CuLSM::UnionFind<PatchType, MergeType> *Quf);

#endif  /* __BASE_BLOCK_CLASS_H__ */
