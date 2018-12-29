#ifndef __DERIVED_PERI_BLOCK_CLASS_H__
#define __DERIVED_PERI_BLOCK_CLASS_H__
#include "AbstractBlockClass.h"
namespace CuLSM
{
	class myPatch : public CuLSM::Patch
	{
	public:
		// int nPerimeterByPixel;//斑块轮廓
		// int nAreaByPixel;//斑块轮廓
		int nMetricsByPixel;
	};
	class derivedMetricsBlockClass:public CuLSM::dataBlock
	{
	public:
		int* mh_compactMetricsByPixel;
	public:
		derivedMetricsBlockClass()
		{
			mh_compactMetricsByPixel = NULL;
		}
		~derivedMetricsBlockClass()
		{
			if (this->mh_compactMetricsByPixel != NULL)
			{
				printf("%s\n", "freeCompactMetricsmeterByPixel");
				free(this->mh_compactMetricsByPixel);
				mh_compactMetricsByPixel = NULL;
			}
		}
		
	};


	// class derivedPeriBlockClass:public CuLSM::dataBlock
	// {
	// public:
	// 	int* mh_compactPerimeterByPixel;	//当前分块中每个斑块的周长（用像元为单位长度表示）
	// public:
	// 	derivedPeriBlockClass()
	// 	{
	// 		mh_compactPerimeterByPixel = NULL;
	// 	}
	// 	~derivedPeriBlockClass()
	// 	{
	// 		if (this->mh_compactPerimeterByPixel != NULL)
	// 		{
	// 			printf("%s\n", "freeCompactPerimeterByPixel");
	// 			free(this->mh_compactPerimeterByPixel);
	// 			mh_compactPerimeterByPixel = NULL;
	// 		}
	// 	}
		
	// };
	// class derivedAreaBlockClass:public CuLSM::dataBlock
	// {
	// public:
	// 	int* mh_compactAreaByPixel;
	// public:
	// 	derivedAreaBlockClass()
	// 	{
	// 		mh_compactAreaByPixel = NULL;
	// 	}
	// 	~derivedAreaBlockClass()
	// 	{
	// 		if (this->mh_compactAreaByPixel != NULL)
	// 		{
	// 			printf("%s\n", "freeCompactAreameterByPixel");
	// 			free(this->mh_compactAreaByPixel);
	// 			mh_compactAreaByPixel = NULL;
	// 		}
	// 	}
		
	// };


}

#endif // __DERIVED_PERI_BLOCK_CLASS_H__