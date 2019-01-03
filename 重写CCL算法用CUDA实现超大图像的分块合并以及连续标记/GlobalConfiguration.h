#ifndef __GLOBAL_CONFIGURATION_H__
#define __GLOBAL_CONFIGURATION_H__
#include <string>
#include "Array2D.h"
#include "Array2D_CUDA.h"
//这个类用于定义所有的用户需要配置的信息
//包括：土地利用类型描述表，类型相似度表、对比度表……
	

static const int NUM_CLASSES = 10;	//最多可能出现多少类
class GlobalConfiguration
{
public:
	bool USE_DIAGS;		//是否使用对角线（选择4/8连通）
	bool *USE;			//要参与计算的用地类型
	Array2D< Cutype<bool> >* d_USE;

	bool SAVE_CCL;		//是否保留连通域发现的结果

	float *CONTRAST;	//类之间的对比度权重
	float *SIMILAR;		//相似度
	int *DEEP_INFLUENCE;//类之间相互影响的深度
	
public:
	GlobalConfiguration();
	void set_USE_DIAGS(bool b);
	void set_USE(bool* use);
	void set_SAVE_CCL(bool b);


	void set_CONTRAST(float* contrast);
	void set_CONTRAST(string file_name);
	void set_SIMILAR(float* similar);
	void set_SIMILAR(string file_name);
	void set_DEEP_INFLUENCE(int* deep_influence);
	void set_DEEP_INFLUENCE(string file_name);



};

GlobalConfiguration::GlobalConfiguration()
{
	USE_DIAGS = 1;
	USE = new bool[NUM_CLASSES];
	d_USE = new Array2D< Cutype<bool> >(NUM_CLASSES);

	SAVE_CCL = 1;
	
	CONTRAST = new float[NUM_CLASSES*NUM_CLASSES];
	SIMILAR = new float[NUM_CLASSES*NUM_CLASSES];
	DEEP_INFLUENCE = new int[NUM_CLASSES*NUM_CLASSES];
}


void GlobalConfiguration::set_USE_DIAGS(bool b)
{
	USE_DIAGS = b;
}
void GlobalConfiguration::set_SAVE_CCL(bool b)
{
	SAVE_CCL = b;
}

void GlobalConfiguration::set_USE(bool* use)
{
	if(use!=NULL)
	{
		for (int i = 0; i < NUM_CLASSES; ++i)
		{
			USE[i] = use[i];
		}	
	}
	else
	{
		for (int i = 0; i < NUM_CLASSES; ++i)
		{
			USE[i] = 1;
		}	
	}
	d_USE->set(USE);
}

void GlobalConfiguration::set_CONTRAST(float* contrast)
{

}
void GlobalConfiguration::set_CONTRAST(string file_name)
{

}
void GlobalConfiguration::set_SIMILAR(float* similar)
{

}
void GlobalConfiguration::set_SIMILAR(string file_name)
{

}
void GlobalConfiguration::set_DEEP_INFLUENCE(int* deep_influence)
{

}
void GlobalConfiguration::set_DEEP_INFLUENCE(string file_name)
{

}



















#endif