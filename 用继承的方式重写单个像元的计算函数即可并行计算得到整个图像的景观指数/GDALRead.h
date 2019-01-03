/************************************************************************/
/* This class is wrote for supercomputer to read image use parallel.
/* Not support Block read & write
/* But support multi-thread
/* Author: Y. Yao
/* E-mail: whuyao@foxmail.com
/* Version: v4.0
/************************************************************************/

#ifndef CLASS_GDAL_READ
#define CLASS_GDAL_READ

#include "gdal_priv.h"
#include "ogr_core.h"
#include "ogr_spatialref.h"
#include <iostream>
using namespace std;

class CGDALRead
{
public:
	CGDALRead(void);
	~CGDALRead(void);

public:
	bool loadMetaData(const char* _filename);
	bool loadFrom(const char* _filename);
	unsigned char* read(size_t _row, size_t _col, size_t _band);
	unsigned char* readL(size_t _row, size_t _col, size_t _band);	//extension read-data
	template<class TT> double linRead(double _row, double _col, size_t _band);

public:
	void close();
	bool isValid();

public:
	GDALDataset* poDataset();
	size_t rows();
	size_t cols();
	size_t bandnum();
	size_t datalength();
	double invalidValue();
	unsigned char* imgData();
	GDALDataType datatype();
	double* geotransform();
	char* projectionRef();
	size_t perPixelSize();

public:
	bool world2Pixel(double lat, double lon, double *pcol, double *prow);
	bool pixel2World(double *lat, double *lon, double col, double row);
	bool pixel2Ground(double col,double row,double* pX,double* pY);
	bool ground2Pixel(double X,double Y,double* pcol,double* prow);

protected:
	template<class TT> bool readData();
public:
	template<class TT> bool readDataBlock(int width, int height, int nXOff, int nYOff, int* sub_data)
	{
		if (mpoDataset == NULL)
			return false;
		
		//new space
		unsigned long datalength = width*height*sizeof(TT);
		unsigned char* pData = new unsigned char[(size_t)datalength];

		//raster IO
		CPLErr _err= mpoDataset->RasterIO(GF_Read, nXOff, nYOff, width, height, pData, \
					width, height, mgDataType, mnBands, 0, 0, 0, 0);

		if (_err != CE_None)
		{
			cout<<"CGDALRead::readData : raster io error!"<<endl;
			return false;
		}
		int i,j;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				//注意这里的转换
				unsigned char* temp = &(pData[(i * width + j) * mnPerPixSize]);
				(sub_data)[i*width+j] = (int)(*(TT*)temp);
			}
		}
		delete[] pData;

		return true;
	}
	bool readDataBlockInt(int width, int height, int nXOff, int nYOff, int** sub_data)
	{
		if (mpoDataset == NULL)
			return false;
		
		//new space
		unsigned long datalength = width*height*4;
		// int datalength = width*height*sizeof(int);
		unsigned char* pData = new unsigned char[(size_t)datalength];

		//raster IO
		CPLErr _err= mpoDataset->RasterIO(GF_Read, nXOff, nYOff, width, height, pData, \
					width, height, mgDataType, mnBands, 0, 0, 0, 0);

		if (_err != CE_None)
		{
			cout<<"CGDALRead::readData : raster io error!"<<endl;
			return false;
		}
		int i,j;
		for (i=0; i<height; i++)
		{
			for (j=0; j<width; j++)
			{
				//注意这里的转换
				unsigned char* temp = &(pData[(i*width + j)*mnPerPixSize]);
				(*sub_data)[i*width+j] = (int)(*(int*)temp);
			}
		}
		delete[] pData;

		return true;
	}
//public:
//	template<class TT> TT* transforData();
public:
	template<class TT> TT* transforData()
	{
		// if (mpoDataset == NULL)
		// 	return false;
		
		TT* pdata = new TT[mnRows*mnCols];
		int i, j;

		for (i=0; i<mnRows; i++)
		{
			for (j=0; j<mnCols; j++)
			{
				//注意这里的转换
				pdata[i*mnCols+j] = *(TT*)read(i, j, 0);
			}
		}

		cout<<"read success!"<<endl;
		
		return pdata;
	}

public:
	GDALDataset* mpoDataset;	//=>
	size_t mnRows;					//
	size_t mnCols;					//
	size_t mnBands;				//
	unsigned char* mpData;		//=>
	GDALDataType mgDataType;	//
	size_t mnDatalength;			//=>
	double mpGeoTransform[6];	//
	char msProjectionRef[2048];	//
	char msFilename[2048];		//
	double mdInvalidValue;
	size_t mnPerPixSize;			//=>

public:
	OGRSpatialReferenceH srcSR;
	OGRSpatialReferenceH latLongSR;
	OGRCoordinateTransformationH poTransform;		//pixel->world
	OGRCoordinateTransformationH poTransformT;		//world->pixel
};

#endif



