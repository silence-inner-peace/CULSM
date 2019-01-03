/************************************************************************/
/* This class is wrote for supercomputer to write image use parallel.
/* Not support Block read & write
/* But support multi-thread
/* Author: Y. Yao
/* E-mail: whuyao@foxmail.com
/* Version: v4.0
/************************************************************************/

#ifndef CLASS_GDAL_WRITE
#define CLASS_GDAL_WRITE

#include "gdal_priv.h"
#include "ogr_core.h"
#include "ogr_spatialref.h"

class CGDALRead;

class CGDALWrite
{
public:
	CGDALWrite(void);
	~CGDALWrite(void);

public:
	bool init(const char* _filename, size_t _rows, size_t _cols, size_t _bandnum,\
				double _pGeoTransform[6], const char* _sProjectionRef, \
				GDALDataType _datatype = GDT_Byte, \
				double _dInvalidVal = 0.0f);

	bool init(const char* _filename, CGDALRead* pRead);

	bool init(const char* _filename, CGDALRead* pRead, size_t bandnum, \
				GDALDataType _datatype = GDT_Byte, \
				double _dInvalidVal = 0.0f);

	void write(size_t _row, size_t _col, size_t _band, void* pVal);

public:
	void close();

public:
	GDALDriver* poDriver();
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
	template<class TT> bool createData();

protected:
	GDALDriver* mpoDriver;		//can not release this, maybe cause some memory error!
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
	double mdInvalidValue;		//
	size_t mnPerPixSize;			//=>

public:
	OGRSpatialReferenceH srcSR;
	OGRSpatialReferenceH latLongSR;
	OGRCoordinateTransformationH poTransform;	//pixel->world
	OGRCoordinateTransformationH poTransformT;	//world->pixel
};



#endif
