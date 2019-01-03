#include "GDALRead.h"
#include "ogrsf_frmts.h"
#include <iostream>
using namespace std;

CGDALRead::CGDALRead(void)
{
	mpoDataset = NULL;
	mpData = NULL;
	mgDataType = GDT_Byte;
	mnRows = mnCols = mnBands = -1;
	mnDatalength = -1;
	mpData = NULL;
	memset(mpGeoTransform, 0, 6*sizeof(double));
	strcpy(msProjectionRef, "");
	strcpy(msFilename, "");
	mdInvalidValue = 0.0f;
	mnPerPixSize = 1;

	//
	srcSR = NULL;
	latLongSR = NULL;
	poTransform = NULL;
	poTransformT = NULL;
}


CGDALRead::~CGDALRead(void)
{
	close();
}

void CGDALRead::close()
{
	if (mpoDataset != NULL)
	{
		GDALClose(mpoDataset);
		mpoDataset = NULL;		
	}

	if (mpData != NULL)
	{
		delete []mpData;
		mpData = NULL;
	}

	mgDataType = GDT_Byte;
	mnDatalength = -1;
	mnRows = mnCols = mnBands = -1;
	mpData = NULL;
	memset(mpGeoTransform, 0, 6*sizeof(double));
	strcpy(msProjectionRef, "");
	strcpy(msFilename, "");
	mdInvalidValue = 0.0f;
	mnPerPixSize = 1;

	//destory
	if (poTransform!=NULL)
		OCTDestroyCoordinateTransformation(poTransform);
	poTransform = NULL;

	if (poTransformT!=NULL)
		OCTDestroyCoordinateTransformation(poTransformT);
	poTransformT = NULL;

	if (latLongSR != NULL)
		OSRDestroySpatialReference(latLongSR);
	latLongSR = NULL;

	if (srcSR!=NULL)
		OSRDestroySpatialReference(srcSR);
	srcSR = NULL;

	
}

bool CGDALRead::isValid()
{
	if (mpoDataset == NULL || mpData == NULL)
	{
		return false;
	}

	return true;

}

GDALDataset* CGDALRead::poDataset()
{
	return mpoDataset;
}

size_t CGDALRead::rows()
{
	return mnRows;
}

size_t CGDALRead::cols()
{
	return mnCols;
}

size_t CGDALRead::bandnum()
{
	return mnBands;
}

unsigned char* CGDALRead::imgData()
{
	return mpData;
}

GDALDataType CGDALRead::datatype()
{
	return mgDataType;
}

double* CGDALRead::geotransform()
{
	return mpGeoTransform;
}

char* CGDALRead::projectionRef()
{
	return msProjectionRef;
}

size_t CGDALRead::datalength()
{
	return mnDatalength;
}

double CGDALRead::invalidValue()
{
	return mdInvalidValue;
}

size_t CGDALRead::perPixelSize()
{
	return mnPerPixSize;
}

bool CGDALRead::loadMetaData( const char* _filename )
{
	close();

	//register
	if(GDALGetDriverCount() == 0)
	{
		GDALAllRegister();
		OGRRegisterAll();
		CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	}

	//open image
	mpoDataset = (GDALDataset*)GDALOpenShared(_filename, GA_ReadOnly);

	if (mpoDataset == NULL)
	{
		cout<<"CGDALRead::loadFrom : read file error!"<<endl;
		return false;
	}

	strcpy(msFilename, _filename);

	//get attribute
	mnRows = mpoDataset->GetRasterYSize();
	mnCols = mpoDataset->GetRasterXSize();
	mnBands = mpoDataset->GetRasterCount();
	mgDataType = mpoDataset->GetRasterBand(1)->GetRasterDataType();
	mdInvalidValue = mpoDataset->GetRasterBand(1)->GetNoDataValue();

	//mapinfo
	mpoDataset->GetGeoTransform(mpGeoTransform);
	strcpy(msProjectionRef, mpoDataset->GetProjectionRef());

	srcSR = OSRNewSpatialReference(msProjectionRef); // ground
	latLongSR = OSRCloneGeogCS(srcSR);  //geo
	poTransform =OCTNewCoordinateTransformation(srcSR, latLongSR);
	poTransformT =OCTNewCoordinateTransformation(latLongSR, srcSR);

	

	//get data
	switch(mgDataType)
	{
	case GDT_Byte:
		mnPerPixSize = sizeof(unsigned char);
		break;
	case GDT_UInt16:
		mnPerPixSize = sizeof(unsigned short);
		break;
	case GDT_Int16:
		mnPerPixSize = sizeof(short);
		break;
	case GDT_UInt32:
		mnPerPixSize = sizeof(unsigned int);
		break;
	case GDT_Int32:
		mnPerPixSize = sizeof(int);
		break;
	case GDT_Float32:
		mnPerPixSize = sizeof(float);
		break;
	case GDT_Float64:
		mnPerPixSize = sizeof(double);
		break;
	default:
		cout<<"CGDALRead::loadFrom : unknown data type!"<<endl;
		close();
		return false;
	}

	return true;
}

bool CGDALRead::loadFrom( const char* _filename )
{
	//close fore image
	close();

	//register
	if(GDALGetDriverCount() == 0)
	{
		GDALAllRegister();
		OGRRegisterAll();
		CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	}

	//open image
	mpoDataset = (GDALDataset*)GDALOpenShared(_filename, GA_ReadOnly);

	if (mpoDataset == NULL)
	{
		cout<<"CGDALRead::loadFrom : read file error!"<<endl;
		return false;
	}

	strcpy(msFilename, _filename);

	//get attribute
	mnRows = mpoDataset->GetRasterYSize();
	mnCols = mpoDataset->GetRasterXSize();
	mnBands = mpoDataset->GetRasterCount();
	mgDataType = mpoDataset->GetRasterBand(1)->GetRasterDataType();
	mdInvalidValue = mpoDataset->GetRasterBand(1)->GetNoDataValue();

	//mapinfo
	mpoDataset->GetGeoTransform(mpGeoTransform);
	strcpy(msProjectionRef, mpoDataset->GetProjectionRef());

	srcSR = OSRNewSpatialReference(msProjectionRef); // ground
	latLongSR = OSRCloneGeogCS(srcSR);  //geo
	poTransform =OCTNewCoordinateTransformation(srcSR, latLongSR);
	poTransformT =OCTNewCoordinateTransformation(latLongSR, srcSR);

	

	//get data
	bool bRlt = false;
	switch(mgDataType)
	{
	case GDT_Byte:
		mnPerPixSize = sizeof(unsigned char);
		bRlt = readData<unsigned char>();
		break;
	case GDT_UInt16:
		mnPerPixSize = sizeof(unsigned short);
		bRlt = readData<unsigned short>();
		break;
	case GDT_Int16:
		mnPerPixSize = sizeof(short);
		bRlt = readData<short>();
		break;
	case GDT_UInt32:
		mnPerPixSize = sizeof(unsigned int);
		bRlt = readData<unsigned int>();
		break;
	case GDT_Int32:
		mnPerPixSize = sizeof(int);
		bRlt = readData<int>();
		break;
	case GDT_Float32:
		mnPerPixSize = sizeof(float);
		bRlt = readData<float>();
		break;
	case GDT_Float64:
		mnPerPixSize = sizeof(double);
		bRlt = readData<double>();
		break;
	default:
		cout<<"CGDALRead::loadFrom : unknown data type!"<<endl;
		close();
		return false;
	}

	if (bRlt == false)
	{
		cout<<"CGDALRead::loadFrom : read data error!"<<endl;
		close();
		return false;
	}


	return true;
}

template<class TT> bool CGDALRead::readData()
{
	if (mpoDataset == NULL)
		return false;
	
	//new space
	mnDatalength = mnRows*mnCols*mnBands*sizeof(TT);
	mpData = new unsigned char[(size_t)mnDatalength];

	//raster IO
	CPLErr _err= mpoDataset->RasterIO(GF_Read, 0, 0, mnCols, mnRows, mpData, \
				mnCols, mnRows, mgDataType, mnBands, 0, 0, 0, 0);

	if (_err != CE_None)
	{
		cout<<"CGDALRead::readData : raster io error!"<<endl;
		return false;
	}
	
	return true;
}


//template<class TT> TT* CGDALRead::transforData()
//{
//	// if (mpoDataset == NULL)
//	// 	return false;
//
//	TT* pdata = new TT[mnRows*mnCols];
//	int i, j;
//
//	for (i = 0; i<mnRows; i++)
//	{
//		for (j = 0; j<mnCols; j++)
//		{
//			//注意这里的转换
//			pdata[i*mnCols + j] = *(TT*)read(i, j, 0);
//		}
//	}
//
//	cout << "read success!" << endl;
//
//	return pdata;
//}


unsigned char* CGDALRead::read( size_t _row, size_t _col, size_t _band )
{
	return &(mpData[(_band*mnRows*mnCols + _row*mnCols + _col)*mnPerPixSize]);
}

// unsigned char* CGDALRead::readL( size_t _row, size_t _col, size_t _band )
// {
// 	//if out of rect, take mirror
// 	if (_row < 0)
// 		_row = -_row;
// 	else if (_row >= mnRows)
// 		_row = mnRows - (_row - (mnRows - 1));

// 	if (_col < 0)
// 		_col = -_col;
// 	else if (_col >= mnCols)
// 		_col = mnCols - (_col - (mnCols - 1));

// 	return &(mpData[(_band*mnRows*mnCols + _row*mnCols + _col)*mnPerPixSize]);
// }

// template<class TT> double CGDALRead::linRead( double _row, double _col, size_t _band )
// {
// 	TT val[4];
// 	double t1, t2, t, tx1, ty1;

// 	//calculate the excursion
// 	float xpos = _row - size_t(_row);
// 	float ypos = _col - size_t(_col);

// 	//get the pixel value of 4-neighbour
// 	tx1 = _row+1.0; ty1 = _col+1.0;

// 	val[0] =*(TT*)ReadL(size_t(_row), size_t(_col), bands);	//band
// 	val[1] =*(TT*)ReadL(size_t(tx1), size_t(_col), bands);
// 	val[2] =*(TT*)ReadL(size_t(_row), size_t(ty1), bands);
// 	val[3] =*(TT*)ReadL(size_t(tx1), size_t(ty1), bands);

// 	//y-direction size_terpolation
// 	t1 = (1-ypos)*(double)val[0]+ypos*(double)val[2];
// 	t2 = (1-ypos)*(double)val[1]+ypos*(double)val[3];

// 	//x-direction size_terpolation
// 	t = (1-xpos)*t1 + xpos*t2;

// 	return (double)t;
// }

bool CGDALRead::world2Pixel( double lat, double lon, double *x, double *y )
{
// 	if (poTransformT==NULL)
// 	{
// 		poTransformT =OCTNewCoordinateTransformation(latLongSR, srcSR);
// 	}

	if(poTransformT != NULL)
	{
		double height;
		OCTTransform(poTransformT,1, &lon, &lat, &height);

		double  adfInverseGeoTransform[6];
		GDALInvGeoTransform(mpGeoTransform, adfInverseGeoTransform);
		GDALApplyGeoTransform(adfInverseGeoTransform, lon,lat, x, y);
	
		return true;
	}
	else
	{
		return false;
	}

}

bool CGDALRead::pixel2World( double *lat, double *lon, double x, double y )
{
	if (poTransform!=NULL)
	{
		OCTDestroyCoordinateTransformation(poTransform);
		poTransform =OCTNewCoordinateTransformation(latLongSR, srcSR);
	}

	GDALApplyGeoTransform(mpGeoTransform, x, y, lon, lat);

	if(poTransform != NULL)
	{
		double height;
		OCTTransform(poTransform,1, lon, lat, &height);
		return true;
	}
	else
	{
		return false;
	}
}

bool CGDALRead::pixel2Ground( double x,double y,double* pX,double* pY )
{
	GDALApplyGeoTransform(mpGeoTransform, x, y, pX, pY);

	return true;
}

bool CGDALRead::ground2Pixel( double X,double Y,double* px,double* py )
{
	double  adfInverseGeoTransform[6];

	GDALInvGeoTransform(mpGeoTransform, adfInverseGeoTransform);
	GDALApplyGeoTransform(adfInverseGeoTransform, X, Y, px, py);

	return true;
}



