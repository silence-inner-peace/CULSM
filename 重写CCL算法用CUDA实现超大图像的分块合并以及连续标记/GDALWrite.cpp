#include "GDALWrite.h"
#include "ogrsf_frmts.h"
#include "GDALRead.h"
#include <iostream>
using namespace std;


CGDALWrite::CGDALWrite(void)
{
	mpoDriver = NULL;
	mpoDataset = NULL;
	mnRows = mnCols = mnBands = -1;
	mpData = NULL;
	mgDataType = GDT_Byte;
	mnDatalength = 0;
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


CGDALWrite::~CGDALWrite(void)
{
	close();
}

void CGDALWrite::close()
{
	//write into data
	if (mpoDataset!=NULL && mpData!=NULL)
	{
		mpoDataset->RasterIO(GF_Write, 0, 0, mnCols, mnRows, \
			mpData, mnCols, mnRows, mgDataType, mnBands, 0, 0, 0, 0);
		mpoDataset->FlushCache();
	}


	////release memory
	if (mpoDataset!=NULL)
	{
		GDALClose(mpoDataset);
		mpoDataset = NULL;
	}

	mnRows = mnCols = mnBands = -1;
	
	if (mpData!=NULL)
	{
		delete []mpData;
		mpData = NULL;
	}

	

	mgDataType = GDT_Byte;
	mnDatalength = 0;
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

// 	if (mpoDriver!=NULL)
// 	{
// 		//GDALDestroyDriver(mpoDriver);
// 		delete mpoDriver;
// 		mpoDriver = NULL;
// 	}

}

GDALDriver* CGDALWrite::poDriver()
{
	return mpoDriver;
}

GDALDataset* CGDALWrite::poDataset()
{
	return mpoDataset;
}

size_t CGDALWrite::rows()
{
	return mnRows;
}

size_t CGDALWrite::cols()
{
	return mnCols;
}

size_t CGDALWrite::bandnum()
{
	return mnBands;
}

size_t CGDALWrite::datalength()
{
	return mnDatalength;
}

double CGDALWrite::invalidValue()
{
	return mdInvalidValue;
}

unsigned char* CGDALWrite::imgData()
{
	return mpData;
}

GDALDataType CGDALWrite::datatype()
{
	return mgDataType;
}

double* CGDALWrite::geotransform()
{
	return mpGeoTransform;
}

char* CGDALWrite::projectionRef()
{
	return msProjectionRef;
}

size_t CGDALWrite::perPixelSize()
{
	return mnPerPixSize;
}

bool CGDALWrite::init( const char* _filename, size_t _rows, size_t _cols, size_t _bandnum, double _pGeoTransform[6], const char* _sProjectionRef, GDALDataType _datatype /*= GDT_Byte*/, double _dInvalidVal /*= 0.0f*/ )
{
	close();

	//register
	if(GDALGetDriverCount() == 0)
	{
		GDALAllRegister();
		OGRRegisterAll();
		CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	}

	//load
	mpoDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (mpoDriver == NULL)
	{
		cout<<"CGDALWrite::init : Create poDriver Failed."<<endl;
		close();
		return false;
	}

	//
	strcpy(msFilename, _filename);
	mnRows = _rows;
	mnCols = _cols;
	mnBands = _bandnum;

	for (size_t i=0; i<6; i++)
		mpGeoTransform[i] = _pGeoTransform[i];

	strcpy(msProjectionRef, _sProjectionRef);
	mgDataType = _datatype;
	mdInvalidValue = _dInvalidVal;

	//create podataset
	char** papseMetadata = mpoDriver->GetMetadata();
	mpoDataset = mpoDriver->Create(msFilename, mnCols, mnRows, mnBands, mgDataType, papseMetadata);
	if (mpoDataset == NULL)
	{
		cout<<"CGDALWrite::init : Create poDataset Failed."<<endl;
		close();
		return false;		
	}

	//create others
	srcSR = OSRNewSpatialReference(msProjectionRef); // ground
	latLongSR = OSRCloneGeogCS(srcSR);  //geo
	poTransform =OCTNewCoordinateTransformation(srcSR, latLongSR);
	poTransformT =OCTNewCoordinateTransformation(latLongSR, srcSR);

	//add projection and coordinate
	poDataset()->SetGeoTransform(mpGeoTransform);
	poDataset()->SetProjection(msProjectionRef);
	for (size_t i =0; i<mnBands; i++)
	{
		poDataset()->GetRasterBand(i+1)->SetNoDataValue(mdInvalidValue);
	}

	//create data
	bool bRlt = false;
	switch(mgDataType)
	{
	case GDT_Byte:
		bRlt = createData<unsigned char>();
		break;
	case GDT_UInt16:
		bRlt = createData<unsigned short>();
		break;
	case GDT_Int16:
		bRlt = createData<short>();
		break;
	case GDT_UInt32:
		bRlt = createData<unsigned int>();
		break;
	case GDT_Int32:
		bRlt = createData<int>();
		break;
	case GDT_Float32:
		bRlt = createData<float>();
		break;
	case GDT_Float64:
		bRlt = createData<double>();
		break;
	default:
		cout<<"CGDALWrite::init : unknown data type!"<<endl;
		close();
		return false;
	}

	if (bRlt == false)
	{
		cout<<"CGDALWrite::init : Create data error!"<<endl;
		close();
		return false;
	}

	return true;
}

bool CGDALWrite::init( const char* _filename, CGDALRead* pRead )
{
	if (pRead == NULL)
	{
		cout<<"CGDALWrite::init : CGDALRead Point is Null."<<endl;
		return false;
	}

	return init(_filename, pRead->rows(), pRead->cols(), pRead->bandnum(), \
		pRead->geotransform(), pRead->projectionRef(), pRead->datatype(), 
		pRead->invalidValue());
}

bool CGDALWrite::init( const char* _filename, CGDALRead* pRead, size_t bandnum, GDALDataType _datatype /*= GDT_Byte*/, double _dInvalidVal /*= 0.0f*/ )
{
	if (pRead == NULL)
	{
		cout<<"CGDALWrite::init : CGDALRead Point is Null."<<endl;
		return false;
	}

	return init(_filename, pRead->rows(), pRead->cols(), bandnum, \
				pRead->geotransform(), pRead->projectionRef(), _datatype, 
				_dInvalidVal);

}

template<class TT> bool CGDALWrite::createData()
{
	if (mpoDataset == NULL)
		return false;

	if (mpData!=NULL)
		delete mpData;
	mpData = NULL;
	
	mnPerPixSize = sizeof(TT);
	mnDatalength = mnRows*mnCols*mnBands*mnPerPixSize;
	mpData = new unsigned char[mnDatalength];
	memset(mpData, 0, mnDatalength);
	return true;
}

void CGDALWrite::write( size_t _row, size_t _col, size_t _band, void* pVal )
{
	size_t nloc = (_band*mnRows*mnCols + _row*mnCols + _col)*mnPerPixSize;
	memcpy(mpData+nloc, pVal, mnPerPixSize);	
}

bool CGDALWrite::world2Pixel( double lat, double lon, double *x, double *y )
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

bool CGDALWrite::pixel2World( double *lat, double *lon, double x, double y )
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

bool CGDALWrite::pixel2Ground( double x,double y,double* pX,double* pY )
{
	GDALApplyGeoTransform(mpGeoTransform, x, y, pX, pY);

	return true;
}

bool CGDALWrite::ground2Pixel( double X,double Y,double* px,double* py )
{
	double  adfInverseGeoTransform[6];

	GDALInvGeoTransform(mpGeoTransform, adfInverseGeoTransform);
	GDALApplyGeoTransform(adfInverseGeoTransform, X, Y, px, py);

	return true;
}
