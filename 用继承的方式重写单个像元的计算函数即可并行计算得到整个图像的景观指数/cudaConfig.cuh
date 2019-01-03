#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__

class cudaConfig
{
public:
	static void setBlock1Dim(int blockdim){ m_blockdim = blockdim; };
	static void setBlock2Dim(int blockdim1, int blockdim2){ m_blockdim1 = blockdim1; m_blockdim2 = blockdim2; };
	static int getBlockDim1(){ return m_blockdim1; };
	static int getBlockDim2(){ return m_blockdim2; };
	static int getBlockDim(){ return m_blockdim; };

	static dim3 getBlock2D(){ dim3 cudablock(m_blockdim1, m_blockdim2); return cudablock; };
	static dim3 getBlock1D(){ dim3 cudablock(m_blockdim); return cudablock; };
	static dim3 getGrid(int width, int height){ dim3 cudagrid(width% m_blockdim1 == 0 ? width / m_blockdim1 : width / m_blockdim1 + 1, height % m_blockdim2 == 0 ? height / m_blockdim2 : height / m_blockdim2 + 1); return cudagrid; };
	static dim3 getGrid(int nTask){ dim3 cudagrid(nTask%m_blockdim == 0 ? nTask / m_blockdim : nTask / m_blockdim + 1); return cudagrid; };

private:
	static int m_blockdim1;
	static int m_blockdim2;
	static int m_blockdim;
};

int cudaConfig::m_blockdim1 = 32;
int cudaConfig::m_blockdim2 = 16;

int cudaConfig::m_blockdim = 512;


#endif