#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>
#include <time.h>
#define VNAME(name) (#name)

struct GpuTimer
{
  cudaEvent_t startTime;
  cudaEvent_t stopTime;

  GpuTimer()
  {
	  cudaEventCreate(&startTime);
	  cudaEventCreate(&stopTime);
  }

  ~GpuTimer()
  {
	  cudaEventDestroy(startTime);
	  cudaEventDestroy(stopTime);
  }

  void start()
  {
	  cudaEventRecord(startTime, 0);
  }

  void stop()
  {
	  cudaEventRecord(stopTime, 0);
  }

  float elapsed()
  {
    float elapsed;
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&elapsed, startTime, stopTime);
    return elapsed;
  }
};


struct CpuTimer
{
	clock_t startTime;
	clock_t endTime;
	void start()
	{
		startTime = clock();
	}
	void stop()
	{
		endTime = clock();
	}
	float elapsed()
	{
		float elapsed = ((float)(endTime - startTime)) / CLOCKS_PER_SEC;
		return elapsed;
	}
};


#endif  /* GPU_TIMER_H__ */
