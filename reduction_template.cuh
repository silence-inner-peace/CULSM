
// template<unsigned int numThreads>
// __global__ void reduction_kernel(int *out, const unsigned int *in, size_t N)
// {
//     extern __shared__ int sPartials[];
//     int sum = 0;
//     const int tid = threadIdx.x;

//     for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
//     {
//         sum += in[i];
//     }
//     sPartials[tid] = sum; 
//     __syncthreads();

//     unsigned int floorPow2 = blockDim.x;
//     if (floorPow2 & (floorPow2 - 1))
//     {
//         while(floorPow2 & (floorPow2 - 1))
//         {
//             floorPow2 &= (floorPow2 - 1);
//         }
//         if (tid >= floorPow2)
//         {
//             sPartials[tid - floorPow2] += sPartials[tid];
//         }
//         __syncthreads();
//     }

//     if (floorPow2 >= 1024)
//     {
//         if (tid < 512) sPartials[tid] += sPartials[tid + 512];
//         __syncthreads();
//     }
//     if (floorPow2 >= 512)
//     {
//         if (tid < 256) sPartials[tid] += sPartials[tid + 256];
//         __syncthreads();
//     }
//     if (floorPow2 >= 256)
//     {
//         if (tid < 128) sPartials[tid] += sPartials[tid + 128];
//         __syncthreads();
//     }
//     if (floorPow2 >= 128)
//     {
//         if (tid < 64) sPartials[tid] += sPartials[tid + 64];
//         __syncthreads();
//     }

//     if (tid < 32)
//     {
//         volatile int *wsSum = sPartials;
//         if (floorPow2 >= 64) wsSum[tid] += wsSum[tid + 32];
//         if (floorPow2 >= 32) wsSum[tid] += wsSum[tid + 16];
//         if (floorPow2 >= 16) wsSum[tid] += wsSum[tid + 8];
//         if (floorPow2 >= 8) wsSum[tid] += wsSum[tid + 4];
//         if (floorPow2 >= 4) wsSum[tid] += wsSum[tid + 2];
//         if (floorPow2 >= 2) wsSum[tid] += wsSum[tid + 1];
//         if (tid == 0)
//         {
//             volatile int *wsSum = sPartials;
//             atomicAdd(out, wsSum[0]);
//         }
//     }
// }

// template<unsigned int numThreads>
// void reduction_template(int *answer, const unsigned int *in, const size_t N, const int numBlocks)
// {
//     unsigned int sharedSize = numThreads * sizeof(int);
//     cudaMemset(answer, 0, sizeof(int));

//     reduction_kernel<numThreads><<<numBlocks, numThreads, sharedSize>>>(answer, in, N);
// }

// void reduction_t(int *answer, const unsigned int *in, const size_t N, const int numBlocks, int numThreads)
// {
//     switch (numThreads)
//     {
//         case 1: reduction_template<1>(answer, in, N, numBlocks); break;
//         case 2: reduction_template<2>(answer, in, N, numBlocks); break;
//         case 4: reduction_template<4>(answer, in, N, numBlocks); break;
//         case 8: reduction_template<8>(answer, in, N, numBlocks); break;
//         case 16: reduction_template<16>(answer, in, N, numBlocks); break;
//         case 32: reduction_template<32>(answer, in, N, numBlocks); break;
//         case 64: reduction_template<64>(answer, in, N, numBlocks); break;
//         case 128: reduction_template<128>(answer, in, N, numBlocks); break;
//         case 256: reduction_template<256>(answer, in, N, numBlocks); break;
//         case 512: reduction_template<512>(answer, in, N, numBlocks); break;
//         case 1024: reduction_template<1024>(answer, in, N, numBlocks); break;
//     }
// }

#ifndef __REDUCTION_TEMPLATE_CUH__
#define __REDUCTION_TEMPLATE_CUH__
template<unsigned int numThreads, class T1, class T2>
__global__ void reduction_kernel(int *out, const T1 *in, T2 N)
{
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;

    for (T2 i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }
    sPartials[tid] = sum; 
    __syncthreads();

    unsigned int floorPow2 = blockDim.x;
    if (floorPow2 & (floorPow2 - 1))
    {
        while(floorPow2 & (floorPow2 - 1))
        {
            floorPow2 &= (floorPow2 - 1);
        }
        if (tid >= floorPow2)
        {
            sPartials[tid - floorPow2] += sPartials[tid];
        }
        __syncthreads();
    }

    if (floorPow2 >= 1024)
    {
        if (tid < 512) sPartials[tid] += sPartials[tid + 512];
        __syncthreads();
    }
    if (floorPow2 >= 512)
    {
        if (tid < 256) sPartials[tid] += sPartials[tid + 256];
        __syncthreads();
    }
    if (floorPow2 >= 256)
    {
        if (tid < 128) sPartials[tid] += sPartials[tid + 128];
        __syncthreads();
    }
    if (floorPow2 >= 128)
    {
        if (tid < 64) sPartials[tid] += sPartials[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile int *wsSum = sPartials;
        if (floorPow2 >= 64) wsSum[tid] += wsSum[tid + 32];
        if (floorPow2 >= 32) wsSum[tid] += wsSum[tid + 16];
        if (floorPow2 >= 16) wsSum[tid] += wsSum[tid + 8];
        if (floorPow2 >= 8) wsSum[tid] += wsSum[tid + 4];
        if (floorPow2 >= 4) wsSum[tid] += wsSum[tid + 2];
        if (floorPow2 >= 2) wsSum[tid] += wsSum[tid + 1];
        if (tid == 0)
        {
            volatile int *wsSum = sPartials;
            atomicAdd(out, wsSum[0]);
        }
    }
}

template<unsigned int numThreads, class T1, class T2>
void reduction_template(int *answer, const T1 *in, const T2 N, const int numBlocks)
{
    unsigned int sharedSize = numThreads * sizeof(int);
    cudaMemset(answer, 0, sizeof(int));

    reduction_kernel<numThreads><<<numBlocks, numThreads, sharedSize>>>(answer, in, N);
}

template <class T1, class T2>
void reduction_t(int *answer, const T1 *in, const T2 N, const int numBlocks, int numThreads)
{
    switch (numThreads)
    {
        case 1: reduction_template<1>(answer, in, N, numBlocks); break;
        case 2: reduction_template<2>(answer, in, N, numBlocks); break;
        case 4: reduction_template<4>(answer, in, N, numBlocks); break;
        case 8: reduction_template<8>(answer, in, N, numBlocks); break;
        case 16: reduction_template<16>(answer, in, N, numBlocks); break;
        case 32: reduction_template<32>(answer, in, N, numBlocks); break;
        case 64: reduction_template<64>(answer, in, N, numBlocks); break;
        case 128: reduction_template<128>(answer, in, N, numBlocks); break;
        case 256: reduction_template<256>(answer, in, N, numBlocks); break;
        case 512: reduction_template<512>(answer, in, N, numBlocks); break;
        case 1024: reduction_template<1024>(answer, in, N, numBlocks); break;
    }
}
#endif