#include "bench.hu"
#include <cuda.h>
#include <cuda_runtime.h>

namespace curoy{

    __global__ void doHaarTransform(const double* data, double* result, int length, int threadCount){
        int pos = blockDim.x * blockIdx.x + threadIdx.x;
        int resultPos = length * pos;
		double d_k = data[k];
		double d_kn = data[k+1];

        if(pos < threadCount){
            int half = length >> 1;
            for (int i = 0; i < half; i++)
            {
                int k = (i << 1);
                result[resultPos + i] = (d_k + d_kn) / sqrt(2.0);
                result[resultPos + i + half] = (d_k - d_kn) / sqrt(2.0);
            }
        }
    }

    double* cuBench::doTransform(const double* data, int length)
    {
        int threadCount = 5000;
        double* deviceResult;
        double* hostResult = new double[threadCount * length];
        double* deviceData;
        cudaMalloc(&deviceResult, threadCount * sizeof(double) * length);
        cudaMalloc(&deviceData, length * sizeof(double));
        cudaMemcpy(deviceData, data, length * sizeof(double), cudaMemcpyHostToDevice);
        doHaarTransform<<<5, 1000>>>(deviceData, deviceResult, length, threadCount);
        cudaMemcpy(hostResult, deviceResult, threadCount * length * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(deviceResult);
        cudaFree(deviceData);
        return hostResult;
    }
}
