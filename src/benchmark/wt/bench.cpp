#include "bench.hpp"
#include <math.h>

namespace curoy{

    void doHaarWaveletTransform(const double* data, double* result, int length)
    {
        int half = length >> 1;

        for (size_t i = 0; i < half; i++)
        {
            size_t k = (i << 1);
            result[i] = (data[k] + data[k + 1]) / sqrt(2.0);
            result[i + half] = (data[k] - data[k + 1]) / sqrt(2.0);
        }
    }

    double* xBench::doTransform(const double* data, int length)
    {
        double* transformedData = new double[length * 5000];
        for(int i = 0; i < 5000; i++){
            doHaarWaveletTransform(data, (transformedData + i * length), length);
        }
        return transformedData;
    }
}