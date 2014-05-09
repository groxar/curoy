#pragma once
#include <stddef.h>
#include "xFilter.hpp"
#include "waveletReturn.hpp"
#include <vector>

namespace curoy{
class xWaveletTransform
{
    public:
        WaveletReturn* doWaveletTransform(const double* data, size_t length, size_t level, xFilter filter);
    //private:
        WaveletReturn* doOneLevelWaveletTransform(const double* data, size_t length, xFilter filter);

        double *doWaveletReconstruction(const double *data, std::vector<size_t> levelLengths, xFilter filter);

        void convolutionAndUpsampling(const double *data, double* out, double* filterCoeff, size_t inputLength, size_t outLength, size_t filterLength);
};
}