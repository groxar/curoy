#pragma once
#include <stddef.h>
#include "Filter.hpp"
#include "WaveletReturn.hpp"
#include "IPadding.hpp"
#include <vector>
#include <string>

namespace curoy{
class WaveletTransformator
{
    public:
        WaveletReturn* waveletDecomposition(const double* data, size_t length, size_t level, std::string waveletName);

        WaveletReturn* waveletDecomposition(const double* data, size_t length, size_t level, std::string waveletName, IPadding *padding);

        WaveletReturn* waveletDecomposition(const double* data, size_t length, size_t level, Filter filter);

        WaveletReturn* waveletDecomposition(const double* data, size_t length, size_t level, Filter filter, IPadding *padding);

        double *waveletReconstruction(const double *data, std::vector<size_t> levelLengths, std::string waveletName);

        double *waveletReconstruction(const double *data, std::vector<size_t> levelLengths, Filter filter);

    private:

        WaveletReturn* oneLevelWaveletDecomposition(const double* data, size_t length, Filter filter, IPadding *padding);

        void convolutionAndUpsampling(const double *data, double* out, double* filterCoeff, size_t inputLength, size_t outLength, size_t filterLength);
};
}