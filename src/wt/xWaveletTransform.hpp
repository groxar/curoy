#pragma once
#include <stddef.h>
#include "xFilter.hpp"
#include "waveletReturn.hpp"

namespace curoy{
class xWaveletTransform
{
    public:
        WaveletReturn* doWaveletTransform(const double* data, size_t length, size_t level, xFilter filter);
    //private:
        WaveletReturn* doOneLevelWaveletTransform(const double* data, size_t length, xFilter filter);
};
}