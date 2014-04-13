#pragma once
#include <stddef.h>
#include <vector>

namespace curoy{
class WaveletReturn
{
    public:
        std::vector<size_t> levelLengths;
        size_t totalLength;
        double* data;

        //getWaveletCoefficients(ofLevel)
        //getLevelOfTransform()
        //~xWaveletTransform() -> delete everything
};
}
