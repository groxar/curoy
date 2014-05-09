#pragma once
#include <stddef.h>
#include <vector>
#include <iostream>

namespace curoy{
class WaveletReturn
{
    public:
        std::vector<size_t> levelLengths;
        size_t totalLength;
        double* data;

        //getWaveletCoefficients(ofLevel)
        //getLevelOfTransform()

        WaveletReturn(){
            data = 0;
        }
        ~WaveletReturn(){
            if(data)
            {
                delete[] data;
            }

        }
};
}
