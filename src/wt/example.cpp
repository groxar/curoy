#include "xWaveletTransform.hpp"
#include <iostream>

using namespace curoy;


int main(int argc, char **argv)
{
    xWaveletTransform transformator;
    double data[] = {6, 12, 15, 15, 14, 12, 120, 116};
    double *transformedData = transformator.doHaarWaveletTransform(data, 8);

    for(int i=0; i<8; ++i){
    	std::cout << transformedData[i] << " ";
    }
}