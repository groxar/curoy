#include "WaveletTransformator.hpp"
#include "Filter.hpp"
#include "WaveletReturn.hpp"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace curoy;
using namespace std;

static std::chrono::time_point<std::chrono::high_resolution_clock> start;
void startChrono(){
    start = std::chrono::system_clock::now();
}
void endChrono(const string message, int length){
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
    cout << message << "-" << length <<": "<< elapsed_seconds.count() <<"s"<< endl;
    start = std::chrono::system_clock::now();

}
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

//example for the usage of the wavelettransformator class
int main(int argc, char **argv)
{
    srand(346071234.0);
    //create the instance of the WaveletTransformator
    WaveletTransformator transformator;

    //call the waveletdecomposition on some example signal by specifiing the wavelet by name.
    //Alternatively you can always specify a filter (Filter.hpp) by yourself and call the transformator with both a high and a low-pass filter
    //The allowed predefined filters are haar, db1-db15, sym2-sym15 und coif2-coif5
    double *data;
    WaveletReturn* transformedData = 0;

    for(int j = 24; j <= 10000; j *= 2)
    {
        data = new double[j];
        for(int k = 0; k < j; k++)
        {
            data[k] = fRand(-1000, 1000);
        }

        startChrono();
        for(int i= 0; i < 10000; ++i){
            transformedData = transformator.waveletDecomposition(data, j, 3, "haar");
            delete transformedData;
        }
       endChrono("wt-haar-transform", j);
       delete[] data;
    }
}

