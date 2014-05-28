#include "WaveletTransformator.hpp"
#include "Filter.hpp"
#include "WaveletReturn.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace curoy;
using namespace std;

///This function prints the levelLenghts, totalLenghts and  data from a waveletReturnobject to std::out
void printWaveletReturn(WaveletReturn *waveletReturn)
{
    for(vector<size_t>::iterator it = waveletReturn->levelLengths.begin(); it != waveletReturn->levelLengths.end(); it++){
        cout << *it << " ";
    }
    cout << endl;
    cout << waveletReturn->totalLength << ": ";
    for(int i=0; i<waveletReturn->totalLength; ++i){
        cout << waveletReturn->data[i] << ", ";
    }
    cout << endl;
    cout << endl;
}
//example for the usage of the wavelettransformator class
int main(int argc, char **argv)
{
    //create the instance of the WaveletTransformator
    WaveletTransformator transformator;

    //call the waveletdecomposition on some example signal by specifiing the wavelet by name.
    //Alternatively you can always specify a filter (Filter.hpp) by yourself and call the transformator with both a high and a low-pass filter
    //The allowed predefined filters are haar, db1-db15, sym2-sym15 und coif2-coif5
    double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};
    WaveletReturn* transformedData = transformator.waveletDecomposition(data, 23, 3, "sym14");

    //print out the result
    printWaveletReturn(transformedData);

    //reconstruct the original signal from the transformed data, the information about the levels (-> In Matlab this is the L returned from [C, L] = wavedec(...)).
    double *reconstructedData = transformator.waveletReconstruction(transformedData->data, transformedData->levelLengths, "sym14");


    //print out if the reconstructed signal equals (difference < 0.00001) the original signal
    for(int i = 0; i < 23; ++i)
    {
        cout << ((abs(reconstructedData[i] - data[i]) <= 0.00001) ? "True" : "False") << ", ";
    }
    cout << endl;
}

