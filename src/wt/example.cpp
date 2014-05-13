#include "xWaveletTransform.hpp"
#include "xFilter.hpp"
#include "waveletReturn.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace curoy;
using namespace std;


int main(int argc, char **argv)
{

    double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};
    //double data[] = {1,2,3,1,2,3,4,0};
    xWaveletTransform transformator;

    WaveletReturn* transformedData = transformator.doWaveletTransform(data, 24, 3, "sym20");


    for(vector<size_t>::iterator it = transformedData->levelLengths.begin(); it != transformedData->levelLengths.end(); it++){
        cout << *it << " ";
    }
    cout << endl;
    cout << transformedData->totalLength << ": ";
    for(int i=0; i<transformedData->totalLength; ++i){
    	cout << transformedData->data[i] << ", ";
    }
    cout << endl;
    cout << endl;

    double *reconstructedData = transformator.doWaveletReconstruction(transformedData->data, transformedData->levelLengths, "sym20");
    for(int i = 0; i < 24; ++i)
    {
        cout << ((abs(reconstructedData[i] - data[i]) <= 0.00001) ? 0 : reconstructedData[i] - data[i]) << ", ";
    }
    cout << endl;
}