#include "xWaveletTransform.hpp"
#include "xFilter.hpp"
#include "waveletReturn.hpp"
#include <iostream>
#include <vector>

using namespace curoy;
using namespace std;


int main(int argc, char **argv)
{
    xFilter db4;
    //db1 / haar
    double hi[] = {1, -1};
    double lo[] = {1, 1};
    //db2
    //double hi[] = {-0.1830127, -0.3169873, 1.1830127, -0.6830127};
    //double lo[] = {0.6830127, 1.1830127, 0.3169873, -0.1830127};
    //db3
    //double hi[] = {0.0498175, 0.12083221, -0.19093442, -0.650365, 1.14111692, -0.47046721};
    //double lo[] = {0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175};
    //coif1
    //double lo[] = {-0.1028594569415370, 0.4778594569415370, 1.2057189138830700, 0.5442810861169260, -0.1028594569415370, -0.0221405430584631};
    //double hi[] = {-0.0221405430584631, 0.1028594569415370, 0.5442810861169260, -1.2057189138830700, 0.4778594569415370, 0.1028594569415370};
    db4.hiFilterCoeff = hi;
    db4.loFilterCoeff = lo;
    db4.length = 2;

    double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};
    xWaveletTransform transformator;

    //WaveletReturn transformedData;
    //for(int i = 0; i < 10000; ++i)
    //{
        WaveletReturn* transformedData = transformator.doWaveletTransform(data, 24, 3, db4);
    //}

    for(vector<size_t>::iterator it = transformedData->levelLengths.begin(); it != transformedData->levelLengths.end(); it++){
        cout << *it << " ";
    }
    cout << endl;
    cout << transformedData->totalLength << ": ";
    for(int i=0; i<transformedData->totalLength; ++i){
    	cout << transformedData->data[i] << ", ";
    }
    cout << endl;
}