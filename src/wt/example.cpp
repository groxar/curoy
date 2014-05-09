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
    //double lo[] = {1,1};
    //double hi[] = {-1,1};

    //double lo[] = {0.70710678118654757, 0.70710678118654757};
    //double hi[] = {0.70710678118654757, -0.70710678118654757};
    //db2
    //double lo[] = {0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145};
    //double hi[] = {-0.12940952255092145, -0.22414386804185735, 0.83651630373746899, -0.48296291314469025};
    //double lo[] = {0.6830127, 1.1830127, 0.3169873, -0.1830127};
    //double hi[] = {-0.1830127, -0.3169873, 1.1830127, -0.6830127};
    //db3

    double lo[] = {0.33267055295095688, 0.80689150931333875, 0.45987750211933132, -0.13501102001039084, -0.085441273882241486, 0.035226291882100656};
    double hi[] = {0.035226291882100656, 0.085441273882241486, -0.13501102001039084, -0.45987750211933132, 0.80689150931333875, -0.33267055295095688};
    //double lo[] = {0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175};
    //double hi[] = {0.0498175, 0.12083221, -0.19093442, -0.650365, 1.14111692, -0.47046721};
    //coif1
    //double lo[] = {-0.1028594569415370, 0.4778594569415370, 1.2057189138830700, 0.5442810861169260, -0.1028594569415370, -0.0221405430584631};
    //double hi[] = {-0.0221405430584631, 0.1028594569415370, 0.5442810861169260, -1.2057189138830700, 0.4778594569415370, 0.1028594569415370};
    db4.hiFilterCoeff = hi;
    db4.loFilterCoeff = lo;
    db4.length = 6;

    //double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};
    double data[] = {1,2,3,1,2,3,4,0};
    xWaveletTransform transformator;

    //WaveletReturn transformedData;
    //for(int i = 0; i < 10000; ++i)
    //{
        WaveletReturn* transformedData = transformator.doWaveletTransform(data, 8, 3, db4);
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
    cout << endl;

    xFilter haarReconstruction;
    double haarLo[] = {0.70710678118654757, 0.70710678118654757};
    double haarHi[] = {0.70710678118654757, -0.70710678118654757};
    //double haarLo[] = {-0.12940952255092145, 0.22414386804185735, 0.83651630373746899, 0.48296291314469025};
    //double haarHi[] = {-0.48296291314469025, 0.83651630373746899, -0.22414386804185735, -0.12940952255092145};

    //double haarLo[] = {1, 1};
    //double haarHi[] = {1, -1};
    haarReconstruction.hiFilterCoeff = hi;
    haarReconstruction.loFilterCoeff = lo;
    haarReconstruction.length = 6;
    double *reconstructedData = transformator.doWaveletReconstruction(transformedData->data, transformedData->levelLengths, haarReconstruction);
    for(int i = 0; i < 8; ++i)
    {
        cout << reconstructedData[i] << ", ";
    }
    cout << endl;
}