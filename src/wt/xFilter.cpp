#include "xFilter.hpp"
#include "waveletCoefficients.hpp"
#include <string>

using namespace std;

namespace curoy{

    xFilter::xFilter(){

    }
    xFilter::xFilter(string waveletType)
    {

        if (waveletType == "haar"){
            loFilterCoeff = db1[0];
            hiFilterCoeff = db1[1];
            length = 2;
        }
        else if (waveletType == "db1"){
            loFilterCoeff = db1[0];
            hiFilterCoeff = db1[1];
            length = 2;
        }
        else if (waveletType == "db2"){
            loFilterCoeff = db2[0];
            hiFilterCoeff = db2[1];
            length = 4;
        }
        else if (waveletType == "db3"){
            loFilterCoeff = db3[0];
            hiFilterCoeff = db3[1];
            length = 6;
        }
        else if (waveletType == "db4"){
            loFilterCoeff = db4[0];
            hiFilterCoeff = db4[1];
            length = 8;
        }
        else if (waveletType == "db5"){
            loFilterCoeff = db5[0];
            hiFilterCoeff = db5[1];
            length = 10;
        }
        else if (waveletType == "db6"){
            loFilterCoeff = db6[0];
            hiFilterCoeff = db6[1];
            length = 12;
        }
        else if (waveletType == "db7"){
            loFilterCoeff = db7[0];
            hiFilterCoeff = db7[1];
            length = 14;
        }
        else if (waveletType == "db8"){
            loFilterCoeff = db8[0];
            hiFilterCoeff = db8[1];
            length = 16;
        }
        else if (waveletType == "db9"){
            loFilterCoeff = db9[0];
            hiFilterCoeff = db9[1];
            length = 18;
        }
        else if (waveletType == "db10"){
            loFilterCoeff = db10[0];
            hiFilterCoeff = db10[1];
            length = 20;
        }
        else if (waveletType == "db11"){
            loFilterCoeff = db11[0];
            hiFilterCoeff = db11[1];
            length = 22;
        }
        else if (waveletType == "db12"){
            loFilterCoeff = db12[0];
            hiFilterCoeff = db12[1];
            length = 24;
        }
        else if (waveletType == "db13"){
            loFilterCoeff = db13[0];
            hiFilterCoeff = db13[1];
            length = 26;
        }
        else if (waveletType == "db14"){
            loFilterCoeff = db14[0];
            hiFilterCoeff = db14[1];
            length = 28;
        }
        else if (waveletType == "db15"){
            loFilterCoeff = db15[0];
            hiFilterCoeff = db15[1];
            length = 30;
        }
        else if (waveletType == "db16"){
            loFilterCoeff = db16[0];
            hiFilterCoeff = db16[1];
            length = 32;
        }
        else if (waveletType == "db17"){
            loFilterCoeff = db17[0];
            hiFilterCoeff = db17[1];
            length = 34;
        }
        else if (waveletType == "db18"){
            loFilterCoeff = db18[0];
            hiFilterCoeff = db18[1];
            length = 36;
        }
        else if (waveletType == "db19"){
            loFilterCoeff = db19[0];
            hiFilterCoeff = db19[1];
            length = 38;
        }
        else if (waveletType == "db20"){
            loFilterCoeff = db20[0];
            hiFilterCoeff = db20[1];
            length = 40;
        }
        else{
            throw 0; //wrong argument TODO Exception
        }

    }
}
