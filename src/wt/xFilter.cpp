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
        else if (waveletType == "sym2"){
            loFilterCoeff = sym2[0];
            hiFilterCoeff = sym2[1];
            length = 4;
        }
        else if (waveletType == "sym3"){
            loFilterCoeff = sym3[0];
            hiFilterCoeff = sym3[1];
            length = 6;
        }
        else if (waveletType == "sym4"){
            loFilterCoeff = sym4[0];
            hiFilterCoeff = sym4[1];
            length = 8;
        }
        else if (waveletType == "sym5"){
            loFilterCoeff = sym5[0];
            hiFilterCoeff = sym5[1];
            length = 10;
        }
        else if (waveletType == "sym6"){
            loFilterCoeff = sym6[0];
            hiFilterCoeff = sym6[1];
            length = 12;
        }
        else if (waveletType == "sym7"){
            loFilterCoeff = sym7[0];
            hiFilterCoeff = sym7[1];
            length = 14;
        }
        else if (waveletType == "sym8"){
            loFilterCoeff = sym8[0];
            hiFilterCoeff = sym8[1];
            length = 16;
        }
        else if (waveletType == "sym9"){
            loFilterCoeff = sym9[0];
            hiFilterCoeff = sym9[1];
            length = 18;
        }
        else if (waveletType == "sym10"){
            loFilterCoeff = sym10[0];
            hiFilterCoeff = sym10[1];
            length = 20;
        }
        else if (waveletType == "sym11"){
            loFilterCoeff = sym11[0];
            hiFilterCoeff = sym11[1];
            length = 22;
        }
        else if (waveletType == "sym12"){
            loFilterCoeff = sym12[0];
            hiFilterCoeff = sym12[1];
            length = 24;
        }
        else if (waveletType == "sym13"){
            loFilterCoeff = sym13[0];
            hiFilterCoeff = sym13[1];
            length = 26;
        }
        else if (waveletType == "sym14"){
            loFilterCoeff = sym14[0];
            hiFilterCoeff = sym14[1];
            length = 28;
        }
        else if (waveletType == "sym15"){
            loFilterCoeff = sym15[0];
            hiFilterCoeff = sym15[1];
            length = 30;
        }
        else if (waveletType == "sym16"){
            loFilterCoeff = sym16[0];
            hiFilterCoeff = sym16[1];
            length = 32;
        }
        else if (waveletType == "sym17"){
            loFilterCoeff = sym17[0];
            hiFilterCoeff = sym17[1];
            length = 34;
        }
        else if (waveletType == "sym18"){
            loFilterCoeff = sym18[0];
            hiFilterCoeff = sym18[1];
            length = 36;
        }
        else if (waveletType == "sym19"){
            loFilterCoeff = sym19[0];
            hiFilterCoeff = sym19[1];
            length = 38;
        }
        else if (waveletType == "sym20"){
            loFilterCoeff = sym20[0];
            hiFilterCoeff = sym20[1];
            length = 40;
        }
        else if (waveletType == "coif1"){
            loFilterCoeff = coif1[0];
            hiFilterCoeff = coif1[1];
            length = 6;
        }
        else if (waveletType == "coif2"){
            loFilterCoeff = coif2[0];
            hiFilterCoeff = coif2[1];
            length = 12;
        }
        else if (waveletType == "coif3"){
            loFilterCoeff = coif3[0];
            hiFilterCoeff = coif3[1];
            length = 18;
        }
        else if (waveletType == "coif4"){
            loFilterCoeff = coif4[0];
            hiFilterCoeff = coif4[1];
            length = 24;
        }
        else if (waveletType == "coif5"){
            loFilterCoeff = coif5[0];
            hiFilterCoeff = coif5[1];
            length = 30;
        }
        else if (waveletType == "bior1.1"){
            loFilterCoeff = bior1_1[0];
            hiFilterCoeff = bior1_1[1];
            length = 2;
        }
        else if (waveletType == "bior1.3"){
            loFilterCoeff = bior1_3[0];
            hiFilterCoeff = bior1_3[1];
            length = 6;
        }
        else if (waveletType == "bior1.5"){
            loFilterCoeff = bior1_5[0];
            hiFilterCoeff = bior1_5[1];
            length = 10;
        }
        else if (waveletType == "bior2.2"){
            loFilterCoeff = bior2_2[0];
            hiFilterCoeff = bior2_2[1];
            length = 6;
        }
        else if (waveletType == "bior2.4"){
            loFilterCoeff = bior2_4[0];
            hiFilterCoeff = bior2_4[1];
            length = 10;
        }
        else if (waveletType == "bior2.6"){
            loFilterCoeff = bior2_6[0];
            hiFilterCoeff = bior2_6[1];
            length = 14;
        }
        else if (waveletType == "bior2.8"){
            loFilterCoeff = bior2_8[0];
            hiFilterCoeff = bior2_8[1];
            length = 18;
        }
        else if (waveletType == "bior3.1"){
            loFilterCoeff = bior3_1[0];
            hiFilterCoeff = bior3_1[1];
            length = 4;
        }
        else if (waveletType == "bior3.3"){
            loFilterCoeff = bior3_3[0];
            hiFilterCoeff = bior3_3[1];
            length = 8;
        }
        else if (waveletType == "bior3.5"){
            loFilterCoeff = bior3_5[0];
            hiFilterCoeff = bior3_5[1];
            length = 12;
        }
        else if (waveletType == "bior3.7"){
            loFilterCoeff = bior3_7[0];
            hiFilterCoeff = bior3_7[1];
            length = 16;
        }
        else if (waveletType == "bior3.9"){
            loFilterCoeff = bior3_9[0];
            hiFilterCoeff = bior3_9[1];
            length = 20;
        }
        else if (waveletType == "bior4.4"){
            loFilterCoeff = bior4_4[0];
            hiFilterCoeff = bior4_4[1];
            length = 10;
        }
        else if (waveletType == "bior5.5"){
            loFilterCoeff = bior5_5[0];
            hiFilterCoeff = bior5_5[1];
            length = 12;
        }
        else if (waveletType == "bior6.8"){
            loFilterCoeff = bior6_8[0];
            hiFilterCoeff = bior6_8[1];
            length = 18;
        }
        else if (waveletType == "dmey"){
            loFilterCoeff = dmey[0];
            hiFilterCoeff = dmey[1];
            length = 62;
        }
        else{
            throw 0; //wrong argument TODO Exception
        }

    }
}
