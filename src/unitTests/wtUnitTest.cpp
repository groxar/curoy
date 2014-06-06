#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../wt/WaveletTransformator.hpp"
#include "../wt/Filter.hpp"
#include "../wt/WaveletReturn.hpp"
using namespace curoy;
using namespace std;

TEST_CASE( "[wt]", "Wavelet Transformation Unit Test"){

    WaveletTransformator transformator;
    //24 Numbers
    const double data[] = {345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6,345,564,3,234,56,576,324,6};
    double *reconstructedData = 0;
    WaveletReturn *transformedData = 0;

    SECTION("Compare reconstructed Signal with Original - haar"){

        transformedData = transformator.waveletDecomposition(data, 21, 2, "haar");
        reconstructedData = transformator.waveletReconstruction(transformedData->data, transformedData->levelLengths, "haar");

        for(int i = 0; i < 21; ++i)
        {
            REQUIRE(abs(reconstructedData[i] - data[i]) <= 0.000001);
        }

        delete reconstructedData;
        delete transformedData;
    }

    SECTION("Compare reconstructed Signal with Original - db10"){

        transformedData = transformator.waveletDecomposition(data, 23, 2, "db10");
        reconstructedData = transformator.waveletReconstruction(transformedData->data, transformedData->levelLengths, "db10");

        for(int i = 0; i < 23; ++i)
        {
            REQUIRE(abs(reconstructedData[i] - data[i]) <= 0.000001);
        }

        delete reconstructedData;
        delete transformedData;
    }

    SECTION("Compare reconstructed Signal with Original - db5"){

        transformedData = transformator.waveletDecomposition(data, 24, 3, "db5");
        reconstructedData = transformator.waveletReconstruction(transformedData->data, transformedData->levelLengths, "db5");

        for(int i = 0; i < 24; ++i)
        {
            REQUIRE(abs(reconstructedData[i] - data[i]) <= 0.000001);
        }

        delete reconstructedData;
        delete transformedData;
    }

    SECTION("Compare reconstructed Signal with Original - coif4"){

        transformedData = transformator.waveletDecomposition(data, 23, 3, "coif4");
        reconstructedData = transformator.waveletReconstruction(transformedData->data, transformedData->levelLengths, "coif4");

        for(int i = 0; i < 23; ++i)
        {
            REQUIRE(abs(reconstructedData[i] - data[i]) <= 0.000001);
        }

        delete reconstructedData;
        delete transformedData;
    }


    SECTION("Compare with Matlab - db7"){

        transformedData = transformator.waveletDecomposition(data, 22, 2, "db7");
        double matlabData[] = {615.0530, 429.1317, 715.9637, 485.7137, 575.5104, 719.5069, 451.2945, 664.2594, 538.3533, 590.3742, 445.4702, 649.0933, 327.0384, 940.3364, 565.4872, -19.6341, 314.2888, 186.7386, -313.5022, -88.0153, -198.2885, -104.7856, -149.4850, 141.9304, -111.0578, 79.7046, 148.4747, 166.3426, 132.2826, 216.2515, 104.9736, -334.9029, -170.7437, -240.9260, 306.3625, -371.7277, -143.1034, -256.9904, 311.4836, -372.4233, -143.1034, -160.9551, 393.0713, 149.0291, 0.1882, -81.6325, 385.3028};
        REQUIRE(transformedData->totalLength == 47);
        for(int i = 0; i < 47; ++i)
        {
            REQUIRE(abs(matlabData[i] - transformedData->data[i]) <= 0.001);
        }

        delete transformedData;
    }

    SECTION("Compare with Matlab - haar"){

        transformedData = transformator.waveletDecomposition(data, 24, 3, "haar");
        double matlabData[] = {745.2905, 745.2905, 745.2905, 65.0538, 65.0538, 65.0538, 336.0000, 151.0000, 336.0000, 151.0000, 336.0000, 151.0000, -154.8564, -163.3417, -367.6955, 224.8600, -154.8564, -163.3417, -367.6955, 224.8600, -154.8564, -163.3417, -367.6955, 224.8600};
        REQUIRE(transformedData->totalLength == 24);
        for(int i = 0; i < 24; ++i)
        {
            REQUIRE(abs(matlabData[i] - transformedData->data[i]) <= 0.001);
        }

        delete transformedData;
    }
}