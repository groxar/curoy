#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../wt/WaveletTransformator.hpp"
#include "../wt/Filter.hpp"
#include "../wt/WaveletReturn.hpp"
using namespace curoy;

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

}