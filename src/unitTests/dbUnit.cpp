#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <iostream>
#include "catch.hpp"
#include "../lib/xMatrix.hpp"
#include "../db/IxMatrixIOAdapter.hpp"
#include "../db/xMatrixFileAdapter.hpp"
#include "../db/xMatrixRedisBinaryAdapter.hpp"
using namespace curoy;

TEST_CASE( "[db]", "io adapter unit test"){
    SECTION("store and load data to and from redis"){
        double data[2][3] = {{1.0,2.0,3.0},{4.0,5.0,6.0}};
        xMatrix<double> matrix((double*) data, {2,3});
        IxMatrixIOAdapter *redisAdapter = new xMatrixRedisBinaryAdapter("/tmp/redis.sock");
        redisAdapter->Save("test", matrix);
        xMatrix<double> matrix2 = redisAdapter->Load("test");
        REQUIRE( matrix.nDim() == matrix2.nDim() );
        REQUIRE( eq(matrix,matrix2) == true );
    }

}
