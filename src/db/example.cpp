#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <typeinfo>

#include "../lib/xMatrix.hpp"
#include "IxMatrixIOAdapter.hpp"
#include "xMatrixFileAdapter.hpp"
#include "xMatrixRedisBinaryAdapter.hpp"

using namespace curoy;


int main(int argc, char **argv)
{
    IxMatrixIOAdapter *ioAdapter = new xMatrixFileAdapter();
    IxMatrixIOAdapter *redisAdapter = new xMatrixRedisBinaryAdapter("/tmp/redis.sock");
    xMatrix<double> devices = ioAdapter->Load("data/devices.txt");
    redisAdapter->Save("devices", devices);
    //redisAdapter->Save("xdata", matrixX);
    //redisAdapter->Save("ydata", matrixY);
    for(int i = 0; i < devices.dim(0); ++i)
    {
        cout << devices.m_data[i] << endl;
    }
    delete ioAdapter;
}
