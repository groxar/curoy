#pragma once
#include <stddef.h>
#include <string>
#include "IPadding.hpp"

namespace curoy{
class ZeroPadding: public IPadding
{
    public:
        ZeroPadding(){}
        ZeroPadding(const double* data, size_t length): IPadding(data, length) {}
        virtual double get(int);
};
}
