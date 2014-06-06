#pragma once
#include <stddef.h>
#include <string>
#include "IPadding.hpp"

namespace curoy{
class ConstantPadding: public IPadding
{
    public:
        ConstantPadding(){}
        ConstantPadding(const double* data, size_t length): IPadding(data, length) {}
        virtual double get(int);
};
}
