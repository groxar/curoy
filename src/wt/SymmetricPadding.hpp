#pragma once
#include <stddef.h>
#include <string>
#include "IPadding.hpp"

namespace curoy{
class SymmetricPadding: public IPadding
{
    public:
        SymmetricPadding(){}
        SymmetricPadding(const double* data, size_t length): IPadding(data, length) {}
        virtual double get(int);
};
}
