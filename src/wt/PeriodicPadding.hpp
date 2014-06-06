#pragma once
#include <stddef.h>
#include <string>
#include "IPadding.hpp"

namespace curoy{
class PeriodicPadding: public IPadding
{
    public:
        PeriodicPadding(){}
        PeriodicPadding(const double* data, size_t length): IPadding(data, length) {}
        virtual double get(int);
};
}
