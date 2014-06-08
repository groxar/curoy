#include "ZeroPadding.hpp"
#include <iostream>

namespace curoy{
    double ZeroPadding::get(int i){
        if(i >= 0 && i < length)
        {
            return data[i];
        }
        else
        {
            return 0;
        }
    }
}
