#include "SymetricPadding.hpp"
#include <iostream>

namespace curoy{
    double SymetricPadding::get(int i){
        if(i < 0 && -i - 1< length)
        {
            return data[-i - 1];
        }
        else if(i >= 0 && i < length)
        {
            return data[i];
        }
        else if(i >= length && i < 2 * length)
        {
            return data[length - (i - length + 1)];
        }
        else{
            std::cout << "i: " << i << " length: " << length << std::endl;
            throw "invalid";
        }
    }
}
