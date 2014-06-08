#include "ConstantPadding.hpp"
#include <iostream>

namespace curoy{
    double ConstantPadding::get(int i){
        if(i < 0 && -i - 1< length)
        {
            return data[0];
        }
        else if(i >= 0 && i < length)
        {
            return data[i];
        }
        else if(i >= length && i < 2 * length)
        {
            return data[length - 1];
        }
        else{
            std::cout << "Error: filter longer than signal" << "i: " << i << " length: " << length << " which is not allowed and of course doesnt give you better results" << std::endl;
            throw "invalid";
        }
    }
}
