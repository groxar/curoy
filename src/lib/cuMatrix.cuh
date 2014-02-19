#pragma once

#define B_WIDTH 16
#define B_SIZE B_WIDTH*B_WIDTH 
#define CEIL_DIV(a,b) ((((a)-1)/b)+1)

template<typename N> 
extern void addDev(N* lhs, N* rhs, N* result, size_t numElements);
