#include <tgmath.h>
#include <iostream>

template<typename FUNC,typename... N>
int singleOp(FUNC f, N... n){
	return f(12,n...);
}


int main(int argc, const char *argv[]) {
	std::cout << singleOp(&pow,2)<<std::endl;	
	return 0;
}

