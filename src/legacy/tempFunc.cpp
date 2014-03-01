#include <tgmath.h>
#include <iostream>

using namespace std;

template<typename FUNC,typename... N>
int singleOp(FUNC f, N... n){
	return f(n...);
}

int foo(int a){
	return a+2;
}


int main(int argc, const char *argv[]) {
	std::cout << singleOp(&::pow,2,2)<<std::endl;	
	auto fa = [](int a)->int {return !a;};
	std::cout << singleOp(fa,0)<<std::endl;	
	auto pl = [](int a, int b)->int {return a + b;};
	std::cout << singleOp(pl,23,423)<<std::endl;	
	std::cout << singleOp(&foo,423)<<std::endl;	
	return 0;
}

