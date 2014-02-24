#include <iostream>
#include <tgmath.h>
#include <functional>

using namespace std;

template<typename T, typename... A>
T test(function<T(A...)> func,A... args)
{
	return func(args...);
}

int main(int argc, const char *argv[]) {	
	cout << test(function<double(double,double)>([](double v1,double exp) -> double{ return pow(v1,exp);}),10.10,10.10)<<endl;
	return 0;
}
