
/** test move semantics - run with and without BOOST_UBLAS_MOVE_SEMANTICS defined */

//          Copyright Nasos Iliopoulos, Gunter Winkler 2009.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_UBLAS_MOVE_SEMANTICS
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>

namespace ublas= boost::numeric::ublas;

using namespace ublas;
using namespace std;

int main(){
	matrix<double> m (3, 3);
	m = 1,2,3,
		3,5,6,
		7,8,9;
    std::cout << m << std::endl;	

	double* dPtr = &(m(0,0));

	for(int i =0; i<9;++i)
		cout << dPtr[i]<<endl;

	cout << m(3,0) << endl;

    return 0;
}
