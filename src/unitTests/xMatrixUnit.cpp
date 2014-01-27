#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../lib/xMatrix.hpp"

TEST_CASE( "cpu x86 matrix unit test", "[xMatrix]" ) {
	int data[2][3] = {{1,2,3},{4,5,6}};
	int data1[4][5] = {{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{10,11,12,13,14}};
	int data2[2][2][3] = 	{{{1,2,3},{4,5,6}}, 
							{{7,8,9},{10,11,12}}};
	int data3[2][4] = {{1,2,3,4},{4,5,6,7}};
	xMatrix<int> matrix((int*)data,{2,3},memPermission::read);
	xMatrix<int> matrix1((int*)data1,{4,5});
	xMatrix<int> matrix2((int*)data2,{2,2,3},memPermission::read);
	xMatrix<int> matrix3((int*)data3,{2,4},memPermission::read);
	xMatrix<int> matrixT;
	void* pointerT=nullptr;

	SECTION("number of dimensions"){
		REQUIRE( matrix.nDim()  == 2 );
		REQUIRE( matrix1.nDim() == 2 );
		REQUIRE( matrix2.nDim() == 3 );
		REQUIRE( matrix3.nDim() == 2 );
		REQUIRE( matrix2[0].nDim() == 2);
		REQUIRE( matrix2[0][1].nDim() == 1);
	}

	SECTION("dimensions size"){
		REQUIRE( matrix.dim()[0]  == 2);
		REQUIRE( matrix.dim()[1]  == 3);
		REQUIRE( matrix1.dim()[0] == 4);
		REQUIRE( matrix1.dim()[1] == 5);
		REQUIRE( matrix2.dim()[0] == 2);
		REQUIRE( matrix2.dim()[1] == 2);
		REQUIRE( matrix2.dim()[2] == 3);
	}

	SECTION("operator: []"){
		REQUIRE( matrix[0][0] == 1);
		REQUIRE( matrix[0][1] == 2);
		REQUIRE( matrix[0][2] == 3);
		REQUIRE( matrix[1][0] == 4);
		REQUIRE( matrix[1][1] == 5);
		REQUIRE( matrix[1][2] == 6);
		REQUIRE( matrix2[0][0][2] == 3);
		REQUIRE( matrix2[1][0][2] == 9);
		REQUIRE( matrix2[1][1][2] == 12);
		REQUIRE( matrix[vector<size_t>({1,1})] == 5);
	}

	SECTION("operator: +"){
		matrixT = matrix + matrix;
		pointerT = (void*)matrixT.m_data;
		REQUIRE( matrix.m_data != pointerT); 				//data has to be copied because matrix isn't a data memory owner
		REQUIRE( matrix[0][0] == 1);						//unchanged check
		REQUIRE( matrixT[0][0] == 2);	
		REQUIRE( matrixT[0][1] == 4);	
		REQUIRE( matrixT[0][2] == 6);	
		REQUIRE( matrixT[1][0] == 8);	
		REQUIRE( matrixT[1][1] == 10);	
		REQUIRE( matrixT[1][2] == 12);	
		REQUIRE( pointerT != (matrixT + matrixT).m_data);  	// result is within a temporal variable
		REQUIRE( pointerT == (!matrixT + matrixT).m_data); 	// result is calculated in place
		REQUIRE( matrixT[0][0] == 4);	

	}
	
	SECTION("operator: !"){
		
	}

/*
	DEBUG(data2[1]);
	DEBUG(data2[1][0][2]);
	cout<< endl << "Math Test"<< endl;
	cout << (matrix+matrix)[1][2] << endl;
	cout << matrix[1][1]<< endl;
	cout << matrix.m_data<< endl;
	cout << (matrix+matrix)[1][1]<< endl;
	cout << "____________________________"<<endl;
	cout << (matrix+matrix).m_data<< endl;
	cout << "!matrix2+matrix2" << endl;
	cout << matrix2<< endl;
	cout << matrix2.m_data<< endl;
	cout << (!matrix2+matrix2).m_data<< endl;
	cout << matrix2[1][1]<< endl;
	cout << matrix2<< endl;
	cout << matrix2.m_data<< endl;
	cout << (matrix2 + matrix2)[1][1]<< endl;
	cout << matrix2 << endl;
	cout << "matrix2+!matrix2" << endl;
	cout << matrix2<< endl;
	cout << matrix2.m_data<< endl;
	cout << (matrix2+!matrix2).m_data<< endl;
	cout << matrix2[1][1]<< endl;
	cout << matrix2<< endl;
	cout << matrix2.m_data<< endl;
	cout << (matrix2 + matrix2 + matrix2)[1][1]<< endl;
	cout << matrix2 << endl;

	cout << "dim(): " << endl;
	for(auto i : matrix2.dim())
		cout << i << " ";
	cout << endl;

	cout << endl << "copy test"<<endl;
	/ *matrix1=matrix2;
	DEBUG(matrix.m_data);
	DEBUG(matrix);
	DEBUG(matrix1.m_data);
	DEBUG(matrix1);
	DEBUG(sum(matrix+matrix));* /

	cout << endl << "Transpose"<<endl;
	DEBUG(matrix);
	DEBUG(matrix.m_data);
	DEBUG(T(matrix));
	DEBUG(T(matrix).m_data);
	//DEBUG(matrix2);//breaks here
	//DEBUG(T(matrix2));
	//
	cout << endl << "Multiplication"<<endl;
	DEBUG(matrix*T(matrix));
	xMatrix<int> matrix3= T(matrix1)*matrix1;
	DEBUG(matrix1.dim()[0]);
	DEBUG(matrix1.dim()[1]);
	DEBUG(matrix1);
	DEBUG(matrix3.dim()[0]);
	DEBUG(matrix3.dim()[1]);
	DEBUG(matrix3.nDim());
	DEBUG(matrix3);


	cout << endl << "Assignment"<<endl;
	matrix=matrix+matrix;
	DEBUG(!matrix);
	*/
}
