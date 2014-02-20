#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "../lib/xMatrix.hpp"
using namespace curoy;

TEST_CASE( "[xMatrix]", "cpu x86 matrix unit test" ) {
	int data[2][3] = {{1,2,3},{4,5,6}};
	int data1[4][5] = {{1,2,3,4,5},{4,5,6,7,8},{7,8,9,10,11},{10,11,12,13,14}};
	int data2[2][2][3] = 	{{{1,2,3},{4,5,6}}, 
							{{7,8,9},{10,11,12}}};
	int data3[2][4] = {{1,2,3,4},{4,5,6,7}};

	xMatrix<int> matrix((int*)data,{2,3});
	xMatrix<int> matrix1((int*)data1,{4,5});
	xMatrix<int> matrix2((int*)data2,{2,2,3});
	xMatrix<int> matrix3((int*)data3,{2,4});
	xMatrix<int> matrixT;
	void* pointerT=nullptr;


	SECTION("direct constructor"){
		xMatrix<int> matrix4({{{1,2,3},{4,5,6},{1,2,3}},{{7,8,9},{4,5,6,7,8}}});
		REQUIRE(dimCompare(matrix4.dim(), vector<size_t>{2,3,5}) == 0);
		xMatrix<int> matrix5({{1,2,3},{4,5,6}});
		REQUIRE( eq(matrix5, matrix));
	}
	SECTION("print"){
		cout << matrix << endl;
		cout << matrix2 << endl;
	}
	SECTION("number of dimensions"){
		REQUIRE( matrix.nDim()  == 2 );
		REQUIRE( matrix1.nDim() == 2 );
		REQUIRE( matrix2.nDim() == 3 );
		REQUIRE( matrix3.nDim() == 2 );
		REQUIRE( matrix2[0].nDim() == 2);
		REQUIRE( matrix2[0][1].nDim() == 1);
	}

	SECTION("dimensions size"){
		REQUIRE( matrix.dim(0)  == 2);
		REQUIRE( matrix.dim(1)  == 3);
		REQUIRE( matrix1.dim(0) == 4);
		REQUIRE( matrix1.dim(1) == 5);
		REQUIRE( matrix2.dim(0) == 2);
		REQUIRE( matrix2.dim(1) == 2);
		REQUIRE( matrix2.dim(2) == 3);
		REQUIRE( dimCompare(matrix.dim(),matrix.dim()) == 0);
		REQUIRE( dimCompare(matrix.dim(),matrix1.dim()) == 2);
		REQUIRE( dimCompare(matrix.dim(),matrix2.dim()) == 2);
		REQUIRE( dimCompare(matrix.dim(),matrix3.dim()) == 1);
		REQUIRE( dimCompare(vector<size_t>({}),vector<size_t>({}))==0);
		REQUIRE( dimCompare(vector<size_t>({1,1,1,1}),vector<size_t>({}))==0);
		REQUIRE( dimCompare(vector<size_t>({1,1,2,1}),vector<size_t>({}))==1);
		REQUIRE( dimCompare(vector<size_t>({1,1,2,1}),vector<size_t>({2}))==0);
		REQUIRE( dimCompare(vector<size_t>({1,1,1,1}),vector<size_t>({2}))==1);
	}

	SECTION("matrix compare"){
		matrixT = xMatrix<int>(*data,{6});
		REQUIRE(eq(matrix,matrixT) == false);
		matrixT = xMatrix<int>(*data,{2,3});
		REQUIRE(eq(matrix,matrixT) == true);
		matrixT = xMatrix<int>(*data,{3,2});
		REQUIRE(eq(matrix,matrixT) == false);
		matrixT = xMatrix<int>((int*)data2[1],{2,3});
		REQUIRE(eq(matrix,matrixT) == false);
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

	SECTION("assignment"){
		matrixT = matrix;
		REQUIRE( eq(matrixT, matrix));
		matrixT[0][2] = 1234;
		matrixT[1][0] = 12;
		cout << matrixT << endl;
		REQUIRE( matrixT[0][2] == 1234);
		REQUIRE( matrixT[1][0] == 12);
		//ADD diver assginment
	}

	SECTION("fill"){
		matrixT = matrix;
		fill(matrixT,0);
		REQUIRE( eq(matrixT, xMatrix<int>({{0,0,0},{0,0,0}})));
	}

	SECTION("operator: ! (move)") {
		pointerT = (void*)matrix.m_data;	
		matrixT = !matrix;
		REQUIRE( (!matrix).m_data != pointerT); //matrix gets the ownership with !
		REQUIRE( eq(matrixT, xMatrix<int>({{1,2,3},{4,5,6}})));

		pointerT = (void*)matrix.m_data;	
		REQUIRE( (!matrix).m_data == pointerT); //matrix has its ownership to the data, therefore nothing should happen
		REQUIRE( (!!!!matrix).m_data == pointerT);

	}

	SECTION("operator: +"){
		matrixT = matrix + matrix;
		pointerT = (void*)matrixT.m_data;
		REQUIRE( matrix.m_data != pointerT); 				//data has to be copied because matrix isn't a data memory owner
		REQUIRE( matrix[0][0] == 1);						//unchanged check
		REQUIRE( eq(matrixT, xMatrix<int>({{2,4,6},{8,10,12}})));
		REQUIRE( pointerT != (matrixT + matrixT).m_data);  	// result is within a temporal variable
		REQUIRE( pointerT == (!matrixT + matrixT).m_data); 	// result is calculated in place
		REQUIRE( pointerT == (!matrixT + !matrixT).m_data); 	// result is calculated in place within the left element
		REQUIRE( matrixT[0][0] == 8);	
		REQUIRE( pointerT != (matrixT+8).m_data);	
		REQUIRE( pointerT == (!matrixT+8).m_data);	
		REQUIRE( matrixT[0][0] == 16);	
	}
	
	SECTION("operator: -"){
		matrixT = matrix - matrix;
		pointerT = (void*)matrixT.m_data;
		REQUIRE( matrix.m_data != pointerT); 				//data has to be copied because matrix isn't a data memory owner
		REQUIRE( matrix[0][0] == 1);						//unchanged check
		REQUIRE( eq(matrixT, xMatrix<int>({{0,0,0},{0,0,0}})));
		REQUIRE( pointerT != (matrixT - matrixT).m_data);  	// result is within a temporal variable
		REQUIRE( pointerT == (!matrixT - matrixT).m_data); 	// result is calculated in place
		REQUIRE( pointerT == (!matrixT - !matrixT).m_data); 	// result is calculated in place within the left element
		REQUIRE( matrixT[0][0] == 0);	
	}

	SECTION("operator *") {
		matrixT = mult(matrix, T(matrix));
		REQUIRE( eq(matrixT, xMatrix<int>({{14,32},{32,77}})));
		matrixT = mult(T(matrix), matrix3);
		REQUIRE( eq(matrixT, xMatrix<int>({{17,22,27,32},{22,29,36,43},{27,36,45,54}})));
	}

	SECTION("matrix operation"){
		matrixT = pow(matrix,2);
		REQUIRE(eq(matrixT,xMatrix<int>({{1,4,9},{16,25,36}})));
		pow(!matrix,2);
		REQUIRE(eq(matrix,xMatrix<int>({{1,4,9},{16,25,36}})));
	}
	
}
