#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits> 
#include <tgmath.h>
#include <functional>
#include "matrixException.hpp"
#include "dimIterator.hpp"

using namespace std;

namespace curoy{
template<typename N>
class xMatrix{
	public:
		/**
		 * CONSTRUCTOR
		 */
		xMatrix():m_data(nullptr),m_perm(memPermission::user){};
		xMatrix(N* data, initializer_list<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm) {}
		xMatrix(N* data, vector<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm){}
		xMatrix(vector<size_t> dim, enum fillMode):m_data(nullptr){m_perm=memPermission::user;resize(dim);}
		xMatrix(vector<size_t> dim, N value):m_data(nullptr){m_perm=memPermission::user;resize(dim);fill(*this,value);}
		xMatrix(xMatrix<N>&& matrix) : m_data(matrix.m_data), m_vecDim(move(matrix.m_vecDim)), m_perm(matrix.m_perm) { matrix.m_data = nullptr;}

		xMatrix(const xMatrix<N>& matrix){
			m_perm = memPermission::user;
			resize(matrix.m_vecDim);
			memcpy(m_data, matrix.m_data, matrix.size()*sizeof(N));
		}

		xMatrix(initializer_list<N> data){ 
			m_vecDim.push_back(data.size());
			m_perm = memPermission::user;
			rebase(data.size());

			int pos = 0;
			for(auto it : data){
				m_data[pos] = it; 
				++pos;
			}
		}

		xMatrix(initializer_list<xMatrix<N>> matrixList){
			// determine vector dimension
			for(auto matrix : matrixList){
				for(int depth = 0; depth < matrix.nDim();++depth ){
					if(depth == m_vecDim.size())
						m_vecDim.push_back(matrix.dim(depth));
					else if ( depth < matrix.nDim() && matrix.dim(depth) > m_vecDim[depth])
						m_vecDim[depth] = matrix.dim(depth);
				}
			}
			m_vecDim.insert(m_vecDim.begin(),matrixList.size());

			// put data
			m_perm = memPermission::user;
			rebase(size());
			
			fill(*this,0);
			size_t matrixPos = 0;
			for(auto matrix : matrixList){
				memcpy(((*this)[matrixPos]).m_data, matrix.m_data, sizeof(N)*matrix.size());
				++matrixPos;
			}
		}

		~xMatrix(){ 
			if(m_perm == memPermission::owner)
				free(m_data);
		}

		/**
		 * Matrix information
		 */
		size_t nDim() const{return m_vecDim.size();}
		size_t size() const{
			if(m_vecDim.size() == 0)
				return 0;

			size_t numElements = 1;
			auto end = this->m_vecDim.end();
			for(auto i = this->m_vecDim.begin(); i != end; ++i)
				numElements*= *i;
			return numElements;
		}
		vector<size_t> dim() const{return m_vecDim;}
		size_t dim(size_t index) const{return index < m_vecDim.size()?m_vecDim[index]:1;}

		void resize(vector<size_t> vecDim){
			m_vecDim=vecDim;
			rebase(size());
		}

		/**
		 * ACCESS
		 */
		xMatrix<N> operator[](size_t n) const { //access on none const
			if(nDim() == 0 || n >= dim(0)){
				cout << "[] overflow error"<<endl;
				return xMatrix<N>();
			}
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			
			vector<size_t> tempV(++m_vecDim.begin(),end);
			if(tempV.size()==0 && size() > 1  )
				tempV.push_back(1);
			return xMatrix<N>(m_data+n*memJump, tempV,memPermission::diver);
		}

		xMatrix<N> operator[](vector<size_t> nVec) const {
			xMatrix<N> result(this->m_data,this->m_vecDim,memPermission::diver);
			for(auto n: nVec)
				result = result[n];

			return result;
		}


		/**
		 * MOVE
		 */
		friend inline xMatrix<N>&& operator! (xMatrix<N>& matrix){
			if(matrix.m_perm == memPermission::user || matrix.m_perm == memPermission::diver){ //make sure that diver should behave like that
				N* ptr = (N*) malloc(matrix.size()*sizeof(N));
				memcpy(ptr,matrix.m_data,matrix.size()*sizeof(N));
				matrix.m_data = ptr;
				matrix.m_perm = memPermission::owner;
			}
			return move(matrix);
		}
		
		friend inline xMatrix<N>&& operator! (xMatrix<N>&& matrix){
			return move(matrix);
		}


		/**
		 * ASSIGNMENT
		 */
		xMatrix<N>& operator= (xMatrix<N>&& rhs){
			m_data = rhs.m_data;
			m_vecDim = move(rhs.m_vecDim);
			m_perm = rhs.m_perm;
			rhs.m_data = nullptr;
			return *this;
		}
		
		xMatrix<N>& operator= (const xMatrix<N>& rhs){
			if(m_data != rhs.m_data){
				if(m_perm == memPermission::diver){
					if(dimCompare(this->dim(),rhs.dim()) != 0){
						cout << "cant assign that to diver" << endl;
						return *this;
					}
				}
				else
					rebase(rhs.size());	
				m_vecDim = rhs.m_vecDim;
				memcpy(m_data,rhs.m_data, rhs.size()*sizeof(N));
			}

			return *this;
		}

		xMatrix<N>& operator= (const N value){
			if(size()==1)
				m_data[0] = value;
			else
				cout << "Assignment error" << endl;//fix it
			return *this;
		}
		
		/**
		 * MATRIX FILL
		 */
		friend xMatrix<N> fill(xMatrix<N>& matrix, const N number){	
			matrix.rebase(matrix.size());	

			for(size_t i= 0; i < matrix.size();++i ) 
				matrix.m_data[i] = number;
			return matrix;
		}

		friend xMatrix<N>&& fill(xMatrix<N>&& matrix, const N number){
			for(size_t i = 0; i < matrix.size();++i ) 
				matrix.m_data[i] = number;
			return move(matrix);
		}
		
		/**
		 * ADDITION
		 */
		friend xMatrix<N> operator+ (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] + rhs.m_data[i];
			return result;
		}

		//double rvalue
		friend inline xMatrix<N>&& operator+ (xMatrix<N>&& lhs, xMatrix<N>&& rhs){
			return move(!lhs + rhs); 
		}

		friend inline xMatrix<N>&& operator+ (xMatrix<N>& lhs, xMatrix<N>&& rhs){
			return move(!rhs + lhs); 
		}
				
		friend xMatrix<N>&& operator+ (xMatrix<N>&& lhs, xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] + rhs.m_data[i];
			return move(lhs);
		}

		/**
		 * ADDITION SKALAR
		 */

		friend xMatrix<N> operator+ (const xMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] + rhs;
			return result;
		}

		friend xMatrix<N>&& operator+ (xMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] + rhs;
			return move(lhs);
		}
		
		/**
		 * SUBSTRACTION
		 */
		friend xMatrix<N> operator- (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return result;
		}	

		//double rvalue
		friend inline xMatrix<N>&& operator- (xMatrix<N>&& lhs, xMatrix<N>&& rhs){
			return move(!lhs - rhs); 
		}

		friend inline xMatrix<N>&& operator- (xMatrix<N>& lhs, xMatrix<N>&& rhs){
			return move((!rhs) * -1 + lhs); //fix after implementing scalar
		}
	
		friend xMatrix<N>&& operator- (xMatrix<N>&& lhs, xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return move(lhs);
		}
		
		/**
		 * SUBTRACTION SKALAR
		 */

		friend xMatrix<N> operator- (const xMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] - rhs;
			return result;
		}

		friend xMatrix<N>&& operator- (xMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] -  rhs;
			return move(lhs);
		}


		/**
		 * MULTIPLICATION
		 */
		//simple R^2 matrix multiplikation (1,2)
	 	friend xMatrix<N> mult (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			if(lhs.dim(1)!=rhs.dim(0) || lhs.nDim()!=2 || rhs.nDim()!=2){
				cout<< "DIMENSIONS DONT FIT"<< endl;
				return lhs;
			}

			size_t numX = lhs.dim(0);
			size_t numK = lhs.dim(1);
			size_t numY = rhs.dim(1);

			N* temp = (N*) malloc(numX*numY*sizeof(N));
			N tempN=0;
			
			for(size_t y = 0; y < numY; ++y){
				for(size_t x = 0; x < numX; ++x){
					for(size_t i = 0; i < numK; ++i)
						tempN+=lhs.m_data[x*numK + i] * rhs.m_data[i*numY + y];

					temp[numY*x+y] = tempN;
					tempN=0;
				}
			}
			
			return xMatrix<N>(temp,vector<size_t>({numX,numY}),memPermission::owner);
		}
	
		/**
		 * ELEMENTWISE MULTIPLICATION
		 */
		friend xMatrix<N> operator* (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] * rhs.m_data[i];
			return result;
		}	

		//double rvalue
		friend inline xMatrix<N>&& operator* (xMatrix<N>&& lhs, xMatrix<N>&& rhs){
			return move(!lhs * rhs); 
		}

		friend inline xMatrix<N>&& operator* (xMatrix<N>& lhs, xMatrix<N>&& rhs){
			return move(!rhs * lhs); //fix after implementing scalar
		}
	
		friend xMatrix<N>&& operator* (xMatrix<N>&& lhs, xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] * rhs.m_data[i];
			return move(lhs);
		}	

		/**
		 * MULTIPLICATION SKALAR
		 */

		friend xMatrix<N> operator* (const xMatrix<N>& lhs,const N rhs) {
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] * rhs;
			return result;
		}

		friend xMatrix<N>&& operator* (xMatrix<N>&& lhs,const N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] * rhs;
			return move(lhs);
		}
		
	
		/**
		 * ELEMENTWISE MULTIPLICATION
		 */
		friend xMatrix<N> operator/ (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] / rhs.m_data[i];
			return result;
		}	

		//double rvalue
		friend inline xMatrix<N>&& operator/ (xMatrix<N>&& lhs, xMatrix<N>&& rhs){
			return move(!lhs / rhs); 
		}

		friend xMatrix<N>&& operator/ (xMatrix<N>&& lhs, xMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] / rhs.m_data[i];
		}
			
		/**
		 * DIVISION SKALAR
		 */
		friend xMatrix<N> operator/ (const xMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] / rhs;
			return result;
		}

		friend xMatrix<N>&& operator/ (xMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] / rhs;
			return move(lhs);
		}
		


		/**
		 * MATRIX OPERATION
		 */
		friend xMatrix<N> log(const xMatrix<N>& matrix){
			size_t numElements = matrix.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = log(matrix.m_data[i]);
			return result;
		}
		
		friend xMatrix<N>&& log(xMatrix<N>&& matrix) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = log(matrix.m_data[i]);
			return move(matrix);
		}

		
		friend xMatrix<N> log10(const xMatrix<N>& matrix){
			size_t numElements = matrix.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = log10(matrix.m_data[i]);
			return result;
		}
		
		friend xMatrix<N>&& log10(xMatrix<N>&& matrix) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = log10(matrix.m_data[i]);
			return move(matrix);
		}

		friend xMatrix<N> pow(const xMatrix<N>& matrix, N exponent){
			size_t numElements = matrix.size();
			xMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = pow(matrix.m_data[i],exponent);
			return result;
		}

		friend xMatrix<N>&& pow(xMatrix<N>&& matrix, N exponent) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = pow(matrix.m_data[i],exponent);
			return move(matrix);
		}

		friend N sum(const xMatrix<N>& matrix){
			N result=0;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i)
				result+= *(matrix.m_data+i);
			return result;
		}

		friend N prod(const xMatrix<N>& matrix){
			N result=0;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i)
				result*= *(matrix.m_data+i);
			return result;
		}

		friend bool eq(const xMatrix<N>& lhs, const xMatrix<N>& rhs){
			if(dimCompare(lhs.dim(),rhs.dim())!=0)
				return false;

			size_t end = lhs.size();
			for(size_t i=0; i<end;++i){
				if(lhs.m_data[i] != rhs.m_data[i])
					return false;
			}
			return true;
		}

		// TODO IMPLEMENT
		friend xMatrix<N> pinv(const xMatrix<N>& matrix){
			cout << "pinv() is unimplemented"<<endl;
			return matrix;
		}

		//simple Transpose only for 2D Matrix tested
		friend xMatrix<N> T(const xMatrix<N>& matrix){
			vector<size_t> transVec(matrix.m_vecDim);
			if(transVec.size() < 2)
				transVec.push_back(1);
			swap(transVec[0],transVec[1]);
			
			size_t numX = matrix.dim(0);
			size_t numY = matrix.dim(1);
			N* temp = (N*) malloc(numX*numY*sizeof(N));
			
			for(size_t x = 0; x < numX; x++){
				for(size_t y = 0; y < numY; y++){
					temp[numX*y+x]=matrix.m_data[numY*x+y];
				}
			}
			return xMatrix<N>(temp,transVec,memPermission::owner);
		}

		/**
		 * CAST 
		 */
		//operator N () const{ return *m_data;}//needed ?
		
		friend bool operator== (const xMatrix<N>& matrix,const N value){
			return *(matrix.m_data) ==value;
		}

		/**
		 * OUTPUT
		 */
		friend ostream& operator<< (ostream& os, const xMatrix<N>& matrix){
			os << "{";
			auto end = matrix.m_vecDim.end();
			for(auto it = matrix.m_vecDim.begin();it!=end;){
				os << *it;

				if(++it!=end)
					os << ", ";
			}
			os << "}  [";

			for(size_t i = 0; i<matrix.size();){
				os << matrix.m_data[i];
				if(++i<matrix.size())
					os << ", ";
			}
			os <<"]";

			return os;
		}

		N* m_data;//-> to private after tests
	private:
		enum memPermission m_perm;
		vector<size_t> m_vecDim;

		void rebase(size_t numElements){
			if(m_perm == memPermission::user){
				m_data = (N*) malloc(numElements*sizeof(N));
				m_perm = memPermission::owner;
			}
			else if(this->m_perm == memPermission::owner && numElements != this->size()){
				m_data = (N*) realloc(this->m_data,numElements*sizeof(N));
			}

			if(this->m_data == NULL)
				cout << "allocation error";	
		}
};
}
