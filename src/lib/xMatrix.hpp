#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits> 
#include "matrixException.hpp"

using namespace std;

template<typename N>
class xMatrix{
	public:
		/**
		 * CONSTRUKTOR
		 */
		xMatrix():m_data(nullptr),m_perm(memPermission::user){};
		xMatrix(N* data, initializer_list<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm) {}
		xMatrix(N* data, vector<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm){}
	
		xMatrix(const xMatrix<N>& matrix){
			rebase(matrix.size());
			memcpy(m_data, matrix.m_data, matrix.size()*sizeof(N));
			m_vecDim = matrix.m_vecDim;
			m_perm = memPermission::owner;
		}

		xMatrix(xMatrix<N>&& matrix) : m_data(matrix.m_data), m_vecDim(move(matrix.m_vecDim)), m_perm(matrix.m_perm) { matrix.m_data = nullptr;}

		~xMatrix(){ 
			if(m_perm == memPermission::owner)
				free(m_data);
		}

		/**
		 * MISC 
		 */
		size_t nDim() const{return m_vecDim.size();}
		size_t size() const{
			size_t numElements = 1;
			auto end = this->m_vecDim.end();
			for(auto i = this->m_vecDim.begin(); i != end; ++i)
				numElements*= *i;
			return numElements;
		}
		vector<size_t> dim() const{return m_vecDim;}
		size_t dim(size_t index) const{return m_vecDim[index];}
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

		/**
		 * ACCESS
		 */
		xMatrix<N> operator[](size_t n) const { //access on none const
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			return xMatrix<N>(m_data+n*memJump, vector<size_t>(++m_vecDim.begin(),end),memPermission::user);
		}

		xMatrix<N> operator[](vector<size_t> nVec) const {
			if(nVec.size()>m_vecDim.size())
				throw nullptr;
			xMatrix<N> result(this->m_data,this->m_vecDim,memPermission::user);
			for(auto n: nVec)
			{
				result = result[n];
			}

			return result;
		}

		/**
		 * MOVE
		 */
		friend inline xMatrix<N>&& operator! (xMatrix<N>& matrix){
			if(matrix.m_perm == memPermission::user){
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
				rebase(rhs.size());	
				memcpy(m_data,rhs.m_data, rhs.size()*sizeof(N));
				m_vecDim = rhs.m_vecDim;
			}

			return *this;
		}
		
		/**
		 * MATRIX FILL
		 */
		friend xMatrix<N> fill(xMatrix<N>& matrix, N number){	
			matrix.rebase(matrix.size());	

			for(size_t i= 0; i < matrix.size();++i ) 
				matrix.m_data[i] = number;
			return matrix;
		}

		friend xMatrix<N>&& fill(xMatrix<N>&& matrix, N number){
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
			return move((!rhs)*-1 + lhs); //fix after implementing scalar
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
	 	friend xMatrix<N> operator* (const xMatrix<N>& lhs,const xMatrix<N>& rhs){
			if(lhs.m_vecDim[1]!=rhs.m_vecDim[0] || lhs.m_vecDim.size()!=2 || rhs.m_vecDim.size()!=2){
				throw "DIMENSIONS DONT FIT";
				return lhs;
			}

			size_t numX = lhs.m_vecDim[0];
			size_t numY = rhs.m_vecDim[1];
			N* temp = (N*) malloc(numX*numY*sizeof(N));
			N tempN=0;
			
			for(size_t y = 0; y < numY; ++y){
				for(size_t x = 0; x < numX; ++x){
					for(size_t i = 0; i < lhs.m_vecDim[1]; ++i)
						tempN+=(N)lhs[x][i] * (N)rhs[i][y];

					temp[numY*x+y] = tempN;
					tempN=0;
				}
			}
			
			return xMatrix<N>(temp,vector<size_t>({numX,numY}),memPermission::owner);
		}
		
		/**
		 * MATRIX OPERATION
		 */
		friend N sum(const xMatrix<N>& matrix){
			N result=0;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i)
				result+= *(matrix.m_data+i);
			return result;
		}

		//simple Transpose only for 2D Matrix tested
		friend xMatrix<N> T(const xMatrix<N>& matrix){
			vector<size_t> transVec(matrix.m_vecDim);
			swap(transVec[0],transVec[1]);
			
			size_t numX = matrix.m_vecDim[0];
			size_t numY = matrix.m_vecDim[1];
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
		operator N () const{ return *m_data;}//needed ?

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
		vector<size_t> m_vecDim;
	private:
			
		enum memPermission m_perm;

};
