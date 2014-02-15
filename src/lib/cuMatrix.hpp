#pragma once
#include <vector>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits> 
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrixException.hpp"
#include "xMatrix.hpp"
#include "cuMatrix.hu"

using namespace std;

namespace curoy{
template<typename N>
class cuMatrix{
	public:
		/**
		 * CONSTRUKTOR
		 */
		cuMatrix():m_data(nullptr),m_perm(memPermission::user){};
		cuMatrix(N* data, initializer_list<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm) {}
		cuMatrix(N* data, vector<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm){}

		cuMatrix(const cuMatrix<N>& matrix){ // TODO rework THAT IS ALL WRONG 
			//resize

			memcpy(m_data, matrix.m_data, matrix.size()*sizeof(N));
			m_vecDim = matrix.m_vecDim;
		}
		cuMatrix(cuMatrix<N>&& matrix) : m_data(matrix.m_data),m_vecDim(move(matrix.m_vecDim)), m_perm(matrix.m_perm) { matrix.m_data = nullptr;}

		~cuMatrix(){ 
			if(m_perm == memPermission::owner)
				cudaFree(m_data);
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

		/**
		 * Access
		 */
		cuMatrix<N> operator[](size_t n) const {// TODO rework
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			return cuMatrix<N>(m_data+n*memJump, vector<size_t>(++m_vecDim.begin(),end),memPermission::user);
		}

		cuMatrix<N> operator[](vector<size_t> nVec) const {  // TODO rework
			if(nVec.size()>m_vecDim.size())
				throw nullptr;
			cuMatrix<N> result(this->m_data,this->m_vecDim,memPermission::user);
			for(auto n: nVec)
			{
				result = result[n];
			}

			return result;
		}

		friend inline cuMatrix<N>&& operator! (cuMatrix<N>& matrix){
			return move(matrix); //handle non ownership
		}

		/**
		 * ASSIGNMENT
		 */
		cuMatrix<N>& operator= (cuMatrix<N>&& rhs){  // TODO rework
			m_data = rhs.m_data;
			m_vecDim = move(rhs.m_vecDim);
			m_perm = rhs.m_perm;
			rhs.m_data = nullptr;
			return *this;
		}
		
		cuMatrix<N>& operator= (const cuMatrix<N>& rhs){ // TODO rework
			//resize on rhs
			memcpy(m_data,rhs.m_data, rhs.size()*sizeof(N));
			m_vecDim = rhs.m_vecDim;
			return *this;
		}

		/**
		 * Data Transfer to/from CUDA device
		 */
		friend cuMatrix<N>& operator>> (const xMatrix<N>& lhs, cuMatrix<N>& rhs){
			rhs.m_vecDim= vector<size_t>(lhs.m_vecDim.begin(),lhs.m_vecDim.end());
			cudaMalloc((int**) &rhs.m_data, lhs.size()*sizeof(N));
			cudaMemcpy(rhs.m_data,lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyHostToDevice);
			return rhs;
		}
		
		friend xMatrix<N>& operator<< (xMatrix<N>& lhs, const cuMatrix<N>& rhs){
			lhs.m_vecDim = vector<size_t>(rhs.m_vecDim.begin(),rhs.m_vecDim.end());
			lhs.m_data = (N*) malloc(rhs.size()*sizeof(N));
			cudaMemcpy(lhs.m_data,rhs.m_data,rhs.size()*sizeof(N),cudaMemcpyDeviceToHost);
			return lhs;
		}
		
		
		/**
		 * ADDITION
		 */
		friend cuMatrix<N> operator+ (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){ 
			size_t numElements = lhs.size();
			cuMatrix<N> result(nullptr,lhs.m_vecDim,memPermission::owner);
			cudaMalloc((int**)&result.m_data,numElements*sizeof(N));
			addDev<N>(lhs.m_data,rhs.m_data,result.m_data,numElements);	
			return result;
		}

		friend inline cuMatrix<N>&& operator+ (cuMatrix<N>& lhs, cuMatrix<N>&& rhs){ // TODO rework
			return move(!rhs + lhs); 
		}
		
		friend cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, cuMatrix<N>& rhs){ 
			size_t numElements = lhs.size();
			addDev<N>(lhs.m_data,rhs.m_data,lhs.m_data,numElements);	
			return move(lhs);
		}
		
		/**
		 * SUBSTRACTION // TODO rework
		 */
		friend cuMatrix<N> operator- (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return result;
		}

		friend inline cuMatrix<N>&& operator- (cuMatrix<N>& lhs, cuMatrix<N>&& rhs){
			return move(!rhs - lhs); 
		}
		
		friend cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, cuMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return move(lhs);
		}

		/**
		 * MULTIPLICATION // TODO rework
		 */
		//simple R^2 matrix multiplikation (1,2)
	 	friend cuMatrix<N> operator* (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			if(lhs.m_vecDim[1]!=rhs.m_vecDim[0] || lhs.m_vecDim.size()!=2 || rhs.m_vecDim.size()!=2)
				throw "X DIMENSIONS DONT FIT";

			size_t numX = lhs.m_vecDim[0];
			size_t numY = rhs.m_vecDim[1];
			N* temp = (N*) malloc(numX*numY*sizeof(N));
			N tempN=0;
			
			for(size_t y = 0; y < numY; ++y){
				for(size_t x = 0; x < numX; ++x){
					for(size_t i = 0; i < lhs.m_vecDim[1];++i)
					{
						tempN+=(N)lhs[x][i] * (N)rhs[i][y];
					}
					temp[numX*y+x]=tempN;
					tempN=0;
				}
			}
			
			return cuMatrix<N>(temp,vector<size_t>({numX,numY}),memPermission::owner);
		}
		
		/**
		 * MATRIX OPERATION // TODO rework
		 */
		friend N sum(const cuMatrix<N>& matrix){
			N result=0;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i)
				result+= *(matrix.m_data+i);
			return result;
		}

		//simple Transpose only for 2D Matrix tested
		friend cuMatrix<N> T(const cuMatrix<N>& matrix){
			vector<size_t> transVec(matrix.m_vecDim);
			swap(transVec[0],transVec[1]);
			
			size_t numX = matrix.m_vecDim[0];
			size_t numY = matrix.m_vecDim[1];
			N* temp = (N*) malloc(matrix.size()*sizeof(N));
			
			for(size_t x = 0; x < numX; x++){
				for(size_t y = 0; y < numY; y++){
					temp[numX*y+x]=matrix.m_data[numY*x+y];
				}
			}
			return cuMatrix<N>(temp,transVec,memPermission::owner);
		}

		/**
		 * CAST  // TODO rework
		 */
		operator N () const{ return *m_data;}//needed ?

		/**
		 * OUTPUT // TODO rework
		 */
		friend ostream& operator<< (ostream& os, const cuMatrix<N>& matrix){
			if (matrix.nDim() == 0)
				os <<(N) *matrix.m_data;
			else{
				for(size_t i = 0; i < matrix.m_vecDim[0]; ++i)
					os << matrix[i] << "|";
			}
			return os;
		}

		N* m_data;//-> to private after tests
	private:
			
		enum memPermission m_perm;
		vector<size_t> m_vecDim;

};
}
