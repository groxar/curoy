#pragma once
#include "dimIterator.hpp"
#include "xMatrix.hpp"
#include "cuMatrix.hu"
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
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

namespace curoy{
template<typename N>
class cuMatrix{
	public:
		/**
		 * CONSTRUCTOR
		 */
		cuMatrix():m_data(nullptr),m_perm(memPermission::user){}
		cuMatrix(N* data, initializer_list<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm) {}
		cuMatrix(N* data, vector<size_t> dim, enum memPermission mPerm = memPermission::user) : m_data(data), m_vecDim(dim), m_perm(mPerm){}
		cuMatrix(cuMatrix<N>&& matrix) : m_data(matrix.m_data), m_vecDim(move(matrix.m_vecDim)), m_perm(matrix.m_perm) { matrix.m_data = nullptr;}

		cuMatrix(const cuMatrix<N>& matrix){
			m_vecDim = matrix.m_vecDim;
			m_perm = memPermission::user;
			rebase(matrix.size());
			m_perm = memPermission::owner;
			cudaMemcpy(m_data, matrix.m_data, matrix.size() * sizeof(N), cudaMemcpyDeviceToDevice);
		}

		cuMatrix(const xMatrix<N>& matrix){
			m_vecDim = matrix.m_vecDim;
			m_perm = memPermission::user;
			rebase(matrix.size());
			m_perm = memPermission::owner;
			cudaMemcpy(m_data, matrix.m_data, matrix.size() * sizeof(N), cudaMemcpyHostToDevice);
		}


		~cuMatrix(){ 
			if(m_perm == memPermission::owner)
				cudaFree(m_data);
		}


		/**
		 * MISC 
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
		size_t dim(size_t index) const{return m_vecDim[index];}

		void rebase(size_t numElements){
			cudaError_t err;
			if(m_perm == memPermission::user){
				err = cudaMalloc((void**)&m_data,numElements*sizeof(N));
				m_perm = memPermission::owner;
			}
			else if(m_perm == memPermission::owner && numElements != size()){
				cudaFree(m_data);
				err = cudaMalloc((void**)&m_data,numElements*sizeof(N));
			}

			if(m_data == NULL || err != 0)
				cout << "GPU allocation error"<< endl;	
		}

		/**
		 * Data Transfer to/from CUDA device
		 */
		friend cuMatrix<N>& operator>> (const xMatrix<N>& lhs, cuMatrix<N>& rhs){
			rhs.m_vecDim= vector<size_t>(lhs.m_vecDim.begin(),lhs.m_vecDim.end());
			rhs.rebase(lhs.size());
			cudaMemcpy(rhs.m_data,lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyHostToDevice);
			return rhs;
		}
		
		friend xMatrix<N>& operator<< (xMatrix<N>& lhs, const cuMatrix<N>& rhs){
			lhs.m_vecDim = vector<size_t>(rhs.m_vecDim.begin(),rhs.m_vecDim.end());
			lhs.m_data = (N*)malloc(sizeof(N)*rhs.size());
			lhs.rebase(rhs.size());
			cudaMemcpy(lhs.m_data,rhs.m_data,rhs.size()*sizeof(N),cudaMemcpyDeviceToHost);
			return lhs;
		}


		/**
		 * ACCESS
		 */
		cuMatrix<N> operator[](size_t n) const {
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			
			vector<size_t> tempV(++m_vecDim.begin(),end);
			if(tempV.size()==0 && size() > 1  )
				tempV.push_back(1);
			return xMatrix<N>(m_data+n*memJump, tempV,memPermission::diver);
		}

		cuMatrix<N> operator[](vector<size_t> nVec) const {
			if(nVec.size()>m_vecDim.size())
				throw "[] error";
			cuMatrix<N> result(this->m_data,this->m_vecDim,memPermission::diver);
			for(auto n: nVec)
				result = result[n];

			return result;
		}


		/**
		 * MOVE
		 */
		friend inline cuMatrix<N>&& operator! (cuMatrix<N>& matrix){
			if(matrix.m_perm == memPermission::user || matrix.m_perm == memPermission::diver){ //make sure that diver should behave like that
				N* ptr;
				cudaMalloc((void**)&ptr,matrix.size()*sizeof(N));
				cudaMemcpy(ptr,matrix.m_data,matrix.size()*sizeof(N),cudaMemcpyDeviceToDevice);
				matrix.m_data = ptr;
				matrix.m_perm = memPermission::owner;
			}
			return move(matrix);
		}
		
		friend inline cuMatrix<N>&& operator! (cuMatrix<N>&& matrix){
			return move(matrix);
		}


		/**
		 * ASSIGNMENT
		 */
		cuMatrix<N>& operator= (cuMatrix<N>&& rhs){
			m_data = rhs.m_data;
			m_vecDim = move(rhs.m_vecDim);
			m_perm = rhs.m_perm;
			rhs.m_data = nullptr;
			return *this;
		}
		
		cuMatrix<N>& operator= (const cuMatrix<N>& rhs){
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
				cudaMemcpy(m_data,rhs.m_data, rhs.size()*sizeof(N),cudaMemcpyDeviceToDevice);
			}

			return *this;
		}

		cuMatrix<N>& operator= (const N value){// NOT FINISHED
			if(size()==1)
				cudaMemcpy(m_data,&value,sizeof(N),cudaMemcpyHostToDevice);
			else
				cout << "Assignment error" << endl;//fix it
			return *this;
		}
		
		/**
		 * MATRIX FILL
		 */
		friend cuMatrix<N> fill(cuMatrix<N>& matrix, N number){	
			matrix.rebase(matrix.size());	

			for(size_t i= 0; i < matrix.size();++i ) 
				matrix.m_data[i] = number;
			return matrix;
		}

		friend cuMatrix<N>&& fill(cuMatrix<N>&& matrix, N number){
			for(size_t i = 0; i < matrix.size();++i ) 
				matrix.m_data[i] = number;
			return move(matrix);
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

		//double rvalue
		friend inline cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs){
			return move(!lhs + rhs); 
		}

		friend inline cuMatrix<N>&& operator+ (cuMatrix<N>& lhs, cuMatrix<N>&& rhs){
			return move(!rhs + lhs); 
		}
		
		friend cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, cuMatrix<N>& rhs){ 
			size_t numElements = lhs.size();
			addDev<N>(lhs.m_data,rhs.m_data,lhs.m_data,numElements);	
			return move(lhs);
		}	


		/**
		 * ADDITION SKALAR
		 */

		friend cuMatrix<N> operator+ (const cuMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] + rhs;
			return result;
		}

		friend cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] + rhs;
			return move(lhs);
		}
		
		/**
		 * SUBSTRACTION
		 */
		friend cuMatrix<N> operator- (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return result;
		}	

		//double rvalue
		friend inline cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs){
			return move(!lhs - rhs); 
		}

		friend inline cuMatrix<N>&& operator- (cuMatrix<N>& lhs, cuMatrix<N>&& rhs){
			return move((!rhs) * -1 + lhs); //fix after implementing scalar
		}
	
		friend cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, cuMatrix<N>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] - rhs.m_data[i];
			return move(lhs);
		}
		
		/**
		 * SUBTRACTION SKALAR
		 */

		friend cuMatrix<N> operator- (const cuMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] - rhs;
			return result;
		}

		friend cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] -  rhs;
			return move(lhs);
		}


		/**
		 * MULTIPLICATION
		 */
		//simple R^2 matrix multiplikation (1,2)
	 	friend cuMatrix<N> mult (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			if(lhs.m_vecDim[1]!=rhs.dim(0) || lhs.m_vecDim.size()!=2 || rhs.m_vecDim.size()!=2){
				throw "DIMENSIONS DONT FIT";
				return lhs;
			}

			size_t numX = lhs.dim(0);
			size_t numY = rhs.dim(1);
			N* temp;
			cudaMalloc((void**)&temp,numX*numY*sizeof(N));

			multDev(lhs.m_data,rhs.m_data,temp,numX,lhs.dim(1),numY);		
			
			return cuMatrix<N>(temp,vector<size_t>({numX,numY}),memPermission::owner);
		}
		
		/**
		 * MULTIPLICATION SKALAR
		 */

		friend cuMatrix<N> operator* (const cuMatrix<N>& lhs,const N rhs) {
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] * rhs;
			return result;
		}

		friend cuMatrix<N>&& operator* (cuMatrix<N>&& lhs,const N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] * rhs;
			return move(lhs);
		}
		
	
		/**
		 * DIVISION SKALAR
		 */
		friend cuMatrix<N> operator/ (const cuMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] / rhs;
			return result;
		}

		friend cuMatrix<N>&& operator/ (cuMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] / rhs;
			return move(lhs);
		}
		

		/**
		 * DIVISION SKALAR
		 */
		friend cuMatrix<N> operator% (const cuMatrix<N>& lhs, N rhs) {
			size_t numElements = lhs.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),lhs.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = lhs.m_data[i] % rhs;
			return result;
		}

		friend cuMatrix<N>&& operator% (cuMatrix<N>&& lhs, N rhs) {
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_data[i] = lhs.m_data[i] % rhs;
			return move(lhs);
		}


		/**
		 * MATRIX OPERATION
		 */
		friend cuMatrix<N> log(const cuMatrix<N>& matrix){
			size_t numElements = matrix.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = log(matrix.m_data[i]);
			return result;
		}
		
		friend cuMatrix<N>&& log(cuMatrix<N>&& matrix) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = log(matrix.m_data[i]);
			return move(matrix);
		}

		
		friend cuMatrix<N> log10(const cuMatrix<N>& matrix){
			size_t numElements = matrix.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = log10(matrix.m_data[i]);
			return result;
		}
		
		friend cuMatrix<N>&& log10(cuMatrix<N>&& matrix) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = log10(matrix.m_data[i]);
			return move(matrix);
		}

		friend cuMatrix<N> pow(const cuMatrix<N>& matrix, N exponent){
			size_t numElements = matrix.size();
			cuMatrix<N> result((N*)malloc(numElements*sizeof(N)),matrix.m_vecDim,memPermission::owner);
			for(int i = 0; i < numElements; ++i)
				result.m_data[i] = pow(matrix.m_data[i],exponent);
			return result;
		}

		friend cuMatrix<N>&& pow(cuMatrix<N>&& matrix, N exponent) {
			size_t numElements = matrix.size();
			for(int i = 0; i < numElements; ++i)
				matrix.m_data[i] = pow(matrix.m_data[i],exponent);
			return move(matrix);
		}

		friend N sum(const cuMatrix<N>& matrix){
			return sum(matrix.m_data,matrix.size());
		}
		
		friend N prod(const cuMatrix<N>& matrix){
			N result=0;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i)
				result*= *(matrix.m_data+i);
			return result;
		}

		friend bool eq(const xMatrix<N>& lhs, const cuMatrix<N>& rhs){
			return eq(rhs,lhs);
		}
		friend bool eq(const cuMatrix<N>& lhs, const xMatrix<N>& rhs){
			if(dimCompare(lhs.dim(),rhs.dim())!=0)
				return false;
			xMatrix<N> matrix;
			matrix << lhs;
			size_t end = matrix.size();
			for(size_t i=0; i<end;++i){
				if(matrix.m_data[i] != rhs.m_data[i])
					return false;
			}
			return true;
		}

		// TODO IMPLEMENT
		friend cuMatrix<N> pinv(const cuMatrix<N>& matrix){
			cout << "pinv() is unimplemented"<<endl;
			return matrix;
		}

		//simple Transpose only for 2D Matrix tested
		friend cuMatrix<N> T(const cuMatrix<N>& matrix){
			vector<size_t> transVec(matrix.m_vecDim);
			swap(transVec[0],transVec[1]);
			
			size_t numX = matrix.dim(0);
			size_t numY = matrix.dim(1);
			N* temp;
			cudaMalloc((void**) &temp, numX*numY*sizeof(N));
			
			transposeDev(matrix.m_data,temp,numX,numY);
			return cuMatrix<N>(temp,transVec,memPermission::owner);
		}

		/**
		 * CAST 
		 */
		operator N () const{ 
			N result;
			cudaMemcpy(&result,m_data,sizeof(N),cudaMemcpyDeviceToHost);
			return result;
		}

		/**
		 * OUTPUT
		 */
		friend ostream& operator<< (ostream& os, const cuMatrix<N>& matrix){
			xMatrix<N> tempMatrix;
			tempMatrix << matrix;
			return os << tempMatrix;
		}

		N* m_data;//-> to private after tests
		vector<size_t> m_vecDim;
	private:
		enum memPermission m_perm;
};
}
