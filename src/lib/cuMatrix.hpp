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
#include <tuple>
#include <future>
#include "matrixException.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

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
		cuMatrix(vector<size_t> dim, enum fillMode mode):m_data(nullptr){m_perm=memPermission::user;resize(dim);if(mode==fillMode::rnd){fillRnd(*this);}}
		cuMatrix(vector<size_t> dim, N value):m_data(nullptr){m_perm=memPermission::user;resize(dim);fill(*this,value);}
		cuMatrix(cuMatrix<N>&& matrix) : m_data(matrix.m_data), m_vecDim(move(matrix.m_vecDim)), m_perm(matrix.m_perm) { matrix.m_data = nullptr;}

		cuMatrix(const cuMatrix<N>& matrix){
			m_perm = memPermission::user;
			resize(matrix.m_vecDim);
			cudaMemcpy(m_data, matrix.m_data, matrix.size() * sizeof(N), cudaMemcpyDeviceToDevice);
		}

		cuMatrix(const xMatrix<N>& matrix){
			m_perm = memPermission::user;
			resize(matrix.dim());
			cudaMemcpy(m_data, matrix.m_data, matrix.size() * sizeof(N), cudaMemcpyHostToDevice);
		}

		~cuMatrix(){ 
			cudaError_t err = cudaSuccess;
			if(m_perm == memPermission::owner){
				err =cudaFree(m_data);
				if(err!=cudaSuccess)
					cout << cudaGetErrorString(err)<<endl;	
			}
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
			m_vecDim = vecDim;
			rebase(size());
		}

		/**
		 * Data Transfer to/from CUDA device
		 */
		friend cuMatrix<N>& operator>> (const xMatrix<N>& lhs, cuMatrix<N>& rhs){
			rhs.resize(lhs.dim());
			cudaMemcpy(rhs.m_data,lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyHostToDevice);
			return rhs;
		}
		
		friend xMatrix<N>& operator<< (xMatrix<N>& lhs, const cuMatrix<N>& rhs){
			lhs.resize(rhs.dim());
			cudaMemcpy(lhs.m_data,rhs.m_data,rhs.size()*sizeof(N),cudaMemcpyDeviceToHost);
			return lhs;
		}


		/**
		 * ACCESS
		 */
		cuMatrix<N> operator[](size_t n) const {
			if(nDim() == 0 || n >= dim(0)){
				cout << "[] overflow error"<<endl;
				return cuMatrix<N>();
			}
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			
			vector<size_t> tempV(++m_vecDim.begin(),end);
			if(tempV.size()==0 && size() > 1  )
				tempV.push_back(1);
			return cuMatrix<N>(m_data+n*memJump, tempV,memPermission::diver);
		}

		cuMatrix<N> operator[](vector<size_t> nVec) const {
			cuMatrix<N> result(this->m_data,this->m_vecDim,memPermission::diver);
			for(auto n: nVec)
				result = result[n];

			return result;
		}

		//2D only
		cuMatrix<N> operator()(vector<size_t> firstRange,vector<size_t> secondRange){
			if(firstRange.size() == nDim() == secondRange.size()==2){
				cout << "Range error in ()"<< endl;
				return cuMatrix<N>();
			}
			vector<size_t> newDimension;
			size_t size = 1;
			size_t end = nDim();

			if(firstRange[0]>firstRange[1] && dim(0)>=firstRange[1]){
				cout <<"ranges have to be ascending, invalid Input: {"<< firstRange[0]<<","<< firstRange[1]<<"}"<<endl;
				return cuMatrix<N>();
			}
			newDimension.push_back(firstRange[1]-firstRange[0]+1);
			size*=newDimension[0];

			if(secondRange[0]>secondRange[1] && dim(1)>=secondRange[1]){
				cout <<"ranges have to be ascending, invalid Input: {"<< secondRange[0]<<","<< secondRange[1]<<"}"<<endl;
				return cuMatrix<N>();
			}
			newDimension.push_back(secondRange[1]-secondRange[0]+1);
			size*=newDimension[1];

			cuMatrix<N> result(newDimension,fillMode::none);

			for(int i = firstRange[0]; i <= firstRange[1]; ++i){
				cudaMemcpy(&(result.m_data[newDimension[1]*i]),&(m_data[dim(1)*i+secondRange[0]]),sizeof(N)*(newDimension[1]),cudaMemcpyDeviceToDevice);
			}
			return result;
		}
		
		// TODO decide if this is a workaround, fix or a stupid idea. Most likely it's the last one
		N operator~(){
			N result = 0;
			if(size() < 0)
				cout << "~ access error"<<endl;
			else
				cudaMemcpy(&result,m_data,sizeof(N),cudaMemcpyDeviceToHost);
			return result;
		}

		/**
		 * MOVE
		 */
		friend inline cuMatrix<N>&& operator! (cuMatrix<N>& matrix){
			if(matrix.m_perm != memPermission::owner){
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

			if(m_perm==memPermission::owner)
				cudaFree(m_data);
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

		cuMatrix<N>& operator= (const N value){
			this->rebase(this->size());
			fillDev(this->m_data,value,this->size());
			return *this;
		}

		/**
		 * CAST
		 */
		
		template<typename M>
		operator cuMatrix<M>(){
			cuMatrix<M> result(this->m_vecDim,fillMode::none);
			castDev(this->m_data,result.m_data,this->size());
			return result;
		}
		/**
		 * MATRIX FILL
		 */
		friend cuMatrix<N> fill(cuMatrix<N>& matrix,const N value){	
			matrix.rebase(matrix.size());	
			fillDev(matrix.m_data, value, matrix.size());
			return matrix;
		}

		friend cuMatrix<N>&& fill(cuMatrix<N>&& matrix,const N value){
			fillDev(matrix.m_data, value, matrix.size());
			return move(matrix);
		}

		friend cuMatrix<N>&& fillRnd(cuMatrix<N>& matrix){	
			matrix.rebase(matrix.size());	
			return move(fillRnd(!matrix));
		}
		friend cuMatrix<N>&& fillRnd(cuMatrix<N>&& matrix){
			curandGenerator_t gen;
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(gen, 4651635983ULL);
			curandGenerateUniformDouble(gen, (double*) matrix.m_data, matrix.size()); // TODO FIX (double*)
			return move(matrix);
		}

		/**
		 * MATRIX Concatination
		 */	
		friend cuMatrix<N> operator& (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			if(!((dimCompare(lhs.dim(),rhs.dim())==1&&lhs.dim(0)!=rhs.dim(0) )||dimCompare(lhs.dim(),rhs.dim())==0)){
				cout << "concatination error"<<endl;
				return lhs;
			}
			vector<size_t>  vecDim = lhs.dim(); 
			vecDim[0] = lhs.dim(0) + rhs.dim(0);
			cuMatrix<N> result(vecDim);
			cudaMemcpy(result.m_data,lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&(result.m_data[lhs.size()]),rhs.m_data,rhs.size()*sizeof(N),cudaMemcpyDeviceToDevice);

			return result;
		}
		friend cuMatrix<N> operator& (const cuMatrix<N>& lhs,const N value){
			vector<size_t>  vecDim = lhs.dim(); 
			vecDim[0] = lhs.dim(0) + 1;
			cuMatrix<N> result(vecDim,value);
			cudaMemcpy(result.m_data,lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyDeviceToDevice);
			
			return result;
		}
		friend cuMatrix<N> operator& (const N value, const cuMatrix<N>& lhs){
			vector<size_t>  vecDim = lhs.dim(); 
			vecDim[0] = lhs.dim(0) + 1;
			cuMatrix<N> result(vecDim,value);
			cudaMemcpy(&(result.m_data[lhs.dim(1)]),lhs.m_data,lhs.size()*sizeof(N),cudaMemcpyDeviceToDevice);
			
			return result;
		}
		friend cuMatrix<N> operator| (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			return T(T(lhs)&T(rhs));
		}
		friend cuMatrix<N> operator| (const cuMatrix<N>& lhs,const N value){
			return T(T(lhs)&value);
		}
		friend cuMatrix<N> operator| (const N value, const cuMatrix<N>& lhs){
			return T(value&T(lhs));
		}
	
	
		/**
		 * Matrix operator Map
		 */
		template<typename FUNC>
		friend cuMatrix<N> mapFunc(FUNC f,const cuMatrix<N>& lhs, const cuMatrix<N>& rhs){
			if(dimCompare(lhs.dim(),rhs.dim())!=0){
				cout << "mapFunc Failed {"<< lhs.dim(0)<<","<<lhs.dim(1)<<"}  {"<<rhs.dim(0)<<","<<rhs.dim(1)<<"}"<<endl;
				return lhs;
			}
			size_t numElements = lhs.size();
			cuMatrix<N> result;
			result.resize(lhs.dim());
			f(lhs.m_data,rhs.m_data,result.m_data,numElements);	
			return result;
		}

		template<typename FUNC>
		friend cuMatrix<N>&& mapFunc(FUNC f, cuMatrix<N>&& lhs, const cuMatrix<N>& rhs){
			if(dimCompare(lhs.dim(),rhs.dim())!=0){
				cout << "mapFunc Failed {"<< lhs.dim(0)<<","<<lhs.dim(1)<<"}  {"<<rhs.dim(0)<<","<<rhs.dim(1)<<"}"<<endl;
				return move(lhs);
			}
			size_t numElements = lhs.size();
			f(lhs.m_data,rhs.m_data,lhs.m_data,numElements);	
			return move(lhs);
		}


		/**
		 * Matrix operation skalar
		 */
		template<typename FUNC,typename... M>
		friend cuMatrix<N> mapFunc(FUNC f,const cuMatrix<N>& lhs, M... values){
			size_t numElements = lhs.size();
			cuMatrix<N> result;
			result.resize(lhs.dim());
			f(lhs.m_data,values...,result.m_data,numElements);	
			return result;
		}

		template<typename FUNC,typename... M>
		friend cuMatrix<N>&& mapFunc(FUNC f, cuMatrix<N>&& lhs, M... values){
			size_t numElements = lhs.size();
			f(lhs.m_data,values...,lhs.m_data,numElements);	
			return move(lhs);
		}

		/**
		 * ADDITION
		 */
		friend inline cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs + rhs); }
		friend inline cuMatrix<N>&& operator+ (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move(!rhs + lhs); }
		friend inline cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::addDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator+ (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs) 	{ return move(mapFunc(&curoy::addDev<N>,lhs,rhs)); }

		/**
		 * ADDITION SKALAR
		 */
		friend cuMatrix<N>   operator+ (const N lhs, const cuMatrix<N>& rhs) 	{ return rhs + lhs; }
		friend cuMatrix<N>&& operator+ (const N lhs, cuMatrix<N>&& rhs) 		{ return move(!rhs + lhs); }
		friend cuMatrix<N>   operator+ (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::addSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator+ (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::addSkalarDev<N>,!lhs,rhs)); }
		
		/**
		 * SUBSTRACTION
		 */
		friend inline cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs - rhs); }
		friend inline cuMatrix<N>&& operator- (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move((!rhs*-1) + lhs); }
		friend inline cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::subDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator- (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs) 	{ return move(mapFunc(&curoy::subDev<N>,lhs,rhs)); }
		
		/**
		 * SUBTRACTION SKALAR
		 */
		friend cuMatrix<N>   operator- (const N lhs, const cuMatrix<N>& rhs) 	{ return move(-rhs + lhs); }
		friend cuMatrix<N>&& operator- (const N lhs, cuMatrix<N>&& rhs) 		{ return move(-!rhs + lhs); }
		friend cuMatrix<N>   operator- (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::subSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator- (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::subSkalarDev<N>,!lhs,rhs)); }
		
		/**
		 * Elementwise Multiplcation 
		 */
		friend inline cuMatrix<N>&& operator* (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs * rhs); }
		friend inline cuMatrix<N>&& operator* (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move(!rhs * lhs); }
		friend inline cuMatrix<N>&& operator* (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::mulDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator* (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs) 	{ return move(mapFunc(&curoy::mulDev<N>,lhs,rhs)); }

		/**
		 * Elementwise Multiplcation SKALAR
		 */
		friend cuMatrix<N>   operator* (const N lhs, const cuMatrix<N>& rhs) 	{ return rhs * lhs; }
		friend cuMatrix<N>&& operator* (const N lhs, cuMatrix<N>&& rhs) 		{ return move(!rhs * lhs); }
		friend cuMatrix<N>   operator* (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::mulSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator* (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::mulSkalarDev<N>,!lhs,rhs)); }
	
		/**
		 * Divison 
		 */
		friend inline cuMatrix<N>&& operator/ (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs / rhs); }
		friend inline cuMatrix<N>&& operator/ (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move(pow(!rhs,-1) * lhs); }
		friend inline cuMatrix<N>&& operator/ (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::divDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator/ (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs) 	{ return move(mapFunc(&curoy::divDev<N>,lhs,rhs)); }

		/**
		 * Divison Skalar
		 */
		friend cuMatrix<N>   operator/ (const N lhs, const cuMatrix<N>& rhs) 	{ return mapFunc(&divReverseSkalarDev<N>,rhs,lhs); }
		friend cuMatrix<N>&& operator/ (const N lhs, cuMatrix<N>&& rhs) 		{ return mapFunc(&divReverseSkalarDev<N>,!rhs,lhs); }
		friend cuMatrix<N>   operator/ (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::divSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator/ (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::divSkalarDev<N>,!lhs,rhs)); }
	
		/**
		 * Equal
		 */
		friend inline cuMatrix<N>&& operator== (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs == rhs); }
		friend inline cuMatrix<N>&& operator== (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move(!rhs == lhs); }
		friend inline cuMatrix<N>&& operator== (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::eqDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator== (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs) 	{ return move(mapFunc(&curoy::eqDev<N>,lhs,rhs)); }

		/**
		 * Equal Skalar
		 */
		friend cuMatrix<N>   operator== (const N lhs, const cuMatrix<N>& rhs) 	{ return rhs == lhs; }
		friend cuMatrix<N>&& operator== (const N lhs, cuMatrix<N>&& rhs) 		{ return !rhs == lhs; }
		friend cuMatrix<N>   operator== (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::eqSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator== (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::eqSkalarDev<N>,!lhs,rhs)); }

		/**
		 * not Equal
		 */
		friend inline cuMatrix<N>&& operator!= (cuMatrix<N>&& lhs, cuMatrix<N>&& rhs)			{ return move(!lhs != rhs); }
		friend inline cuMatrix<N>&& operator!= (const cuMatrix<N>& lhs, cuMatrix<N>&& rhs)		{ return move(!rhs != lhs); }
		friend inline cuMatrix<N>&& operator!= (cuMatrix<N>&& lhs, const cuMatrix<N>& rhs)		{ return move(mapFunc(&curoy::neqDev<N>,!lhs,rhs)); }	
		friend inline cuMatrix<N>   operator!= (const cuMatrix<N>& lhs, const cuMatrix<N>& rhs)	{ return move(mapFunc(&curoy::neqDev<N>,lhs,rhs)); }

		/**
		 * not Equal Skalar
		 */
		friend cuMatrix<N>   operator!= (const N lhs, const cuMatrix<N>& rhs) 	{ return rhs != lhs; }
		friend cuMatrix<N>&& operator!= (const N lhs, cuMatrix<N>&& rhs) 		{ return !rhs != lhs; }
		friend cuMatrix<N>   operator!= (const cuMatrix<N>& lhs, const N rhs) 	{ return move(mapFunc(&curoy::neqSkalarDev<N>,lhs,rhs)); }
		friend cuMatrix<N>&& operator!= (cuMatrix<N>&& lhs, const N rhs) 		{ return move(mapFunc(&curoy::neqSkalarDev<N>,!lhs,rhs)); }

		/**
		 * Elementwise Matrix Operation
		 */

		friend cuMatrix<N>   operator- (const cuMatrix<N>& lhs) { return move(lhs*-1); }
		friend cuMatrix<N>&& operator- (cuMatrix<N>&& lhs) 		{ return move(!lhs*-1); }

		friend cuMatrix<N>   operator^ (const cuMatrix<N>& lhs, const N exponent) { return move(mapFunc(&curoy::powDev<N>,lhs,exponent)); }
		friend cuMatrix<N>&& operator^ (cuMatrix<N>&& lhs, const N exponent) 		{ return move(mapFunc(&curoy::powDev<N>,!lhs,exponent)); }

		friend cuMatrix<N>   pow (const cuMatrix<N>& lhs, const N exponent) { return move(mapFunc(&curoy::powDev<N>,lhs,exponent)); }
		friend cuMatrix<N>&& pow (cuMatrix<N>&& lhs, const N exponent) 		{ return move(mapFunc(&curoy::powDev<N>,!lhs,exponent)); }
	
		friend cuMatrix<N>   exp (const cuMatrix<N>& lhs)	{ return move(mapFunc(&curoy::expDev<N>,lhs)); }
		friend cuMatrix<N>&& exp (cuMatrix<N>&& lhs) 		{ return move(mapFunc(&curoy::expDev<N>,!lhs)); }

		friend cuMatrix<N>   log (const cuMatrix<N>& lhs) 	{ return move(mapFunc(&curoy::logDev<N>,lhs)); }
		friend cuMatrix<N>&& log (cuMatrix<N>&& lhs) 		{ return move(mapFunc(&curoy::logDev<N>,!lhs)); }
		
		friend cuMatrix<N>   log10 (const cuMatrix<N>& lhs) { return move(mapFunc(&curoy::log10Dev<N>,lhs)); }
		friend cuMatrix<N>&& log10 (cuMatrix<N>&& lhs) 		{ return move(mapFunc(&curoy::log10Dev<N>,!lhs)); }

		/**
		 * MATRIX MULTIPLICATION
		 */
		//simple R^2 matrix multiplikation (1,2)
	 	friend cuMatrix<N> mult (const cuMatrix<N>& lhs,const cuMatrix<N>& rhs){
			if(lhs.m_vecDim[1]!=rhs.dim(0) || lhs.m_vecDim.size()!=2 || rhs.m_vecDim.size()!=2){
				cout << "DIMENSIONS DONT FIT" << endl;
				return lhs;
			}

			size_t numX = lhs.dim(0);
			size_t numY = rhs.dim(1);
			N* temp;
			cudaMalloc((void**)&temp,numX*numY*sizeof(N));

			multDev(lhs.m_data,rhs.m_data,temp,numX,lhs.dim(1),numY);		
			
			return cuMatrix<N>(temp,vector<size_t>({numX,numY}),memPermission::owner);
		}

		friend N sum(const cuMatrix<N>& matrix){
			return sumDev(matrix.m_data,matrix.size());
		}

		//2D dimension sum, rework after implementing align(Transpose with any dimesions)
		friend cuMatrix<N> sum(const cuMatrix<N>& matrix, size_t dimension){
			N* temp;
			cudaMalloc((void**) &temp,matrix.dim((dimension+1)%2)*sizeof(N));
			vector<size_t> tempV(matrix.m_vecDim);
			tempV[dimension]=1;

			if(dimension == 1)
				sumColumneDev(matrix.m_data,temp,matrix.dim(0),matrix.dim(1));
			else
				sumColumneDev(T(matrix).m_data,temp,matrix.dim(1),matrix.dim(0));

			return cuMatrix<N> (temp,tempV,memPermission::owner);
		}

		friend N max(const cuMatrix<N>& matrix){
			return maxDev(matrix.m_data,matrix.size());
		}
		//2D dimension max, rework after implementing align(Transpose with any dimesions)
		friend cuMatrix<N> max(const cuMatrix<N>& matrix, size_t dimension){
			N* temp;
			cudaMalloc((void**) &temp,matrix.dim((dimension+1)%2)*sizeof(N));
			vector<size_t> tempV(matrix.m_vecDim);
			tempV[dimension]=1;

			if(dimension == 1)
				maxColumneDev(matrix.m_data,temp,matrix.dim(0),matrix.dim(1));
			else
				maxColumneDev(T(matrix).m_data,temp,matrix.dim(1),matrix.dim(0));

			return cuMatrix<N> (temp,tempV,memPermission::owner);
		}
		//2D dimension maxPos, rework after implementing align(Transpose with any dimesions)
		friend tuple<cuMatrix<N>,cuMatrix<size_t>> maxPos(const cuMatrix<N>& matrix, size_t dimension){
			N* temp;
			size_t* posTemp;
			size_t position;
			cudaMalloc((void**) &temp,matrix.dim((dimension+1)%2)*sizeof(N));
			cudaMalloc((void**) &posTemp,matrix.dim((dimension+1)%2)*sizeof(N));
			vector<size_t> tempV(matrix.m_vecDim);
			tempV[dimension]=1;

			if(dimension == 1)
				maxPosColumneDev(matrix.m_data,temp,posTemp,matrix.dim(0),matrix.dim(1));
			else
				maxPosColumneDev(T(matrix).m_data,temp,posTemp,matrix.dim(1),matrix.dim(0));

			return make_tuple(cuMatrix<N>(temp,tempV,memPermission::owner),
							  cuMatrix<size_t>(posTemp,tempV,memPermission::owner));
		}

		friend N prod(const cuMatrix<N>& matrix){
			return prodDev(matrix.m_data,matrix.size());
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
		friend bool eq(const cuMatrix<N>& lhs, const cuMatrix<N>& rhs){
			return sum(lhs==rhs) == lhs.size();
		}

		// TODO IMPLEMENT
		friend cuMatrix<N> pinv(const cuMatrix<N>& matrix){
			cout << "pinv() is unimplemented"<<endl;
			return matrix;
		}

		//simple Transpose only for 2D Matrix tested
		friend cuMatrix<N> T(const cuMatrix<N>& matrix){
			vector<size_t> transVec(matrix.m_vecDim);
			if(transVec.size() < 2)
				transVec.push_back(1);
			swap(transVec[0],transVec[1]);
			
			size_t numX = matrix.dim(0);
			size_t numY = matrix.dim(1);
			cuMatrix<N> result(transVec,fillMode::none);
			
			transposeDev(matrix.m_data,result.m_data,numX,numY);
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
	private:
		enum memPermission m_perm;
		vector<size_t> m_vecDim;
		
		void rebase(size_t numElements){
			cudaError_t err = cudaSuccess;
			if(m_perm != memPermission::owner){
				err = cudaMalloc((void**)&m_data,numElements*sizeof(N));
				m_perm = memPermission::owner;
			}
			else if(numElements != size()){
				cudaFree(m_data);
				err = cudaMalloc((void**)&m_data,numElements*sizeof(N));
			}

			if(m_data == NULL || err != 0){
				cout << "GPU allocation error"<< endl;	
				if(m_data == NULL)
					cout << "M Data is NULL"<<endl;
				if(err != 0)
					cout << cudaGetErrorString(err)<<endl;
				cout << "numElements: "<< numElements<<endl;
				cout << "size: "<< size()<<endl;
				if(m_perm == memPermission::owner)
					cout << "memory owner"<<endl;
				else if(m_perm == memPermission::diver)
					cout << "memory diver"<<endl;
				else
					cout << "memory user"<<endl;
			}
		}
};
}
