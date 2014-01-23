#include <vector>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <type_traits> 

using namespace std;

#define DEBUG(x) do { cout << #x << ": " << x << std::endl; } while (0)

class matrixDimException: public exception
{
	virtual const char* what() const throw()
	{
		return "the dimensions don't fit";
	}
} matrixDE;


template<typename T>
class cMatrix{
	public:
		cMatrix():m_matrix(nullptr){};
		cMatrix(T* data, initializer_list<size_t> dim) : m_matrix(data),m_vecDim(dim) {}
		cMatrix(T* data, vector<size_t> dim) : m_matrix(data),m_vecDim(dim) {}
		~cMatrix() {/*delete [] m_matrix; DONT DELETE*/}
		cMatrix(cMatrix<T>&& matrix) : m_matrix(matrix.m_matrix),m_vecDim(move(matrix.m_vecDim)) { matrix.m_matrix = NULL;}
		size_t dim() const{return m_vecDim.size();}
		size_t size() const{
			size_t numElements = 1;
			auto end = this->m_vecDim.end();
			for(auto i = this->m_vecDim.begin(); i != end; ++i)
				numElements*= *i;
			return numElements;
		}
		vector<size_t> getDim() const{return m_vecDim;}

		/**
		 * Access
		 */
		cMatrix<T> operator[](size_t n) const{ //access on const
			size_t memJump = 1;

			auto end = m_vecDim.end();
			for(auto i = (++m_vecDim.begin()); i != end; ++i)
				memJump*= *i;
			return cMatrix<T>(m_matrix+n*memJump, vector<size_t>(++m_vecDim.begin(),end));
		}

		cMatrix<T> operator[](vector<size_t> nVec) const{
			if(nVec.size()>m_vecDim.size())
				throw nullptr;
			cMatrix<T> result(this->m_matrix,this->m_vecDim);
			for(auto n: nVec)
			{
				result = result[n];
			}

			return result;
		}

		friend inline cMatrix<T>&& operator! (cMatrix<T>& matrix){
			return move(matrix);
		}

		/**
		 * ASSIGNMENT
		 */
		cMatrix<T>& operator= (cMatrix<T> rhs){
			this->m_matrix = rhs.m_matrix;
			this->m_vecDim = rhs.m_vecDim;//FULL COPY
			return *this;
		}

		/**
		 * ADDITION
		 */
		friend cMatrix<T> operator+ (cMatrix<T>& lhs, cMatrix<T>& rhs){
			size_t numElements = lhs.size();
			cMatrix<T> result((T*)malloc(numElements*sizeof(T)),lhs.m_vecDim);
			for(int i = 0; i < numElements; ++i)
				result.m_matrix[i] = lhs.m_matrix[i] + rhs.m_matrix[i];
			return result;
		}

		friend inline cMatrix<T>&& operator+ (cMatrix<T>& lhs, cMatrix<T>&& rhs){
			return move(!rhs + lhs); 
		}
		
		friend cMatrix<T>&& operator+ (cMatrix<T>&& lhs, cMatrix<T>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_matrix[i] = lhs.m_matrix[i] + rhs.m_matrix[i];
			return move(lhs);
		}
		
		/**
		 * SUBSTRAKTION
		 */
		friend cMatrix<T> operator- (cMatrix<T>& lhs, cMatrix<T>& rhs){
			size_t numElements = lhs.size();
			cMatrix<T> result((T*)malloc(numElements*sizeof(T)),lhs.m_vecDim);
			for(int i = 0; i < numElements; ++i)
				result.m_matrix[i] = lhs.m_matrix[i] - rhs.m_matrix[i];
			return result;
		}

		friend inline cMatrix<T>&& operator- (cMatrix<T>& lhs, cMatrix<T>&& rhs){
			return move(!rhs - lhs); 
		}
		
		friend cMatrix<T>&& operator- (cMatrix<T>&& lhs, cMatrix<T>& rhs){
			size_t numElements = lhs.size();
			for(int i = 0; i < numElements; ++i)
				lhs.m_matrix[i] = lhs.m_matrix[i] - rhs.m_matrix[i];
			return move(lhs);
		}

		/**
		 * MULTIPLICATION
		 */
		friend cMatrix<T> operator* (cMatrix<T>& lhs, cMatrix<T>& rhs){
						
		}

		operator T () const{ return *m_matrix;}//needed ?

		friend ostream& operator<< (ostream& os, const cMatrix<T>& matrix){
			if (matrix.dim() == 0)
				os << *matrix.m_matrix;
			else{
				for(size_t i = 0; i < matrix.m_vecDim[0]; ++i)
					os << matrix[i] << "|";
			}
			return os;
		}

		T* m_matrix;//-> to private after tests
	private:
		
		vector<size_t> m_vecDim;
};

int main(void){
	int data[2][3] = {{1,2,3},{4,5,6}};
	int data2[2][2][3] = 	{{{1,2,3},{4,5,6}}, 
							{{7,8,9},{10,11,12}}};
	vector<vector<int>> vvi({{1,2,3},{4,5,6}});
	vector<vector<int>> vvi2({{1,2},{3,4},{5,6}});
	int* dataP = (int*)data;
	int* dataP2 = (int*)data2;
	cMatrix<int> matrix(dataP,{2,3});
	cMatrix<int> matrix1(dataP,{3,2});
	cMatrix<int> matrix2(dataP2,{2,2,3});

	DEBUG(matrix.dim());

	cout << "getDim(): " << endl;
	for(auto i : matrix.getDim())
		cout << i << " ";
	cout << endl;

	cout<< endl << "Dimesion Test"<< endl;
	DEBUG(matrix[1].dim());
	DEBUG(matrix2[1].dim());
	DEBUG(matrix[0][0].dim());
	DEBUG(matrix[vector<size_t>({1,1})]);
	DEBUG(matrix[0][0]);
	DEBUG(matrix[0][1]);
	DEBUG(matrix[0][2]);
	DEBUG(matrix[1][0]);
	DEBUG(matrix[1][1]);
	DEBUG(matrix[1][2]);
	DEBUG(matrix2[1][0][2]);
	cout<< endl << "Output Test"<< endl;
	DEBUG(matrix);
	DEBUG(matrix2);
	cout<< endl << "Cast Test"<< endl;
	DEBUG((int) matrix[0]);
	DEBUG((int) matrix[1]);
	DEBUG(data2[1]);
	DEBUG(data2[1][0][2]);
	cout<< endl << "Math Test"<< endl;
	cout << (matrix+matrix)[1][2] << endl;
	cout << matrix[1][1]<< endl;
	cout << matrix.m_matrix<< endl;
	cout << (matrix+matrix)[1][1]<< endl;
	cout << (matrix+matrix).m_matrix<< endl;
	cout << "!matrix2+matrix2" << endl;
	cout << matrix2<< endl;
	cout << matrix2.m_matrix<< endl;
	cout << (!matrix2+matrix2).m_matrix<< endl;
	cout << matrix2[1][1]<< endl;
	cout << matrix2<< endl;
	cout << matrix2.m_matrix<< endl;
	cout << (matrix2 + matrix2)[1][1]<< endl;
	cout << matrix2 << endl;
	cout << "matrix2+!matrix2" << endl;
	cout << matrix2<< endl;
	cout << matrix2.m_matrix<< endl;
	cout << (matrix2+!matrix2).m_matrix<< endl;
	cout << matrix2[1][1]<< endl;
	cout << matrix2<< endl;
	cout << matrix2.m_matrix<< endl;
	cout << (matrix2 + matrix2 + matrix2)[1][1]<< endl;
	cout << matrix2 << endl;

	cout << "getDim(): " << endl;
	for(auto i : matrix2.getDim())
		cout << i << " ";
	cout << endl;


	return 0;
}

