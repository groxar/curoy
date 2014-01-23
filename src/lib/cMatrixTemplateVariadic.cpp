#include <vector>
#include <cstdint>
#include <numeric>
#include <iostream>
#include <cstdlib>

using namespace std;

#define DEBUG(x) do { cout << #x << ": " << x << std::endl; } while (0)

template<typename T, size_t first, size_t... N>
class cuMatrix{
	public:
		cuMatrix(T* matrix) : m_matrix(matrix),m_arrDim{first,N...} {};
		size_t getNumDim(){return m_arrDim.size();};
		vector<size_t> getDim(){return m_arrDim;};

		cuMatrix<T,N...> operator[](size_t n){
			size_t memJump = 1;
			auto end = m_arrDim.end();
			for(auto i = (++m_arrDim.begin()); i != end; ++i)
				memJump*= *i;
			return cuMatrix<T,N...>(m_matrix+n*memJump);
		}
		cuMatrix<T,first,N...> operator+ (const cuMatrix<T,first,N...>& rhs) const{
			cuMatrix<T,first,N...> result((int*)malloc(sizeof(this->m_matrix)*sizeof(T)));
			size_t numElements = 1;
			auto end = m_arrDim.end();
			for(auto i = m_arrDim.begin(); i != end; ++i)
				numElements*= *i;
			for(int i = 0; i < numElements;++i)
				result.m_matrix[i] = this->m_matrix[i] + rhs.m_matrix[i];
			return result;
		}
	private:
		T* m_matrix;
		vector<size_t> m_arrDim;
};

template<typename T, size_t first>
class cuMatrix<T,first>{
	public:
		cuMatrix(T* matrix) : m_matrix(matrix),m_arrDim{first} {};
		size_t getNumDim(){return 1;};
		vector<size_t> getDim(){return m_arrDim;};

		T operator[](size_t n){return *(m_matrix+n);}; 

	private:
		T* m_matrix;
		vector<size_t> m_arrDim;
};

int main(void){
	const int data[2][3] = {{1,2,3},{2,3,4}};
	int data2[2][2][3] = 	{{{1,2,3},{4,5,6}}, 
							{{7,8,9},{10,11,12}}};
	int* dataP = (int*)data;
	int* dataP2 = (int*)data2;
	cuMatrix<int,2,3> matrix(dataP);
	cuMatrix<int,3,2> matrix1(dataP);
	cuMatrix<int,2,2,3> matrix2(dataP2);
	cuMatrix<int,2,2,data[0][2]> matrix3(dataP2);


	DEBUG(matrix.getNumDim());

	cout << "getDim(): " << endl;
	for(auto i : matrix.getDim())
		cout << i << " ";
	cout << endl;

	DEBUG(matrix[1].getNumDim());
	DEBUG(matrix2[1].getNumDim());
	DEBUG(matrix[0].getDim()[0]);
	DEBUG(matrix[0][0]);
	DEBUG(matrix[0][1]);
	DEBUG(matrix[0][2]);
	DEBUG(matrix[1][0]);
	DEBUG(matrix[1][1]);
	DEBUG(matrix[1][2]);
	DEBUG(matrix2[1][0][2]);
	DEBUG(matrix2[1][1].getNumDim());;
	DEBUG(data2[1]);
	DEBUG(data2[1][0][2]);
	cout << (matrix+matrix)[1][2] << endl;
	//cout << (matrix+matrix1)[1][2] << endl;
	return 0;
}

