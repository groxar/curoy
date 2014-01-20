#include <vector>
#include <cstdint>
#include <iostream>

using namespace std;



template<typename T, size_t first, size_t... N>
class cuMatrix{
	public:
		cuMatrix(T* matrix) :m_numDim(sizeof...(N)+1), m_matrix(matrix),m_arrDim{first,N...} {};
		size_t getNumDim(){return m_numDim;};
		vector<size_t> getDim(){return m_arrDim;};

		cuMatrix<T,N...> operator[](size_t n){return cuMatrix<T,N...>(m_matrix+m_arrDim[sizeof...(N)]*n);}; 

	private:
		T* m_matrix;
		size_t m_numDim;
		vector<size_t> m_arrDim;
};

template<typename T, size_t first>
class cuMatrix<T,first>{
	public:
		cuMatrix(T* matrix) :m_numDim(1), m_matrix(matrix),m_arrDim{first} {};
		size_t getNumDim(){return m_numDim;};
		vector<size_t> getDim(){return m_arrDim;};

		T operator[](size_t n){return *(m_matrix+n);}; 

	private:
		T* m_matrix;
		size_t m_numDim;
		vector<size_t> m_arrDim;
};

int main(void){
	int data[2][3] = {{1,2,3},{2,3,4}};
	int* dataP = (int*)data;
	cuMatrix<int,2,3> matrix(dataP);

	cout <<"getNumDim(): "<< matrix.getNumDim()<< endl;

	cout << "getDim(): " << endl;
	for(auto i : matrix.getDim())
		cout << i << endl;

	cout << "[1].getNumDim(): "<< matrix[1].getNumDim()<< endl;
	cout << "[0].getDim()[0]: "<< matrix[0].getDim()[0]<< endl;
	cout << "[0][0]: "<< matrix[0][0]<< endl;
	cout << "[0][1]: "<< matrix[0][1]<< endl;
	cout << "[0][2]: "<< matrix[0][2]<< endl;
	cout << "[1][0]: "<< matrix[1][0]<< endl;
	cout << "[1][1]: "<< matrix[1][1]<< endl;
	cout << "[1][2]: "<< matrix[1][2]<< endl;
	return 0;
}

