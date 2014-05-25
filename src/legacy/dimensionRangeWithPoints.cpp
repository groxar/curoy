//2D only
		cuMatrix<N> operator()(vector<size_t> startPoint,vector<size_t> endPoint){
			if(startPoint.size() == nDim() == endPoint.size()==2){
				cout << "Dimension error in ()"<< endl;
				return cuMatrix<N>();
			}
			vector<size_t> newDimension;
			size_t size = 1;
			size_t end = nDim();

			for(size_t i = 0; i < end; ++i){
				if(startPoint[i]>endPoint[i]){
					cout << "startPoint > endPoint"<<endl;
					return cuMatrix<N>();
				}
				newDimension.push_back(endPoint[i]-startPoint[i]+1);
				size*=newDimension[i];
			}

			N* tempData;
			cudaMalloc((void**)&tempData,sizeof(N)*size);

			for(int i = startPoint[0]; i <= endPoint[0];++i){
				cudaMemcpy(&(tempData[newDimension[1]*i]),&(m_data[dim(1)*i+startPoint[1]]),sizeof(N)*(endPoint[1]-startPoint[1]+1),cudaMemcpyDeviceToDevice);
			}
			return cuMatrix(tempData,newDimension,memPermission::owner);
		}
