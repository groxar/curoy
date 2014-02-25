CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x


phony: cuMatrixUnit xMatrixUnit powerANN run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+


../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixUnit: cuMatrixUnit.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

powerANN: powerANN.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	echo "\n\nCUMATRIX"
	./cuMatrixUnit 
	echo "\n\nXMATRIX"
	./xMatrixUnit
	echo "\n\nANN"
	./powerANN
