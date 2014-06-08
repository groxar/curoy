CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35 -l curand
OPTION = -std=c++0x -g


phony: cuMatrixMul run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+

../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixMul: cuMatrixMul.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	./cuMatrixMul

clean:
	rm cuMatrixMul

run:
	@echo "\n\nCUMATRIXMUL"
	./cuMatrixMul
