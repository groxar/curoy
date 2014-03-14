CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g


phony: cuMatrixMul run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+

../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixMul: cuMatrixMul.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm cuMatrixMul

run:
	@echo "\n\nCUMATRIXMUL"
	./cuMatrixMul
