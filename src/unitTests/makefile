CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x


phony: cuMatrixUnit xMatrixUnit run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+

../lib/cuMatrix.o:
	make ../lib/makefile

../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixUnit: cuMatrixUnit.cpp ../lib/cuMatrix.o ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	./cuMatrixUnit 
	./xMatrixUnit
