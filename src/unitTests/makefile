CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x


phony: cuMatrixUnit xMatrixUnit featureNormalize run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+


../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixUnit: cuMatrixUnit.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

powerANN: powerANN.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

gradientUnit: gradientUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

featureNormalize: featureNormalizeUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	@echo "\n\nCUMATRIX"
	./cuMatrixUnit 
	@echo "\n\nXMATRIX"
	./xMatrixUnit
	@echo "\n\nNormalized"
	./featureNormalize
