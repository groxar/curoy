CC = clang
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g


phony: cuMatrixUnit xMatrixUnit run

xMatrixUnit: xMatrixUnit.cpp
	$(CXX) $(OPTION) -o $@ $+


../lib/cudaMatrix.o:
	make ../lib/makefile

cuMatrixUnit: cuMatrixUnit.cpp  ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

annUnit: annUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nANNUNIT"
	./annUnit

powerANN: powerANN.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nANN"
	./powerANN

gradientUnit: gradientUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nGradient"
	./gradientUnit

featureNormalizeUnit: featureNormalizeUnit.cpp ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 
	@echo "\n\nFeatureNormalize"
	./featureNormalizeUnit

clean:
	rm xMatrixUnit cuMatrixUnit

run:
	@echo "\n\nCUMATRIX"
	./cuMatrixUnit 
	@echo "\n\nXMATRIX"
	./xMatrixUnit
