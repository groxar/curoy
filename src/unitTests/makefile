CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = powerANN
XMATRIX = xMatrixUnit
CUMATRIX = cuMatrixUnit
OPTION = -std=c++0x -g

TARGET = $(CUMATRIX)

phony: $(TARGET) run

$(XMATRIX): $(XMATRIX).cpp
	$(CXX) $(OPTION) -o $@ $+

../lib/cuMatrix.o:
	make ../lib/makefile

../lib/cudaMatrix.o:

$(CUMATRIX): $(CUMATRIX).cpp ../lib/cuMatrix.o ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm $(XMATRIX)

run:
	./$(TARGET)
