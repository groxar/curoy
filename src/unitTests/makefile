CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = powerANN
XMATRIXU = xMatrixUnit
OPTION = -std=c++0x -g

phony: $(XMATRIXU) run

$(XMATRIXU): $(XMATRIXU).cpp
	$(CXX) $(OPTION) -o $@ $+

clean:
	rm -rf $(XMATRIXU)

run:
	./$(XMATRIXU)
