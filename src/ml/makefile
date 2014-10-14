CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_20 -lcurand -ccbin=/usr/bin/g++
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g

phony: ann.o 

../lib/cuMatrix.o:
	make ../lib/makefile


libann.so: ann.cu
	$(NVCC) $(CUDACNFG) -o $@ -Xcompiler '-fPIC' --shared $+

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

