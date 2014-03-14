CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g

phony: ann.o 

../lib/cuMatrix.o:
	make ../lib/makefile

../lib/cudaMatrix.o:
	make ../lib/makefile

ann.o: ann.cu
	$(NVCC) $(CUDACNFG) -o $@ -c $+

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

