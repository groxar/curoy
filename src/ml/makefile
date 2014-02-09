CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
OPTION = -std=c++0x -g

phony: powerANN $(OBJDIR) run

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

../lib/cuMatrix.o:
	make ../lib/makefile

../lib/cudaMatrix.o:
	make ../lib/makefile

powerANN: powerANN.cpp ../lib/cuMatrix.o ../lib/cudaMatrix.o
	$(NVCC) $(CUDACNFG) -Xcompiler '$(OPTION)' -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./powerANN
