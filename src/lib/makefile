CC = gcc
CXX = clang++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/
TARGET = powerANN
CMATRIX = cMatrix
OPTION = -std=c++0x -g

phony: $(TARGET) $(OBJDIR) run

$(CMATRIX): $(CMATRIX).cpp
	$(CXX) $(OPTION) -o $@ $+

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

$(TARGET): $(TARGET).cu
	$(NVCC) $(CUDACNFG) -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./$(CMATRIX)
