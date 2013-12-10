CC = gcc
CXX = g++
NVCC = nvcc
CUDACNFG= -arch=sm_35
LIBDIR = lib/
OBJDIR = obj/

phony: test $(OBJDIR) run

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

test: test.cu
	$(NVCC) $(CUDACNFG) -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./test
