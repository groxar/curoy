CC = gcc
CXX = g++
NVCC = nvcc
LIBDIR = lib/
OBJDIR = obj/

phony: test $(OBJDIR)

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

test: test.cu
	$(NVCC) -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	
