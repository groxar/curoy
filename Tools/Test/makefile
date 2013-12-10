CC = gcc
CXX = g++
NVCC = nvcc
LIBDIR = lib/
OBJDIR = obj/

phony: test $(OBJDIR) run

$(LIBDIR):
	mkdir -p $@
$(OBJDIR):
	mkdir -p $@

test: test.cu
	$(NVCC) -o $@ $+ 

clean:
	rm -rf $(OBJDIR) $(LIBDIR)	

run:
	./test
