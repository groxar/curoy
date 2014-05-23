OPTION = -g -std=c++0x

phony: example

Filter.o: Filter.cpp
	clang++ $(OPTION) -o $@ -c $+

WaveletTransformator.o: WaveletTransformator.cpp Filter.o
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp WaveletTransformator.o Filter.o
	clang++ $(OPTION) -o $@ $+
