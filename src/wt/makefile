OPTION = -g -std=c++0x

phony: example

Filter.o: Filter.cpp
	clang++ $(OPTION) -o $@ -c $+

SymetricPadding.o: SymetricPadding.cpp
	clang++ $(OPTION) -o $@ -c $+

WaveletTransformator.o: WaveletTransformator.cpp Filter.o SymetricPadding.o
	clang++ $(OPTION) -o $@ -c $+

example: example.cpp WaveletTransformator.o Filter.o SymetricPadding.o
	clang++ $(OPTION) -o $@ $+
