all: clean pgm.o HoughBase HoughConstant HoughShared

HoughBase:	HoughBase.cu pgm.o
	nvcc -o HoughBase pgm.cpp HoughBase.cu 

HoughConstant:	HoughConstant.cu pgm.o
	nvcc -o HoughConstant pgm.cpp HoughConstant.cu

HoughShared:	HoughShared.cu pgm.o
	nvcc -o HoughShared pgm.cpp HoughShared.cu

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o

clean:
	rm -f HoughBase HoughConstant HoughShared