# A simple Makefile

all: main cluster

main: main.o randomness.o
	g++ -O2 -lm -o kmeans main.o randomness.o

main.o: main.cc master.h kmeans.h triple.h
	g++ -O2 -c -std=gnu++0x main.cc randomness.cpp

cluster: cluster.o randomness.o
	g++ -O2 -lm -o cluster cluster.o randomness.o

cluster.o: cluster.cc master.h kmeans.h triple.h
	g++ -O2 -c -std=gnu++0x cluster.cc randomness.cpp

clean:
	rm -f *.o kmeans
