#include "master.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <fstream>

// Fast implementation of k-means++ with optional projections

// ./bico/bin/BICO_Quickstart embeddings/07_P0_BBC_Nephrin635_Podocin594_Series007_MIP\(1-6\)__240_480.txt 25617 196 16 10000 out.txt 10 ,
// ./kmeans++/cluster ../bico/out.txt embeddings/07_P0_BBC_Nephrin635_Podocin594_Series007_MIP\(1-6\)__240_480.txt 192 16 25617 clusters_192.out 5

double l2distance(int d, double const * point1, double const * point2)
{
	double sum = 0;
	for (unsigned int i=0; i<d; ++i)
	{
		double delta = point1[i]-point2[i];
		sum += delta*delta;
	}
	return sum;
}

int findClosest(double* point, double*** centers, int k, int d)
{
    int min_i = -1;
    double min_v = 9999;

    for (unsigned int i = 0; i < k; ++i)
    {
        double distance = l2distance(d, point, (*centers)[i]);
        if (distance < min_v)
        {
            min_i = i;
            min_v = distance;
        }
    }
    return min_i;
}


int main(int argc, char **argv)
{
    if (argc < 8)
    {
        std::cout << "Usage: input_coreset input_points k d output iterations [seed] [projections (0 or 1)]" << std::endl;
        std::cout << "5 arguments expected, got " << argc - 1 << ":" << std::endl;
        for (int i = 1; i < argc; ++i)
            std::cout << argv[i] << std::endl;
        return 1;
    }

    int k = atoi(argv[3]);
    int dimension = atoi(argv[4]);
    int nbPoints = atoi(argv[5]);
    std::ofstream output(argv[6], std::ifstream::out);
    int iterations = atoi(argv[7]);
    bool projections = false;

    if (argc >= 9)
    {
        CluE::Randomness::initialize(atoi(argv[8]));
    }
    if (argc >= 10)
    {
        if (1 == atoi(argv[9]))
        {
            projections = true;
        }
    }

    MASTER master = MASTER(argv[1], k, dimension, iterations, projections);
    double***centers = master.run();

    std::FILE*input = std::fopen(argv[2], "r");
    double *point = new double[dimension];
    int pt_i = 0;
    while (pt_i < nbPoints && !std::feof(input))
    {
        for (unsigned int i = 0; i < dimension-1; i++)
            std::fscanf(input, "%lf,", &point[i]);
        std::fscanf(input, "%lf\n", &point[dimension-1]);

        int clusterNb = findClosest(point, centers, k, dimension);
        clusterNb++;
        //std::cout << clusterNb << std::endl;
        //output << pt_i << " " << clusterNb << "\n";
        output << clusterNb << "\n";
        pt_i++;
    }

    output.close();
}


