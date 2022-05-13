#include "master.h"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <fstream>

// Fast implementation of k-means++ with optional projections

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cout << "Usage: input k d output iterations [seed] [projections (0 or 1)]" << std::endl;
        std::cout << "5 arguments expected, got " << argc - 1 << ":" << std::endl;
        for (int i = 1; i < argc; ++i)
            std::cout << argv[i] << std::endl;
        return 1;
    }

    int k = atoi(argv[2]);
    int dimension = atoi(argv[3]);
    std::ofstream output(argv[4], std::ifstream::out);
    int iterations = atoi(argv[5]);
    bool projections = false;

    std::cout << "You are clustering " << argv[1] << " consisting of " << dimension << " dimensions using " << k << " centers." << std::endl;

    if (iterations > 0)
    {
        std::cout << "Computation is repeated 5 times until convergence or " << iterations << " many iterations" << std::endl;
    }
    if (argc >= 7)
    {
        CluE::Randomness::initialize(atoi(argv[6]));
        std::cout << "The random seed is " << atoi(argv[6]) << std::endl;
    }
    if (argc >= 8)
    {
        if (1 == atoi(argv[7]))
        {
            projections = true;
            std::cout << "Nearest center search is sped up using projections." << std::endl;
        }
    }

    MASTER master = MASTER(argv[1], k, dimension, iterations, projections);
    double***centers = master.run();

    /*
    std::cout << "centers" <<std::endl;
    for(int i = 0; i < k; i++) {
            for(int j = 0; j < dimension; j++) {
                    std::cout<< (*centers)[i][j]<<" ";
            }
            std::cout<< std::endl;
    }
    */

    std::cout << "Outputting final centers" << std::endl;

    for (int i = 0; i < k; i++)
    {
        output << "1 ";
        for (int j = 0; j < dimension; j++)
        {
            output << (*centers)[i][j];
            if (j < dimension - 1)
            {
                output << " ";
            }
        }
        output << "\n";
    }
    output.close();
}



