#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <time.h>

#include <boost/algorithm/string.hpp>

#include "src/point/l2metric.h"
#include "src/point/squaredl2metric.h"
#include "src/point/point.h"
#include "src/point/pointweightmodifier.h"
#include "src/clustering/bico.h"
#include "src/misc/randomness.h"
#include "src/misc/randomgenerator.h"
#include "src/datastructure/proxysolution.h"
#include "src/point/pointcentroid.h"
#include "src/point/pointweightmodifier.h"
#include "src/point/realspaceprovider.h"

int main(int argc, char **argv)
{
    using namespace CluE;

    time_t starttime, endtime;
    double difference;

    if (argc < 8)
    {
        std::cout << "Usage: input n k d space output projections [seed]" << std::endl;
        std::cout << "  input       = path to input file" << std::endl;
        std::cout << "  n           = number of input points" << std::endl;
        std::cout << "  k           = number of desired centers" << std::endl;
        std::cout << "  d           = dimension of an input point" << std::endl;
        std::cout << "  space       = coreset size" << std::endl;
        std::cout << "  projections = number of random projections used for nearest neighbour search" << std::endl;
        std::cout << "                in first level" << std::endl;
        std::cout << "  seed        = random seed (optional)" << std::endl;
        std::cout << std::endl;
        std::cout << "7 arguments expected, got " << argc - 1 << ":" << std::endl;
        for (int i = 1; i < argc; ++i)
            std::cout << i << ".: " << argv[i] << std::endl;
        return 1;
    }

    // Read arguments
    std::ifstream filestream(argv[1], std::ifstream::in);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int d = atoi(argv[4]);
    int space = atoi(argv[5]);
    std::ofstream outputstream(argv[6], std::ifstream::out);
    int p = atoi(argv[7]);
    if (argc >= 9)
        Randomness::initialize(atoi(argv[8]));

    time(&starttime);

    // Initialize BICO
    Bico<Point> bico(d, n, k, p, space, new SquaredL2Metric(), new PointWeightModifier());

    int pos = 0;
    while (filestream.good())
    {
        // Read line and construct point
        std::string line;
        std::getline(filestream, line);
        std::vector<std::string> stringcoords;
        boost::split(stringcoords, line, boost::is_any_of(" "));

        std::vector<double> coords;
        coords.reserve(stringcoords.size());
        for (size_t i = 0; i < stringcoords.size(); ++i)
            coords.push_back(atof(stringcoords[i].c_str()));
        Point p(coords);

        if (p.dimension() != d)
        {
            std::clog << "Line skipped because line dimension is " << p.dimension() << " instead of " << d << std::endl;
            continue;
        }

        // Call BICO point update
        bico << p;
    }

    // Retrieve coreset
    ProxySolution<Point>* sol = bico.compute();

    // Output coreset size
    outputstream << sol->proxysets[0].size() << "\n";

    // Output coreset points
    for (size_t i = 0; i < sol->proxysets[0].size(); ++i)
    {
        // Output weight
        outputstream << sol->proxysets[0][i].getWeight() << " ";
        // Output center of gravity
        for (size_t j = 0; j < sol->proxysets[0][i].dimension(); ++j)
        {
            outputstream << sol->proxysets[0][i][j];
            if (j < sol->proxysets[0][i].dimension() - 1)
                outputstream << " ";
        }
        outputstream << "\n";
    }
    outputstream.close();

    return 0;
}
