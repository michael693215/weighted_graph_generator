// Weighted graph generator class for flat 2D planes with variable obstacle density and cell entry cost
#include <cstdint>
#include <vector>

#ifndef GRAPH_H
#define GRAPH_H

struct Graph {
    // book-keeping class for cold graph data
    struct Properties {
        Properties(std::int64_t _n, std::int64_t _w_max, int _k, float _obstacle_d, std::uint64_t seed, std::int64_t _min_w);
        std::int64_t size;
        std::int64_t w_max;
        int k;
        float obstacle_d;
        std::uint64_t seed;
        std::int64_t w_min; 
        std::int64_t t; // target node
    };
    Properties p;

    // n: square root of the size of the grid.
    // w_min, w_max: minimum and maximum weight respectively. minimum of 1 allows 0 to be used for obstacles.
    // k: number of different weights possible in the graph. setting to 0 ignores this quality.
    // obstacle_d: obstacle density; ratio of 0s in a graph.
    // seed: seed for the prng (pseudo-random number generator). helpful for generating the same graphs over different systems.
    Graph(std::int64_t _n, std::int64_t _w_max, int _k = 0, float _obstacle_d = 0, std::uint64_t _seed = 1, std::int64_t _w_min = 1);
    std::vector<std::uint32_t> weights;
};
#endif