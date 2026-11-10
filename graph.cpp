#include "graph.hpp"
#include "rng.hpp"

Graph::Properties::Properties(std::int64_t _n, std::int64_t _w_max, int _k, float _obstacle_d, std::uint64_t _seed, std::int64_t _w_min) 
: size(_n * _n), w_max(_w_max), k(_k), obstacle_d(_obstacle_d), seed(_seed), w_min(_w_min) {}

Graph::Graph(std::int64_t _n, std::int64_t _w_max, int _k, float _obstacle_d, std::uint64_t _seed, std::int64_t _w_min)
: p(_n, _w_max, _k, _obstacle_d, _seed, _w_min) {
    // minimum weight < 1 makes no sense if 0 is an obstacle
    if (p.w_min <= 0) {
        p.w_min = 1;
    }
    // if granularity is ignored, treat it as infinitely fine
    if (p.k <= 0 || p.k > p.w_max - p.w_min + 1) {
        p.k = p.w_max - p.w_min + 1;
    }
    Rng rng(p.seed); 
    weights.resize(p.size);
    // get destination node
    p.t = rng.bounded(p.size);
    for (std::int64_t i = 0; i < p.size; ++i) {
        bool obstacle = rng.u01() < p.obstacle_d && i != p.t;
        std::int64_t level = rng.bounded(p.k);
        // if there's an obstacle, mask the cell, otherwise compute its weight with granularity as a factor
        weights[i] = obstacle ? 0 : p.k == 1 ? p.w_min : p.w_min + (level * (p.w_max - p.w_min)) / (p.k - 1);
    }
}