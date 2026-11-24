#ifndef SERIAL_DIJKSTRA_H
#define SERIAL_DIJKSTRA_H

#include "graph.hpp"
#include <cstdint>
#include <limits>
#include <vector>

struct FloodField {
    static constexpr std::uint64_t INF = std::numeric_limits<std::uint64_t>::max();

    // Direction an agent in cell i should move to reach target p.t:
    // -1 = Target, Obstacle, or Unreachable | 0 = Up | 1 = Right | 2 = Down | 3 = Left
    std::vector<std::uint64_t> dist;
    std::vector<std::int8_t> dir;
    std::int64_t n = 0;
    std::int64_t reachable_cells = 0;
};

FloodField compute_flood_field_serial(const Graph& g);

#endif