#ifndef SERIAL_DIJKSTRA_H
#define SERIAL_DIJKSTRA_H

#include "graph.hpp"
#include <cstdint>
#include <limits>
#include <vector>

enum class ALGORITHM { DIJKSTRA, DIAL };

struct FloodField {
    static constexpr std::uint64_t INF = std::numeric_limits<std::uint64_t>::max();

    // Direction an agent in cell i should move to reach target p.t:
    // -1 = Target, Obstacle, or Unreachable | 0 = Up | 1 = Right | 2 = Down | 3 = Left
    std::vector<std::uint64_t> dist;
    std::vector<std::int8_t> dir;

    // Populated only by traced solvers: elapsed nanoseconds when cell i was settled (-1 if unset)
    std::vector<std::int64_t> settle_ns;

    std::int64_t n = 0;
    std::int64_t reachable_cells = 0;
};

// Pure serial baseline (zero logging overhead for accurate timing)
FloodField compute_flood_field_serial(const Graph& g, ALGORITHM a);

// Traced serial solver (records per-cell nanosecond settle timestamps for animation)
FloodField compute_flood_field_serial_traced(const Graph& g, ALGORITHM a);

#endif