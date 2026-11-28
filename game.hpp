#include "graph.hpp"
#include "parallel_dijkstra.hpp"
#include "serial_dijkstra.hpp"
#include <cstdint>
#include <string>

#ifndef GAME_H
#define GAME_H

struct GameConfig {
    bool use_parallel = false;
    ALGORITHM serial_algo = ALGORITHM::DIJKSTRA;
    ParallelConfig parallel;
    std::string algo_label;
    int max_zombies = 500;
    int num_portals = 8;
    int spawns_per_tick = 1;
    int tick_hz = 10;
    int fps_cap = 120; // 0 = uncapped (burns a core, competes with the worker)
    std::uint64_t seed = 42;
};

// Takes g by non-const reference: portals are written into the weights before play.
int run_game(Graph& g, const GameConfig& cfg);

#endif