#ifndef PARALLEL_DIJKSTRA_H
#define PARALLEL_DIJKSTRA_H

#include "graph.hpp"
#include "serial_dijkstra.hpp"
#include <cstdint>
#include <string>

enum class PARALLEL_BACKEND { OPENMP, MPI };

struct ParallelConfig {
    PARALLEL_BACKEND backend = PARALLEL_BACKEND::OPENMP;
    std::uint64_t delta = 0;   // 0 = auto-select based on graph weights
    int num_threads = 0;       // 0 = OpenMP default (omp_get_max_threads())
};

struct ParallelMetrics {
    std::string backend_name;
    std::uint64_t delta_used = 0;
    int workers = 1;           // OpenMP threads or MPI ranks
    double elapsed_ns = 0.0;
    std::uint64_t bucket_steps = 0;
    std::uint64_t light_phases = 0;
    std::int64_t reachable_cells = 0;
    bool verified = false;
};

struct ParallelResult {
    FloodField ff;
    ParallelMetrics metrics;
};

// Pure parallel solver (zero tracing overhead for accurate benchmarking)
ParallelResult compute_flood_field_parallel(const Graph& g, const ParallelConfig& cfg);

// Traced parallel solver (records per-cell elapsed nanoseconds for animation)
ParallelResult compute_flood_field_parallel_traced(const Graph& g, const ParallelConfig& cfg);

#endif