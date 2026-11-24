#include "graph.hpp"
#include "serial_dijkstra.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void export_csv(const Graph& g, const FloodField& ff) {
    std::ofstream f_weights("weights.csv");
    std::ofstream f_dist("flood_dist.csv");
    std::ofstream f_dir("flood_dir.csv");

    for (std::int64_t r = 0; r < ff.n; ++r) {
        for (std::int64_t c = 0; c < ff.n; ++c) {
            const std::int64_t idx = r * ff.n + c;
            const char sep = (c + 1 < ff.n) ? ',' : '\n';

            f_weights << g.weights[idx] << sep;
            if (ff.dist[idx] == FloodField::INF) {
                f_dist << -1 << sep;
            } else {
                f_dist << ff.dist[idx] << sep;
            }
            f_dir << static_cast<int>(ff.dir[idx]) << sep;
        }
    }
    std::cout << "Exported weights.csv, flood_dist.csv, and flood_dir.csv\n";
}

} // namespace

int main(int argc, char* argv[]) {
    // Defaults: 500x500 grid, weights 1..100, full granularity (k=0), 20% obstacles, seed=42
    std::int64_t n = 500;
    std::int64_t w_max = 100;
    int k = 0;
    float obstacle_d = 0.20f;
    std::uint64_t seed = 42;
    bool do_export = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") {
            do_export = true;
        } else if (i == 1) {
            n = std::stoll(arg);
        } else if (i == 2) {
            w_max = std::stoll(arg);
        } else if (i == 3) {
            k = std::stoi(arg);
        } else if (i == 4) {
            obstacle_d = std::stof(arg);
        } else if (i == 5) {
            seed = std::stoull(arg);
        }
    }

    // 1. Generate deterministic terrain map
    Graph g(n, w_max, k, obstacle_d, seed);

    // 2. Run & time serial Dijkstra baseline
    const auto t0 = std::chrono::steady_clock::now();
    FloodField serial_ff = compute_flood_field_serial(g);
    const auto t1 = std::chrono::steady_clock::now();

    const double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "=== Flood Field Benchmark ===\n"
              << "Grid size      : " << n << " x " << n << " (" << g.p.size << " cells)\n"
              << "Target cell    : " << g.p.t << " (row " << (g.p.t / n) << ", col " << (g.p.t % n) << ")\n"
              << "Obstacle ratio : " << g.p.obstacle_d << "\n"
              << "Reachable cells: " << serial_ff.reachable_cells << " / " << g.p.size << "\n"
              << "Serial Dijkstra: " << serial_ms << " ms\n";

    // TODO: Run parallel Delta-Stepping here and verify `parallel_ff.dist == serial_ff.dist`

    if (do_export) {
        export_csv(g, serial_ff);
    }

    return 0;
}