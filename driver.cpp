#include "graph.hpp"
#include "serial_dijkstra.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string print_algorithm(ALGORITHM a) {
    return a == ALGORITHM::DIJKSTRA ? "Dijkstra" : "Dial";
}

std::string format_duration(double ns) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (ns < 1'000.0) {
        oss << ns << " ns";
    } else if (ns < 1'000'000.0) {
        oss << (ns / 1'000.0) << " us";
    } else if (ns < 1'000'000'000.0) {
        oss << (ns / 1'000'000.0) << " ms";
    } else {
        oss << (ns / 1'000'000'000.0) << " s";
    }
    return oss.str();
}

void export_csv(const Graph& g, const FloodField& ff, bool include_trace) {
    std::ofstream f_weights("weights.csv");
    std::ofstream f_dist("flood_dist.csv");
    std::ofstream f_dir("flood_dir.csv");
    std::ofstream f_trace;
    if (include_trace) {
        f_trace.open("trace_serial.csv");
    }

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

            if (include_trace) {
                f_trace << ff.settle_ns[idx] << sep;
            }
        }
    }

    if (include_trace) {
        std::cout << "Exported weights.csv, flood_dist.csv, flood_dir.csv, and trace_serial.csv\n";
    } else {
        std::cout << "Exported weights.csv, flood_dist.csv, and flood_dir.csv\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::int64_t n = 500;
    std::int64_t w_max = 100;
    int k = 0;
    float obstacle_d = 0.20f;
    std::uint64_t seed = 42;
    bool do_export = false;
    bool do_trace = false;
    ALGORITHM a = ALGORITHM::DIJKSTRA;

    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") {
            do_export = true;
        } else if (arg == "--trace") {
            do_trace = true;
            do_export = true;
        } else if (arg == "--dijkstra") {
            a = ALGORITHM::DIJKSTRA;
        } else if (arg == "--dial") {
            a = ALGORITHM::DIAL;
        }
        else {
            pos_args.push_back(arg);
        }
    }

    if (pos_args.size() >= 1) n = std::stoll(pos_args[0]);
    if (pos_args.size() >= 2) w_max = std::stoll(pos_args[1]);
    if (pos_args.size() >= 3) k = std::stoi(pos_args[2]);
    if (pos_args.size() >= 4) obstacle_d = std::stof(pos_args[3]);
    if (pos_args.size() >= 5) seed = std::stoull(pos_args[4]);

    // 1. Generate deterministic terrain map
    Graph g(n, w_max, k, obstacle_d, seed);

    // 2. Run & time pure serial Dijkstra baseline
    const auto t0 = std::chrono::steady_clock::now();
    FloodField serial_ff = compute_flood_field_serial(g, a);
    const auto t1 = std::chrono::steady_clock::now();

    const double serial_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();

    std::cout << "=== Flood Field Benchmark ===\n"
              << "Algorithm      : " << print_algorithm(a) << '\n'
              << "Grid size      : " << n << " x " << n << " (" << g.p.size << " cells)\n"
              << "Target cell    : " << g.p.t << " (row " << (g.p.t / n) << ", col " << (g.p.t % n) << ")\n"
              << "Obstacle ratio : " << g.p.obstacle_d << "\n"
              << "Reachable cells: " << serial_ff.reachable_cells << " / " << g.p.size << "\n"
              << "Duration: " << format_duration(serial_ns) << "\n";

    // TODO: Run parallel Delta-Stepping here and verify `parallel_ff.dist == serial_ff.dist`

    if (do_trace) {
        FloodField traced_ff = compute_flood_field_serial_traced(g, a);
        export_csv(g, traced_ff, true);
    } else if (do_export) {
        export_csv(g, serial_ff, false);
    }

    return 0;
}