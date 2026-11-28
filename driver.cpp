#include "graph.hpp"
#include "metrics.hpp"
#include "parallel_dijkstra.hpp"
#include "serial_dijkstra.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace fs = std::filesystem;

namespace {

enum class SOLVER_MODE { DIJKSTRA, DIAL, OMP, MPI_SOLVER };

std::string mode_name(SOLVER_MODE m) {
    switch (m) {
        case SOLVER_MODE::DIJKSTRA:   return "Serial Dijkstra";
        case SOLVER_MODE::DIAL:       return "Serial Dial";
        case SOLVER_MODE::OMP:        return "OpenMP Delta-Stepping";
        case SOLVER_MODE::MPI_SOLVER: return "MPI Delta-Stepping";
    }
    return "Unknown";
}

std::string trace_filename(SOLVER_MODE m, std::uint64_t delta, int workers) {
    std::ostringstream oss;
    switch (m) {
        case SOLVER_MODE::DIJKSTRA:
            oss << "trace_dijkstra.csv";
            break;
        case SOLVER_MODE::DIAL:
            oss << "trace_dial.csv";
            break;
        case SOLVER_MODE::OMP:
            oss << "trace_omp_d" << delta << "_t" << workers << ".csv";
            break;
        case SOLVER_MODE::MPI_SOLVER:
            oss << "trace_mpi_d" << delta << "_r" << workers << ".csv";
            break;
    }
    return oss.str();
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

void export_csv(const Graph& g,
                const FloodField& ff,
                const std::string& trace_file,
                bool include_trace,
                const fs::path& out_dir) {
    fs::create_directories(out_dir);

    const fs::path weights_path = out_dir / "weights.csv";
    const fs::path dist_path    = out_dir / "flood_dist.csv";
    const fs::path dir_path     = out_dir / "flood_dir.csv";
    const fs::path trace_path   = out_dir / trace_file;

    std::ofstream f_weights(weights_path);
    std::ofstream f_dist(dist_path);
    std::ofstream f_dir(dir_path);
    std::ofstream f_trace;
    if (include_trace) {
        f_trace.open(trace_path);
    }

    for (std::int64_t r = 0; r < ff.n; ++r) {
        for (std::int64_t c = 0; c < ff.n; ++c) {
            const std::int64_t idx = r * ff.n + c;
            const char sep = (c + 1 < ff.n) ? ',' : '\n';

            f_weights << g.weights[idx] << sep;
            const std::uint32_t d = ff.dist_at(idx);
            if (d == FloodField::INF_DIST) {
                f_dist << -1 << sep;
            } else {
                f_dist << d << sep;
            }
            f_dir << static_cast<int>(ff.dir_at(idx)) << sep;

            if (include_trace) {
                f_trace << ff.settle_ns[idx] << sep;
            }
        }
    }

    if (include_trace) {
        std::cout << "Exported " << weights_path.string() << ", "
                  << dist_path.string() << ", "
                  << dir_path.string() << ", and "
                  << trace_path.string() << '\n';
    } else {
        std::cout << "Exported " << weights_path.string() << ", "
                  << dist_path.string() << ", and "
                  << dir_path.string() << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    int rank = 0;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::int64_t n = 500;
    std::int64_t w_max = 100;
    int k = 0;
    float obstacle_d = 0.20f;
    std::uint64_t seed = 42;
    bool do_export = false;
    bool do_trace = false;
    bool skip_verify = false;
    SOLVER_MODE mode = SOLVER_MODE::OMP;
    std::uint64_t delta = 0;
    int num_threads = 0;
    fs::path out_dir = "output";

    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export") {
            do_export = true;
        } else if (arg == "--trace") {
            do_trace = true;
            do_export = true;
        } else if (arg == "--no-verify") {
            skip_verify = true;
        } else if (arg == "--dijkstra") {
            mode = SOLVER_MODE::DIJKSTRA;
        } else if (arg == "--dial") {
            mode = SOLVER_MODE::DIAL;
        } else if (arg == "--omp") {
            mode = SOLVER_MODE::OMP;
        } else if (arg == "--mpi") {
            mode = SOLVER_MODE::MPI_SOLVER;
        } else if (arg.rfind("--delta=", 0) == 0) {
            delta = std::stoull(arg.substr(8));
        } else if (arg.rfind("--threads=", 0) == 0) {
            num_threads = std::stoi(arg.substr(10));
        } else if (arg.rfind("--out-dir=", 0) == 0) {
            out_dir = arg.substr(10);
        } else {
            pos_args.push_back(arg);
        }
    }

    if (pos_args.size() >= 1) n = std::stoll(pos_args[0]);
    if (pos_args.size() >= 2) w_max = std::stoll(pos_args[1]);
    if (pos_args.size() >= 3) k = std::stoi(pos_args[2]);
    if (pos_args.size() >= 4) obstacle_d = std::stof(pos_args[3]);
    if (pos_args.size() >= 5) seed = std::stoull(pos_args[4]);

    // 1. Generate deterministic terrain map (identical across all MPI ranks via seed)
    Graph g(n, w_max, k, obstacle_d, seed);

    if (mode == SOLVER_MODE::DIJKSTRA || mode == SOLVER_MODE::DIAL) {
        if (rank == 0) {
            ALGORITHM a = (mode == SOLVER_MODE::DIJKSTRA) ? ALGORITHM::DIJKSTRA : ALGORITHM::DIAL;

            const ResourceSnapshot cpu_start = capture_cpu_snapshot();
            const auto t0 = std::chrono::steady_clock::now();
            FloodField serial_ff = compute_flood_field_serial(g, a);
            const auto t1 = std::chrono::steady_clock::now();
            const double serial_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            const ResourceStats rstats = finalize_resource_stats(cpu_start, serial_ns);

            bool ok = true;
            if (!skip_verify) {
                ok = verify_bellman_optimality(g, serial_ff);
            }

            std::cout << "=== Flood Field Benchmark ===\n"
                      << "Algorithm      : " << mode_name(mode) << '\n'
                      << "Grid size      : " << n << " x " << n << " (" << g.p.size << " cells)\n"
                      << "Target cell    : " << g.p.t << " (row " << (g.p.t / n) << ", col " << (g.p.t % n) << ")\n"
                      << "Obstacle ratio : " << g.p.obstacle_d << "\n"
                      << "Reachable cells: " << serial_ff.reachable_cells << " / " << g.p.size << "\n"
                      << "Duration       : " << format_duration(serial_ns) << "\n"
                      << "Peak RSS       : " << std::fixed << std::setprecision(2) << rstats.peak_rss_mb() << " MB\n"
                      << "CPU Load       : " << rstats.effective_cores() << " eff. cores ("
                      << rstats.cpu_efficiency_pct(1) << "% of 1)\n";

            if (!skip_verify) {
                std::cout << "Verification   : "
                          << (ok ? "PASS (Bellman optimality + flow verified)" : "FAIL (Bellman optimality violated)")
                          << "\n";
            }

            if (do_trace) {
                FloodField traced_ff = compute_flood_field_serial_traced(g, a);
                export_csv(g, traced_ff, trace_filename(mode, 0, 1), true, out_dir);
            } else if (do_export) {
                export_csv(g, serial_ff, trace_filename(mode, 0, 1), false, out_dir);
            }
        }
    } else {
        ParallelConfig cfg;
        cfg.backend = (mode == SOLVER_MODE::OMP) ? PARALLEL_BACKEND::OPENMP : PARALLEL_BACKEND::MPI;
        cfg.delta = delta;
        cfg.num_threads = num_threads;

        const ResourceSnapshot cpu_start = capture_cpu_snapshot();
        ParallelResult par_res = compute_flood_field_parallel(g, cfg);
        ResourceStats rstats = finalize_resource_stats(cpu_start, par_res.metrics.elapsed_ns);

        std::uint64_t max_rss_bytes = rstats.peak_rss_bytes;
        std::uint64_t total_rss_bytes = rstats.peak_rss_bytes;
#ifdef USE_MPI
        if (mode == SOLVER_MODE::MPI_SOLVER) {
            std::uint64_t local_rss = rstats.peak_rss_bytes;
            MPI_Reduce(&local_rss, &max_rss_bytes, 1, MPI_UINT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_rss, &total_rss_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        }
#endif

        if (rank == 0) {
            bool ok = true;
            if (!skip_verify) {
                ok = verify_bellman_optimality(g, par_res.ff);
                par_res.metrics.verified = ok;
            }

            std::cout << "=== Flood Field Benchmark ===\n"
                      << "Algorithm      : " << par_res.metrics.backend_name << '\n'
                      << "Workers        : " << par_res.metrics.workers << '\n'
                      << "Delta (bucket) : " << par_res.metrics.delta_used << '\n'
                      << "Grid size      : " << n << " x " << n << " (" << g.p.size << " cells)\n"
                      << "Target cell    : " << g.p.t << " (row " << (g.p.t / n) << ", col " << (g.p.t % n) << ")\n"
                      << "Obstacle ratio : " << g.p.obstacle_d << "\n"
                      << "Reachable cells: " << par_res.ff.reachable_cells << " / " << g.p.size << "\n"
                      << "Bucket steps   : " << par_res.metrics.bucket_steps
                      << " (light phases: " << par_res.metrics.light_phases << ")\n"
                      << "Duration       : " << format_duration(par_res.metrics.elapsed_ns) << "\n";

            std::cout << std::fixed << std::setprecision(2);
            if (mode == SOLVER_MODE::MPI_SOLVER) {
                std::cout << "Peak RSS       : " << (static_cast<double>(max_rss_bytes) / (1024.0 * 1024.0))
                          << " MB max/rank (" << (static_cast<double>(total_rss_bytes) / (1024.0 * 1024.0))
                          << " MB total)\n";
            } else {
                std::cout << "Peak RSS       : " << rstats.peak_rss_mb() << " MB\n"
                          << "CPU Load       : " << rstats.effective_cores() << " eff. cores ("
                          << rstats.cpu_efficiency_pct(par_res.metrics.workers) << "% of "
                          << par_res.metrics.workers << ")\n";
            }

            if (!skip_verify) {
                if (ok) {
                    std::cout << "Verification   : PASS (Bellman optimality + flow verified)\n";
                } else {
                    std::cerr << "Verification   : FAIL (Bellman optimality violated)!\n";
                }
            }
        }

        if (do_trace) {
            ParallelResult traced_res = compute_flood_field_parallel_traced(g, cfg);
            if (rank == 0) {
                const std::string tfile = trace_filename(mode, traced_res.metrics.delta_used, traced_res.metrics.workers);
                export_csv(g, traced_res.ff, tfile, true, out_dir);
            }
        } else if (do_export && rank == 0) {
            const std::string tfile = trace_filename(mode, par_res.metrics.delta_used, par_res.metrics.workers);
            export_csv(g, par_res.ff, tfile, false, out_dir);
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}