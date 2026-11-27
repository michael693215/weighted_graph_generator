#include "parallel_dijkstra.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace {

constexpr std::int64_t DR[4] = {-1, 0, 1, 0};
constexpr std::int64_t DC[4] = {0, 1, 0, -1};

std::uint64_t resolve_delta(const Graph& g, std::uint64_t requested_delta) {
    if (requested_delta > 0) {
        return requested_delta;
    }
    // For 2D grids, ~1.5x max_w gives wide enough wavefronts to amortize barriers
    std::uint64_t max_w = 1;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        if (g.weights[i] > max_w) {
            max_w = g.weights[i];
        }
    }
    return std::max<std::uint64_t>(1, (max_w * 3) / 2);
}

// Lock-free atomic min directly on raw uint64_t array (zero std::atomic init overhead)
inline bool atomic_min_u64(std::uint64_t* addr, std::uint64_t new_val) {
    std::uint64_t prev = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (new_val < prev) {
        if (__atomic_compare_exchange_n(addr, &prev, new_val, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            return true;
        }
    }
    return false;
}

// Compute flow directions after distances converge to avoid torn (dist, dir) writes
void compute_directions_omp(const Graph& g, FloodField& ff, int num_threads) {
    const std::int64_t n = ff.n;
    const std::int64_t target = g.p.t;
    ff.dir.assign(g.p.size, -1);

    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return;
    }

    std::int64_t reachable = 0;

    #pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:reachable)
    for (std::int64_t idx = 0; idx < g.p.size; ++idx) {
        if (g.weights[idx] == 0 || ff.dist[idx] == FloodField::INF) {
            continue;
        }
        ++reachable;
        if (idx == target) {
            ff.dir[idx] = -1;
            continue;
        }

        const std::int64_t r = idx / n;
        const std::int64_t c = idx % n;
        std::uint64_t best_cost = FloodField::INF;
        std::int8_t best_dir = -1;

        for (int i = 0; i < 4; ++i) {
            const std::int64_t nr = r + DR[i];
            const std::int64_t nc = c + DC[i];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                continue;
            }
            const std::int64_t nb = nr * n + nc;
            if (g.weights[nb] == 0 || ff.dist[nb] == FloodField::INF) {
                continue;
            }
            const std::uint64_t cand = ff.dist[nb] + static_cast<std::uint64_t>(g.weights[nb]);
            if (cand < best_cost) {
                best_cost = cand;
                best_dir = static_cast<std::int8_t>(i);
            }
        }
        ff.dir[idx] = best_dir;
    }

    ff.reachable_cells = reachable;
}

struct QueueEntry {
    std::uint64_t dist;
    std::int64_t vertex;
};

template <bool RecordTrace>
ParallelResult run_openmp_delta_stepping(const Graph& g, const ParallelConfig& cfg) {
    ParallelResult res;
    res.metrics.backend_name = "OpenMP Delta-Stepping";

    if (g.p.size <= 0) {
        return res;
    }

    const std::int64_t n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    res.ff.n = n;
    res.ff.dist.assign(g.p.size, FloodField::INF);
    res.ff.dir.assign(g.p.size, -1);
    if constexpr (RecordTrace) {
        res.ff.settle_ns.assign(g.p.size, -1);
    }

    const std::int64_t target = g.p.t;
    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return res;
    }

    int threads = cfg.num_threads;
#ifdef _OPENMP
    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
#else
    threads = 1;
#endif
    res.metrics.workers = threads;

    const std::uint64_t delta = resolve_delta(g, cfg.delta);
    res.metrics.delta_used = delta;

    std::uint64_t max_w = 1;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        if (g.weights[i] > max_w) {
            max_w = g.weights[i];
        }
    }

    const std::size_t num_buckets = static_cast<std::size_t>((max_w / delta) + 3);

    // Thread-local buckets store (tentative_dist, vertex) pairs so stale entries are skipped safely
    std::vector<std::vector<std::vector<QueueEntry>>> local_buckets(
        threads, std::vector<std::vector<QueueEntry>>(num_buckets));

    // Per-thread deferred heavy vertices for the current bucket
    std::vector<std::vector<QueueEntry>> local_heavy(threads);

    std::vector<QueueEntry> shared_frontier;
    std::vector<std::size_t> thread_offsets(threads + 1, 0);

    // Shared loop control state across the single persistent parallel region
    std::uint64_t current_bucket = 0;
    std::size_t active_slot = 0;
    std::size_t total_items = 0;
    bool done = false;
    bool in_heavy_phase = false;
    std::uint64_t bucket_steps = 0;
    std::uint64_t light_phases = 0;

    const auto t0 = std::chrono::steady_clock::now();

    std::uint64_t* dist_ptr = res.ff.dist.data();
    dist_ptr[target] = 0;
    local_buckets[0][0].push_back({0, target});

    // Single persistent parallel region eliminates thousands of thread fork/joins
    #pragma omp parallel num_threads(threads)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        while (true) {
            // Thread 0 selects the next non-empty bucket or next light/heavy phase
            #pragma omp single
            {
                in_heavy_phase = false;

                // Check if current bucket still has light items to drain
                std::size_t slot = static_cast<std::size_t>(current_bucket % num_buckets);
                thread_offsets[0] = 0;
                for (int t = 0; t < threads; ++t) {
                    thread_offsets[t + 1] = thread_offsets[t] + local_buckets[t][slot].size();
                }
                total_items = thread_offsets[threads];

                if (total_items > 0) {
                    active_slot = slot;
                    ++light_phases;
                    if (shared_frontier.size() < total_items) {
                        shared_frontier.resize(total_items);
                    }
                } else {
                    // Current bucket's light phase is drained; check if heavy vertices were deferred
                    thread_offsets[0] = 0;
                    for (int t = 0; t < threads; ++t) {
                        thread_offsets[t + 1] = thread_offsets[t] + local_heavy[t].size();
                    }
                    total_items = thread_offsets[threads];

                    if (total_items > 0) {
                        in_heavy_phase = true;
                        if (shared_frontier.size() < total_items) {
                            shared_frontier.resize(total_items);
                        }
                    } else {
                        // Advance to the next non-empty bucket
                        bool found = false;
                        for (std::size_t offset = 1; offset <= num_buckets; ++offset) {
                            const std::uint64_t cand_b = current_bucket + offset;
                            const std::size_t cand_slot = static_cast<std::size_t>(cand_b % num_buckets);
                            thread_offsets[0] = 0;
                            for (int t = 0; t < threads; ++t) {
                                thread_offsets[t + 1] = thread_offsets[t] + local_buckets[t][cand_slot].size();
                            }
                            if (thread_offsets[threads] > 0) {
                                current_bucket = cand_b;
                                active_slot = cand_slot;
                                total_items = thread_offsets[threads];
                                ++bucket_steps;
                                ++light_phases;
                                if (shared_frontier.size() < total_items) {
                                    shared_frontier.resize(total_items);
                                }
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            done = true;
                        }
                    }
                }
            } // implicit barrier after omp single

            if (done) {
                break;
            }

            // Flatten thread-local queues into shared_frontier in parallel
            if (!in_heavy_phase) {
                auto& my_vec = local_buckets[tid][active_slot];
                if (!my_vec.empty()) {
                    std::memcpy(shared_frontier.data() + thread_offsets[tid],
                                my_vec.data(),
                                my_vec.size() * sizeof(QueueEntry));
                    my_vec.clear();
                }
            } else {
                auto& my_heavy = local_heavy[tid];
                if (!my_heavy.empty()) {
                    std::memcpy(shared_frontier.data() + thread_offsets[tid],
                                my_heavy.data(),
                                my_heavy.size() * sizeof(QueueEntry));
                    my_heavy.clear();
                }
            }

            #pragma omp barrier

            std::int64_t phase_ns = 0;
            if constexpr (RecordTrace) {
                if (tid == 0) {
                    const auto now = std::chrono::steady_clock::now();
                    phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                }
            }

            // Relax frontier in parallel
            #pragma omp for schedule(static)
            for (std::size_t i = 0; i < total_items; ++i) {
                const QueueEntry entry = shared_frontier[i];
                const std::int64_t u = entry.vertex;
                const std::uint64_t d_u = __atomic_load_n(&dist_ptr[u], __ATOMIC_RELAXED);

                // Skip stale queue entries
                if (entry.dist > d_u) {
                    continue;
                }

                if constexpr (RecordTrace) {
                    res.ff.settle_ns[u] = phase_ns;
                }

                const std::uint64_t step_cost = static_cast<std::uint64_t>(g.weights[u]);

                // During light phase, defer heavy vertices to the bucket's heavy phase
                if (!in_heavy_phase && step_cost > delta) {
                    local_heavy[tid].push_back({d_u, u});
                    continue;
                }

                const std::int64_t r = u / n;
                const std::int64_t c = u % n;

                for (int dir_i = 0; dir_i < 4; ++dir_i) {
                    const std::int64_t nr = r + DR[dir_i];
                    const std::int64_t nc = c + DC[dir_i];
                    if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                        continue;
                    }
                    const std::int64_t v = nr * n + nc;
                    if (g.weights[v] == 0) {
                        continue;
                    }

                    const std::uint64_t new_dist = d_u + step_cost;
                    if (atomic_min_u64(&dist_ptr[v], new_dist)) {
                        const std::uint64_t target_b = new_dist / delta;
                        local_buckets[tid][target_b % num_buckets].push_back({new_dist, v});
                    }
                }
            } // implicit barrier after omp for
        }
    }

    compute_directions_omp(g, res.ff, threads);

    const auto t1 = std::chrono::steady_clock::now();
    res.metrics.elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    res.metrics.bucket_steps = bucket_steps + 1;
    res.metrics.light_phases = light_phases;
    res.metrics.reachable_cells = res.ff.reachable_cells;

    return res;
}

#ifdef USE_MPI
template <bool RecordTrace>
ParallelResult run_mpi_delta_stepping(const Graph& g, const ParallelConfig& cfg) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ParallelResult res;
    res.metrics.backend_name = "MPI Delta-Stepping";
    res.metrics.workers = world_size;

    if (g.p.size <= 0) {
        return res;
    }

    const std::int64_t n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    const std::uint64_t delta = resolve_delta(g, cfg.delta);
    res.metrics.delta_used = delta;

    // 1D row decomposition across MPI ranks
    const std::int64_t base_rows = n / world_size;
    const std::int64_t rem_rows = n % world_size;
    const std::int64_t r_start = rank * base_rows + std::min<std::int64_t>(rank, rem_rows);
    const std::int64_t local_rows = base_rows + (rank < rem_rows ? 1 : 0);
    const std::int64_t r_end = r_start + local_rows;
    const std::int64_t local_cells = local_rows * n;
    const std::int64_t cell_offset = r_start * n;

    auto owner_rank = [&](std::int64_t row) -> int {
        const std::int64_t split = rem_rows * (base_rows + 1);
        if (row < split) {
            return static_cast<int>(row / (base_rows + 1));
        }
        return static_cast<int>(rem_rows + (row - split) / base_rows);
    };

    std::uint64_t max_w = 1;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        if (g.weights[i] > max_w) {
            max_w = g.weights[i];
        }
    }
    const std::size_t num_buckets = static_cast<std::size_t>((max_w / delta) + 3);

    std::vector<std::uint64_t> local_dist(local_cells, FloodField::INF);
    std::vector<std::int64_t> local_trace;
    if constexpr (RecordTrace) {
        local_trace.assign(local_cells, -1);
    }

    std::vector<std::vector<QueueEntry>> buckets(num_buckets);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    const std::int64_t target = g.p.t;
    if (target >= 0 && target < g.p.size && g.weights[target] != 0) {
        const std::int64_t tr = target / n;
        if (tr >= r_start && tr < r_end) {
            const std::int64_t l_idx = target - cell_offset;
            local_dist[l_idx] = 0;
            buckets[0].push_back({0, target});
        }
    }

    struct RemoteUpdate {
        std::int64_t global_idx;
        std::uint64_t new_dist;
    };

    std::vector<std::vector<RemoteUpdate>> send_bufs(world_size);
    std::vector<int> send_counts(world_size, 0);
    std::vector<int> recv_counts(world_size, 0);
    std::vector<int> sdispls(world_size, 0);
    std::vector<int> rdispls(world_size, 0);
    std::vector<RemoteUpdate> flat_send;
    std::vector<RemoteUpdate> flat_recv;

    auto exchange_remote_updates = [&]() {
        std::size_t total_send = 0;
        for (int p = 0; p < world_size; ++p) {
            send_counts[p] = static_cast<int>(send_bufs[p].size() * sizeof(RemoteUpdate));
            sdispls[p] = static_cast<int>(total_send * sizeof(RemoteUpdate));
            total_send += send_bufs[p].size();
        }
        flat_send.resize(total_send);
        std::size_t cursor = 0;
        for (int p = 0; p < world_size; ++p) {
            if (!send_bufs[p].empty()) {
                std::memcpy(flat_send.data() + cursor,
                            send_bufs[p].data(),
                            send_bufs[p].size() * sizeof(RemoteUpdate));
                cursor += send_bufs[p].size();
                send_bufs[p].clear();
            }
        }

        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        int total_recv_bytes = 0;
        for (int p = 0; p < world_size; ++p) {
            rdispls[p] = total_recv_bytes;
            total_recv_bytes += recv_counts[p];
        }
        flat_recv.resize(total_recv_bytes / sizeof(RemoteUpdate));

        MPI_Alltoallv(flat_send.data(), send_counts.data(), sdispls.data(), MPI_BYTE,
                      flat_recv.data(), recv_counts.data(), rdispls.data(), MPI_BYTE,
                      MPI_COMM_WORLD);

        for (const auto& upd : flat_recv) {
            const std::int64_t l_idx = upd.global_idx - cell_offset;
            if (upd.new_dist < local_dist[l_idx]) {
                local_dist[l_idx] = upd.new_dist;
                const std::uint64_t target_b = upd.new_dist / delta;
                buckets[target_b % num_buckets].push_back({upd.new_dist, upd.global_idx});
            }
        }
    };

    std::uint64_t current_bucket = 0;
    std::uint64_t bucket_steps = 0;
    std::uint64_t light_phases = 0;
    std::vector<QueueEntry> heavy_candidates;

    while (true) {
        std::uint64_t local_next = FloodField::INF;
        for (std::size_t offset = 0; offset < num_buckets; ++offset) {
            const std::uint64_t cand_b = current_bucket + offset;
            if (!buckets[cand_b % num_buckets].empty()) {
                local_next = cand_b;
                break;
            }
        }

        std::uint64_t global_next = FloodField::INF;
        MPI_Allreduce(&local_next, &global_next, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        if (global_next == FloodField::INF) {
            break;
        }

        current_bucket = global_next;
        ++bucket_steps;
        const std::size_t slot = static_cast<std::size_t>(current_bucket % num_buckets);
        heavy_candidates.clear();

        // Phase 1: Drain light vertices across all MPI ranks synchronously
        while (true) {
            int local_has_work = buckets[slot].empty() ? 0 : 1;
            int global_has_work = 0;
            MPI_Allreduce(&local_has_work, &global_has_work, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (global_has_work == 0) {
                break;
            }

            ++light_phases;
            std::vector<QueueEntry> frontier;
            frontier.swap(buckets[slot]);

            std::int64_t phase_ns = 0;
            if constexpr (RecordTrace) {
                const auto now = std::chrono::steady_clock::now();
                phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
            }

            for (const auto& entry : frontier) {
                const std::int64_t u = entry.vertex;
                const std::int64_t l_u = u - cell_offset;
                const std::uint64_t d_u = local_dist[l_u];
                if (entry.dist > d_u) {
                    continue;
                }

                if constexpr (RecordTrace) {
                    local_trace[l_u] = phase_ns;
                }

                const std::uint64_t step_cost = static_cast<std::uint64_t>(g.weights[u]);
                if (step_cost > delta) {
                    heavy_candidates.push_back({d_u, u});
                    continue;
                }

                const std::int64_t r = u / n;
                const std::int64_t c = u % n;
                for (int dir_i = 0; dir_i < 4; ++dir_i) {
                    const std::int64_t nr = r + DR[dir_i];
                    const std::int64_t nc = c + DC[dir_i];
                    if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                        continue;
                    }
                    const std::int64_t v = nr * n + nc;
                    if (g.weights[v] == 0) {
                        continue;
                    }

                    const std::uint64_t new_dist = d_u + step_cost;
                    if (nr >= r_start && nr < r_end) {
                        const std::int64_t l_v = v - cell_offset;
                        if (new_dist < local_dist[l_v]) {
                            local_dist[l_v] = new_dist;
                            const std::uint64_t target_b = new_dist / delta;
                            buckets[target_b % num_buckets].push_back({new_dist, v});
                        }
                    } else {
                        send_bufs[owner_rank(nr)].push_back({v, new_dist});
                    }
                }
            }

            exchange_remote_updates();
        }

        // Phase 2: Relax heavy vertices once across all ranks
        for (const auto& entry : heavy_candidates) {
            const std::int64_t u = entry.vertex;
            const std::int64_t l_u = u - cell_offset;
            const std::uint64_t d_u = local_dist[l_u];
            if (entry.dist > d_u) {
                continue;
            }

            const std::uint64_t step_cost = static_cast<std::uint64_t>(g.weights[u]);
            const std::int64_t r = u / n;
            const std::int64_t c = u % n;
            for (int dir_i = 0; dir_i < 4; ++dir_i) {
                const std::int64_t nr = r + DR[dir_i];
                const std::int64_t nc = c + DC[dir_i];
                if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                    continue;
                }
                const std::int64_t v = nr * n + nc;
                if (g.weights[v] == 0) {
                    continue;
                }

                const std::uint64_t new_dist = d_u + step_cost;
                if (nr >= r_start && nr < r_end) {
                    const std::int64_t l_v = v - cell_offset;
                    if (new_dist < local_dist[l_v]) {
                        local_dist[l_v] = new_dist;
                        const std::uint64_t target_b = new_dist / delta;
                        buckets[target_b % num_buckets].push_back({new_dist, v});
                    }
                } else {
                    send_bufs[owner_rank(nr)].push_back({v, new_dist});
                }
            }
        }
        exchange_remote_updates();

        ++current_bucket;
    }

    // Gather converged distances (and traces if enabled) to Rank 0
    std::vector<int> recv_cell_counts(world_size, 0);
    std::vector<int> cell_displs(world_size, 0);
    if (rank == 0) {
        res.ff.n = n;
        res.ff.dist.resize(g.p.size);
        if constexpr (RecordTrace) {
            res.ff.settle_ns.resize(g.p.size);
        }
        for (int p = 0; p < world_size; ++p) {
            const std::int64_t p_rows = base_rows + (p < rem_rows ? 1 : 0);
            recv_cell_counts[p] = static_cast<int>(p_rows * n);
            cell_displs[p] = (p == 0) ? 0 : (cell_displs[p - 1] + recv_cell_counts[p - 1]);
        }
    }

    MPI_Gatherv(local_dist.data(), static_cast<int>(local_cells), MPI_UINT64_T,
                rank == 0 ? res.ff.dist.data() : nullptr,
                recv_cell_counts.data(), cell_displs.data(), MPI_UINT64_T,
                0, MPI_COMM_WORLD);

    if constexpr (RecordTrace) {
        MPI_Gatherv(local_trace.data(), static_cast<int>(local_cells), MPI_INT64_T,
                    rank == 0 ? res.ff.settle_ns.data() : nullptr,
                    recv_cell_counts.data(), cell_displs.data(), MPI_INT64_T,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        compute_directions_omp(g, res.ff, 1);
    }

    const auto t1 = std::chrono::steady_clock::now();
    res.metrics.elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    res.metrics.bucket_steps = bucket_steps;
    res.metrics.light_phases = light_phases;
    res.metrics.reachable_cells = res.ff.reachable_cells;

    return res;
}
#endif

} // namespace

ParallelResult compute_flood_field_parallel(const Graph& g, const ParallelConfig& cfg) {
    if (cfg.backend == PARALLEL_BACKEND::OPENMP) {
        return run_openmp_delta_stepping<false>(g, cfg);
    }
#ifdef USE_MPI
    return run_mpi_delta_stepping<false>(g, cfg);
#else
    throw std::runtime_error("MPI backend requested, but binary was compiled without -DUSE_MPI.");
#endif
}

ParallelResult compute_flood_field_parallel_traced(const Graph& g, const ParallelConfig& cfg) {
    if (cfg.backend == PARALLEL_BACKEND::OPENMP) {
        return run_openmp_delta_stepping<true>(g, cfg);
    }
#ifdef USE_MPI
    return run_mpi_delta_stepping<true>(g, cfg);
#else
    throw std::runtime_error("MPI backend requested, but binary was compiled without -DUSE_MPI.");
#endif
}