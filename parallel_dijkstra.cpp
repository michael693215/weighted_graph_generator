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
constexpr std::int8_t FLOW_TO_CURR[4] = {2, 3, 0, 1};

std::uint32_t resolve_delta(const Graph& g, std::uint64_t requested_delta) {
    if (requested_delta > 0) {
        return static_cast<std::uint32_t>(requested_delta);
    }
    std::uint32_t max_w = 1;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        if (g.weights[i] > max_w) {
            max_w = g.weights[i];
        }
    }
    return std::max<std::uint32_t>(1, (max_w * 3) / 2);
}

// Atomically relaxes a bit-packed (29-bit dist | 3-bit dir) cell.
// Returns true ONLY if the unpacked distance strictly decreased (meaning `v` must be queued).
// Sets `first_visit = true` on the unique transition from INF_PACKED -> reachable.
inline bool atomic_relax_packed(std::uint32_t* addr,
                                std::uint32_t new_dist,
                                std::int8_t flow_dir,
                                bool& first_visit) {
    const std::uint32_t new_packed = FloodField::pack(new_dist, flow_dir);
    std::uint32_t prev = __atomic_load_n(addr, __ATOMIC_RELAXED);
    while (new_packed < prev) {
        if (__atomic_compare_exchange_n(addr, &prev, new_packed, true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            first_visit = (prev == FloodField::INF_PACKED);
            return FloodField::unpack_dist(prev) > new_dist;
        }
    }
    first_visit = false;
    return false;
}

// Fixed-size 2-level hierarchical bitset:
// - l0: 1 bit per cell (N^2 / 8 bytes)
// - l1: 1 bit per 64 l0 words (4,096 cells per bit; fits in L1 cache)
// Eliminates dynamic queue reallocations, duplicate pushes, and empty-word scans.
struct HierarchicalBitset {
    std::vector<std::uint64_t> l0;
    std::vector<std::uint64_t> l1;

    explicit HierarchicalBitset(std::size_t num_bits = 0) {
        const std::size_t n0 = (num_bits + 63) / 64;
        const std::size_t n1 = (n0 + 63) / 64;
        l0.assign(n0, 0ULL);
        l1.assign(n1, 0ULL);
    }

    inline void set_non_atomic(std::uint32_t idx) noexcept {
        const std::size_t w0 = static_cast<std::size_t>(idx) >> 6;
        l0[w0] |= (1ULL << (idx & 63u));
        l1[w0 >> 6] |= (1ULL << (w0 & 63u));
    }

    inline void set_atomic(std::uint32_t idx) noexcept {
        const std::size_t w0 = static_cast<std::size_t>(idx) >> 6;
        const std::uint64_t m0 = 1ULL << (idx & 63u);
        if ((__atomic_fetch_or(&l0[w0], m0, __ATOMIC_RELAXED) & m0) == 0) {
            const std::size_t w1 = w0 >> 6;
            const std::uint64_t m1 = 1ULL << (w0 & 63u);
            if ((__atomic_load_n(&l1[w1], __ATOMIC_RELAXED) & m1) == 0) {
                __atomic_fetch_or(&l1[w1], m1, __ATOMIC_RELAXED);
            }
        }
    }

    void swap(HierarchicalBitset& other) noexcept {
        l0.swap(other.l0);
        l1.swap(other.l1);
    }
};

struct alignas(64) ThreadLocalState {
    std::int64_t reachable = 0;
    std::uint64_t min_bucket = std::numeric_limits<std::uint64_t>::max();
    bool has_next = false;
    bool has_future = false;
    bool extracted = false;
    bool rem_future = false;
};

template <bool RecordTrace>
ParallelResult run_openmp_delta_stepping(const Graph& g, const ParallelConfig& cfg) {
    ParallelResult res;
    res.metrics.backend_name = "OpenMP Delta-Stepping";

    if (g.p.size <= 0) {
        return res;
    }
    if (static_cast<std::uint64_t>(g.p.size) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Grid cell count exceeds uint32_t index limit.");
    }

    const std::int64_t n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    res.ff.n = n;
    res.ff.cells.assign(g.p.size, FloodField::INF_PACKED);
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

    const std::size_t grid_size = static_cast<std::size_t>(g.p.size);
    HierarchicalBitset active_bits(grid_size);
    HierarchicalBitset next_bits(grid_size);
    HierarchicalBitset future_bits(grid_size);
    const std::size_t num_l1 = active_bits.l1.size();

    std::vector<ThreadLocalState> thread_state(threads);
    thread_state[0].reachable = 1; // Target cell

    constexpr std::uint64_t INF_BUCKET = std::numeric_limits<std::uint64_t>::max();

    std::uint64_t current_bucket = 0;
    std::uint64_t bucket_lower = 0;
    std::uint64_t bucket_upper = delta;
    std::uint64_t cand_bucket = 0;
    std::uint64_t cand_lower = 0;
    std::uint64_t cand_upper = 0;

    bool done = false;
    bool future_non_empty = false;
    bool need_future_extract = false;
    bool need_second_extract = false;

    std::int64_t phase_ns = 0;
    std::uint64_t bucket_steps = 1;
    std::uint64_t light_phases = 0;

    const auto t0 = std::chrono::steady_clock::now();

    std::uint32_t* cells_ptr = res.ff.cells.data();
    cells_ptr[target] = FloodField::pack(0, -1);
    active_bits.set_non_atomic(static_cast<std::uint32_t>(target));

    #pragma omp parallel num_threads(threads)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        while (true) {
            #pragma omp single
            {
                ++light_phases;
                if constexpr (RecordTrace) {
                    const auto now = std::chrono::steady_clock::now();
                    phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                }
            }

            bool my_next = false;
            bool my_future = false;
            std::int64_t my_newly_reached = 0;

            #pragma omp for schedule(dynamic, 8)
            for (std::size_t w1 = 0; w1 < num_l1; ++w1) {
                std::uint64_t m1 = active_bits.l1[w1];
                if (m1 == 0) {
                    continue;
                }
                active_bits.l1[w1] = 0;

                while (m1 != 0) {
                    const int b1 = __builtin_ctzll(m1);
                    m1 &= (m1 - 1);
                    const std::size_t w0 = (w1 << 6) + static_cast<std::size_t>(b1);
                    std::uint64_t m0 = active_bits.l0[w0];
                    if (m0 == 0) {
                        continue;
                    }
                    active_bits.l0[w0] = 0;

                    while (m0 != 0) {
                        const int b0 = __builtin_ctzll(m0);
                        m0 &= (m0 - 1);
                        const std::uint32_t u = static_cast<std::uint32_t>((w0 << 6) + static_cast<std::size_t>(b0));

                        const std::uint32_t packed_u = __atomic_load_n(&cells_ptr[u], __ATOMIC_RELAXED);
                        const std::uint32_t d_u = FloodField::unpack_dist(packed_u);

                        if (d_u < bucket_lower || d_u >= bucket_upper) {
                            continue;
                        }

                        if constexpr (RecordTrace) {
                            res.ff.settle_ns[u] = phase_ns;
                        }

                        const std::uint32_t step_cost = static_cast<std::uint32_t>(g.weights[u]);
                        const std::int64_t r = static_cast<std::int64_t>(u) / n;
                        const std::int64_t c = static_cast<std::int64_t>(u) % n;

                        for (int dir_i = 0; dir_i < 4; ++dir_i) {
                            const std::int64_t nr = r + DR[dir_i];
                            const std::int64_t nc = c + DC[dir_i];
                            if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                                continue;
                            }
                            const std::uint32_t v = static_cast<std::uint32_t>(nr * n + nc);
                            if (g.weights[v] == 0) {
                                continue;
                            }

                            const std::uint32_t new_dist = d_u + step_cost;
                            bool first_visit = false;
                            if (atomic_relax_packed(&cells_ptr[v], new_dist, FLOW_TO_CURR[dir_i], first_visit)) {
                                if (first_visit) {
                                    ++my_newly_reached;
                                }
                                if (new_dist < bucket_upper) {
                                    next_bits.set_atomic(v);
                                    my_next = true;
                                } else {
                                    future_bits.set_atomic(v);
                                    my_future = true;
                                }
                            }
                        }
                    }
                }
            } // implicit barrier

            thread_state[tid].has_next = my_next;
            thread_state[tid].has_future = my_future;
            thread_state[tid].reachable += my_newly_reached;

            #pragma omp barrier

            #pragma omp single
            {
                bool any_next = false;
                for (int t = 0; t < threads; ++t) {
                    if (thread_state[t].has_next) {
                        any_next = true;
                    }
                    if (thread_state[t].has_future) {
                        future_non_empty = true;
                    }
                }

                need_future_extract = false;
                if (any_next) {
                    active_bits.swap(next_bits);
                } else if (!future_non_empty) {
                    done = true;
                } else if (delta >= max_w) {
                    // O(1) swap: every pending cell in future_bits belongs to current_bucket + 1
                    ++current_bucket;
                    bucket_lower = current_bucket * delta;
                    bucket_upper = bucket_lower + delta;
                    active_bits.swap(future_bits);
                    future_non_empty = false;
                    ++bucket_steps;
                } else {
                    need_future_extract = true;
                    cand_bucket = current_bucket + 1;
                    cand_lower = cand_bucket * delta;
                    cand_upper = cand_lower + delta;
                }
            }

            if (done) {
                break;
            }

            if (need_future_extract) {
                bool my_extracted = false;
                bool my_rem_future = false;
                std::uint64_t my_min_b = INF_BUCKET;

                #pragma omp for schedule(dynamic, 8)
                for (std::size_t w1 = 0; w1 < num_l1; ++w1) {
                    std::uint64_t m1 = future_bits.l1[w1];
                    if (m1 == 0) {
                        continue;
                    }

                    std::uint64_t new_future_m1 = 0;
                    std::uint64_t new_active_m1 = 0;

                    while (m1 != 0) {
                        const int b1 = __builtin_ctzll(m1);
                        m1 &= (m1 - 1);
                        const std::size_t w0 = (w1 << 6) + static_cast<std::size_t>(b1);
                        std::uint64_t m0 = future_bits.l0[w0];
                        if (m0 == 0) {
                            continue;
                        }

                        std::uint64_t rem_m0 = 0;
                        std::uint64_t act_m0 = 0;

                        while (m0 != 0) {
                            const int b0 = __builtin_ctzll(m0);
                            m0 &= (m0 - 1);
                            const std::uint32_t v = static_cast<std::uint32_t>((w0 << 6) + static_cast<std::size_t>(b0));
                            const std::uint32_t d_v = FloodField::unpack_dist(cells_ptr[v]);

                            if (d_v < cand_lower) {
                                continue; // Stale entry settled in an earlier bucket
                            } else if (d_v < cand_upper) {
                                act_m0 |= (1ULL << b0);
                            } else {
                                rem_m0 |= (1ULL << b0);
                                const std::uint64_t b_v = static_cast<std::uint64_t>(d_v) / delta;
                                if (b_v < my_min_b) {
                                    my_min_b = b_v;
                                }
                            }
                        }

                        future_bits.l0[w0] = rem_m0;
                        if (rem_m0 != 0) {
                            new_future_m1 |= (1ULL << b1);
                            my_rem_future = true;
                        }
                        if (act_m0 != 0) {
                            active_bits.l0[w0] = act_m0;
                            new_active_m1 |= (1ULL << b1);
                            my_extracted = true;
                        }
                    }

                    future_bits.l1[w1] = new_future_m1;
                    if (new_active_m1 != 0) {
                        active_bits.l1[w1] = new_active_m1;
                    }
                }

                thread_state[tid].extracted = my_extracted;
                thread_state[tid].rem_future = my_rem_future;
                thread_state[tid].min_bucket = my_min_b;

                #pragma omp barrier

                #pragma omp single
                {
                    bool any_extracted = false;
                    bool any_rem = false;
                    std::uint64_t global_min_b = INF_BUCKET;
                    for (int t = 0; t < threads; ++t) {
                        if (thread_state[t].extracted) {
                            any_extracted = true;
                        }
                        if (thread_state[t].rem_future) {
                            any_rem = true;
                        }
                        if (thread_state[t].min_bucket < global_min_b) {
                            global_min_b = thread_state[t].min_bucket;
                        }
                    }

                    future_non_empty = any_rem;
                    need_second_extract = false;

                    if (any_extracted) {
                        current_bucket = cand_bucket;
                        bucket_lower = cand_lower;
                        bucket_upper = cand_upper;
                        ++bucket_steps;
                    } else if (!any_rem || global_min_b == INF_BUCKET) {
                        done = true;
                    } else {
                        current_bucket = global_min_b;
                        bucket_lower = current_bucket * delta;
                        bucket_upper = bucket_lower + delta;
                        ++bucket_steps;
                        need_second_extract = true;
                    }
                }

                if (done) {
                    break;
                }

                if (need_second_extract) {
                    bool my_rem2 = false;

                    #pragma omp for schedule(dynamic, 8)
                    for (std::size_t w1 = 0; w1 < num_l1; ++w1) {
                        std::uint64_t m1 = future_bits.l1[w1];
                        if (m1 == 0) {
                            continue;
                        }

                        std::uint64_t new_future_m1 = 0;
                        std::uint64_t new_active_m1 = 0;

                        while (m1 != 0) {
                            const int b1 = __builtin_ctzll(m1);
                            m1 &= (m1 - 1);
                            const std::size_t w0 = (w1 << 6) + static_cast<std::size_t>(b1);
                            std::uint64_t m0 = future_bits.l0[w0];
                            if (m0 == 0) {
                                continue;
                            }

                            std::uint64_t rem_m0 = 0;
                            std::uint64_t act_m0 = 0;

                            while (m0 != 0) {
                                const int b0 = __builtin_ctzll(m0);
                                m0 &= (m0 - 1);
                                const std::uint32_t v = static_cast<std::uint32_t>((w0 << 6) + static_cast<std::size_t>(b0));
                                const std::uint32_t d_v = FloodField::unpack_dist(cells_ptr[v]);

                                if (d_v < bucket_upper) {
                                    act_m0 |= (1ULL << b0);
                                } else {
                                    rem_m0 |= (1ULL << b0);
                                }
                            }

                            future_bits.l0[w0] = rem_m0;
                            if (rem_m0 != 0) {
                                new_future_m1 |= (1ULL << b1);
                                my_rem2 = true;
                            }
                            if (act_m0 != 0) {
                                active_bits.l0[w0] = act_m0;
                                new_active_m1 |= (1ULL << b1);
                            }
                        }

                        future_bits.l1[w1] = new_future_m1;
                        if (new_active_m1 != 0) {
                            active_bits.l1[w1] = new_active_m1;
                        }
                    }

                    thread_state[tid].rem_future = my_rem2;
                    #pragma omp barrier

                    #pragma omp single
                    {
                        bool any_rem2 = false;
                        for (int t = 0; t < threads; ++t) {
                            if (thread_state[t].rem_future) {
                                any_rem2 = true;
                            }
                        }
                        future_non_empty = any_rem2;
                    }
                }
            }
        }
    }

    std::int64_t total_reachable = 0;
    for (int t = 0; t < threads; ++t) {
        total_reachable += thread_state[t].reachable;
    }
    res.ff.reachable_cells = total_reachable;

    const auto t1 = std::chrono::steady_clock::now();
    res.metrics.elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    res.metrics.bucket_steps = bucket_steps;
    res.metrics.light_phases = light_phases;
    res.metrics.reachable_cells = total_reachable;

    return res;
}

#ifdef USE_MPI
struct QueueEntry {
    std::uint32_t dist;
    std::uint32_t vertex;
};

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
    const std::uint32_t delta = resolve_delta(g, cfg.delta);
    res.metrics.delta_used = delta;

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

    std::uint32_t max_w = 1;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        if (g.weights[i] > max_w) {
            max_w = g.weights[i];
        }
    }
    const std::size_t num_buckets = static_cast<std::size_t>((max_w / delta) + 3);

    std::vector<std::uint32_t> local_packed(local_cells, FloodField::INF_PACKED);
    std::vector<std::int64_t> local_trace;
    if constexpr (RecordTrace) {
        local_trace.assign(local_cells, -1);
    }

    std::vector<std::vector<QueueEntry>> buckets(num_buckets);
    std::int64_t local_reachable = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();

    const std::int64_t target = g.p.t;
    if (target >= 0 && target < g.p.size && g.weights[target] != 0) {
        const std::int64_t tr = target / n;
        if (tr >= r_start && tr < r_end) {
            const std::int64_t l_idx = target - cell_offset;
            local_packed[l_idx] = FloodField::pack(0, -1);
            local_reachable = 1;
            buckets[0].push_back({0, static_cast<std::uint32_t>(target)});
        }
    }

    struct RemoteUpdate {
        std::uint32_t global_idx;
        std::uint32_t packed_val;
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
            const std::int64_t l_idx = static_cast<std::int64_t>(upd.global_idx) - cell_offset;
            const std::uint32_t prev = local_packed[l_idx];
            if (upd.packed_val < prev) {
                if (prev == FloodField::INF_PACKED) {
                    ++local_reachable;
                }
                local_packed[l_idx] = upd.packed_val;
                const std::uint32_t new_dist = FloodField::unpack_dist(upd.packed_val);
                if (new_dist < FloodField::unpack_dist(prev)) {
                    const std::uint32_t target_b = new_dist / delta;
                    buckets[target_b % num_buckets].push_back({new_dist, upd.global_idx});
                }
            }
        }
    };

    std::uint32_t current_bucket = 0;
    std::uint64_t bucket_steps = 0;
    std::uint64_t light_phases = 0;
    std::vector<QueueEntry> heavy_candidates;

    while (true) {
        std::uint32_t local_next = FloodField::INF_DIST;
        for (std::size_t offset = 0; offset < num_buckets; ++offset) {
            const std::uint32_t cand_b = current_bucket + static_cast<std::uint32_t>(offset);
            if (!buckets[cand_b % num_buckets].empty()) {
                local_next = cand_b;
                break;
            }
        }

        std::uint32_t global_next = FloodField::INF_DIST;
        MPI_Allreduce(&local_next, &global_next, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
        if (global_next == FloodField::INF_DIST) {
            break;
        }

        current_bucket = global_next;
        ++bucket_steps;
        const std::size_t slot = static_cast<std::size_t>(current_bucket % num_buckets);
        heavy_candidates.clear();

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
                const std::uint32_t u = entry.vertex;
                const std::int64_t l_u = static_cast<std::int64_t>(u) - cell_offset;
                const std::uint32_t d_u = FloodField::unpack_dist(local_packed[l_u]);
                if (entry.dist > d_u) {
                    continue;
                }

                const std::uint32_t step_cost = static_cast<std::uint32_t>(g.weights[u]);
                if (step_cost > delta) {
                    heavy_candidates.push_back({d_u, u});
                    continue;
                }

                if constexpr (RecordTrace) {
                    local_trace[l_u] = phase_ns;
                }

                const std::int64_t r = static_cast<std::int64_t>(u) / n;
                const std::int64_t c = static_cast<std::int64_t>(u) % n;
                for (int dir_i = 0; dir_i < 4; ++dir_i) {
                    const std::int64_t nr = r + DR[dir_i];
                    const std::int64_t nc = c + DC[dir_i];
                    if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                        continue;
                    }
                    const std::uint32_t v = static_cast<std::uint32_t>(nr * n + nc);
                    if (g.weights[v] == 0) {
                        continue;
                    }

                    const std::uint32_t new_dist = d_u + step_cost;
                    const std::uint32_t new_packed = FloodField::pack(new_dist, FLOW_TO_CURR[dir_i]);
                    if (nr >= r_start && nr < r_end) {
                        const std::int64_t l_v = static_cast<std::int64_t>(v) - cell_offset;
                        const std::uint32_t prev = local_packed[l_v];
                        if (new_packed < prev) {
                            if (prev == FloodField::INF_PACKED) {
                                ++local_reachable;
                            }
                            local_packed[l_v] = new_packed;
                            if (new_dist < FloodField::unpack_dist(prev)) {
                                const std::uint32_t target_b = new_dist / delta;
                                buckets[target_b % num_buckets].push_back({new_dist, v});
                            }
                        }
                    } else {
                        send_bufs[owner_rank(nr)].push_back({v, new_packed});
                    }
                }
            }

            exchange_remote_updates();
        }

        std::int64_t heavy_phase_ns = 0;
        if constexpr (RecordTrace) {
            const auto now = std::chrono::steady_clock::now();
            heavy_phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        }

        for (const auto& entry : heavy_candidates) {
            const std::uint32_t u = entry.vertex;
            const std::int64_t l_u = static_cast<std::int64_t>(u) - cell_offset;
            const std::uint32_t d_u = FloodField::unpack_dist(local_packed[l_u]);
            if (entry.dist > d_u) {
                continue;
            }

            if constexpr (RecordTrace) {
                local_trace[l_u] = heavy_phase_ns;
            }

            const std::uint32_t step_cost = static_cast<std::uint32_t>(g.weights[u]);
            const std::int64_t r = static_cast<std::int64_t>(u) / n;
            const std::int64_t c = static_cast<std::int64_t>(u) % n;
            for (int dir_i = 0; dir_i < 4; ++dir_i) {
                const std::int64_t nr = r + DR[dir_i];
                const std::int64_t nc = c + DC[dir_i];
                if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                    continue;
                }
                const std::uint32_t v = static_cast<std::uint32_t>(nr * n + nc);
                if (g.weights[v] == 0) {
                    continue;
                }

                const std::uint32_t new_dist = d_u + step_cost;
                const std::uint32_t new_packed = FloodField::pack(new_dist, FLOW_TO_CURR[dir_i]);
                if (nr >= r_start && nr < r_end) {
                    const std::int64_t l_v = static_cast<std::int64_t>(v) - cell_offset;
                    const std::uint32_t prev = local_packed[l_v];
                    if (new_packed < prev) {
                        if (prev == FloodField::INF_PACKED) {
                            ++local_reachable;
                        }
                        local_packed[l_v] = new_packed;
                        if (new_dist < FloodField::unpack_dist(prev)) {
                            const std::uint32_t target_b = new_dist / delta;
                            buckets[target_b % num_buckets].push_back({new_dist, v});
                        }
                    }
                } else {
                    send_bufs[owner_rank(nr)].push_back({v, new_packed});
                }
            }
        }
        exchange_remote_updates();

        ++current_bucket;
    }

    std::int64_t global_reachable = 0;
    MPI_Reduce(&local_reachable, &global_reachable, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    std::vector<int> recv_cell_counts(world_size, 0);
    std::vector<int> cell_displs(world_size, 0);
    if (rank == 0) {
        res.ff.n = n;
        res.ff.cells.resize(g.p.size);
        res.ff.reachable_cells = global_reachable;
        if constexpr (RecordTrace) {
            res.ff.settle_ns.resize(g.p.size);
        }
        for (int p = 0; p < world_size; ++p) {
            const std::int64_t p_rows = base_rows + (p < rem_rows ? 1 : 0);
            recv_cell_counts[p] = static_cast<int>(p_rows * n);
            cell_displs[p] = (p == 0) ? 0 : (cell_displs[p - 1] + recv_cell_counts[p - 1]);
        }
    }

    MPI_Gatherv(local_packed.data(), static_cast<int>(local_cells), MPI_UINT32_T,
                rank == 0 ? res.ff.cells.data() : nullptr,
                recv_cell_counts.data(), cell_displs.data(), MPI_UINT32_T,
                0, MPI_COMM_WORLD);

    if constexpr (RecordTrace) {
        MPI_Gatherv(local_trace.data(), static_cast<int>(local_cells), MPI_INT64_T,
                    rank == 0 ? res.ff.settle_ns.data() : nullptr,
                    recv_cell_counts.data(), cell_displs.data(), MPI_INT64_T,
                    0, MPI_COMM_WORLD);
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