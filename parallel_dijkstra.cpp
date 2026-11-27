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

// Sets bit `idx` in a flat 64-bit word array.
// Because atomic RMWs on the same word are totally ordered, strictly ONE thread
// observes `prev == 0` and records `w0` in its thread-local active-block list.
inline void set_bit_and_track_block(std::uint64_t* bits,
                                    std::uint32_t idx,
                                    std::vector<std::uint32_t>& local_blocks) noexcept {
    const std::uint32_t w0 = idx >> 6;
    const std::uint64_t m0 = 1ULL << (idx & 63u);
    if (__atomic_fetch_or(&bits[w0], m0, __ATOMIC_RELAXED) == 0ULL) {
        local_blocks.push_back(w0);
    }
}

struct alignas(64) AlignedBlockVec {
    std::vector<std::uint32_t> data;
};

struct alignas(64) ThreadStats {
    std::int64_t reachable = 0;
    std::uint64_t min_bucket = std::numeric_limits<std::uint64_t>::max();
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
    const std::size_t num_words = (grid_size + 63) / 64;

    // Flat 1-bit-per-cell deduplication arrays (no l1 scan layer)
    std::vector<std::uint64_t> active_bits(num_words, 0ULL);
    std::vector<std::uint64_t> next_bits(num_words, 0ULL);
    std::vector<std::uint64_t> future_bits(num_words, 0ULL);

    // Compacted per-thread lists of non-zero 64-cell block indices (w0)
    std::vector<AlignedBlockVec> active_chunks(threads);
    std::vector<AlignedBlockVec> next_chunks(threads);
    std::vector<AlignedBlockVec> future_chunks(threads);
    std::vector<AlignedBlockVec> rem_chunks(threads);

    std::vector<std::size_t> active_offsets(threads + 1, 0);
    std::vector<std::size_t> future_offsets(threads + 1, 0);

    std::vector<ThreadStats> thread_stats(threads);
    thread_stats[0].reachable = 1; // Target cell

    constexpr std::uint64_t INF_BUCKET = std::numeric_limits<std::uint64_t>::max();

    std::uint64_t current_bucket = 0;
    std::uint64_t bucket_lower = 0;
    std::uint64_t bucket_upper = delta;
    std::uint64_t cand_bucket = 0;
    std::uint64_t cand_lower = 0;
    std::uint64_t cand_upper = 0;

    std::size_t total_active = 1;
    std::size_t total_future = 0;

    bool done = false;
    bool need_future_extract = false;
    bool need_second_extract = false;

    std::int64_t phase_ns = 0;
    std::uint64_t bucket_steps = 1;
    std::uint64_t light_phases = 1;

    const auto t0 = std::chrono::steady_clock::now();

    std::uint32_t* cells_ptr = res.ff.cells.data();
    const std::uint16_t* weights_ptr = g.weights.data();

    // Seed target cell into active frontier
    cells_ptr[target] = FloodField::pack(0, -1);
    const std::uint32_t target_w0 = static_cast<std::uint32_t>(target >> 6);
    active_bits[target_w0] = 1ULL << (target & 63);
    active_chunks[0].data.push_back(target_w0);
    for (int t = 0; t < threads; ++t) {
        active_offsets[t + 1] = 1;
    }

    auto build_offsets = [threads](const std::vector<AlignedBlockVec>& chunks,
                                   std::vector<std::size_t>& offsets) -> std::size_t {
        offsets[0] = 0;
        for (int t = 0; t < threads; ++t) {
            offsets[t + 1] = offsets[t] + chunks[t].data.size();
        }
        return offsets[threads];
    };

    #pragma omp parallel num_threads(threads)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        while (true) {
            std::uint64_t* const act_b = active_bits.data();
            std::uint64_t* const nxt_b = next_bits.data();
            std::uint64_t* const fut_b = future_bits.data();
            auto& my_next = next_chunks[tid].data;
            auto& my_future = future_chunks[tid].data;

            std::int64_t my_newly_reached = 0;
            int src_t = 0;

            #pragma omp for schedule(static, 8) nowait
            for (std::size_t idx = 0; idx < total_active; ++idx) {
                if (idx < active_offsets[src_t]) {
                    src_t = 0;
                }
                while (idx >= active_offsets[src_t + 1]) {
                    ++src_t;
                }
                const std::uint32_t w0 = active_chunks[src_t].data[idx - active_offsets[src_t]];

                std::uint64_t m0 = act_b[w0];
                if (m0 == 0ULL) {
                    continue;
                }
                act_b[w0] = 0ULL;

                const std::size_t base_u = static_cast<std::size_t>(w0) << 6;
                const std::int64_t base_r = static_cast<std::int64_t>(base_u) / n;
                const std::int64_t base_c = static_cast<std::int64_t>(base_u) % n;

                while (m0 != 0ULL) {
                    const int b0 = __builtin_ctzll(m0);
                    m0 &= (m0 - 1);
                    const std::uint32_t u = static_cast<std::uint32_t>(base_u + static_cast<std::size_t>(b0));

                    const std::uint32_t packed_u = __atomic_load_n(&cells_ptr[u], __ATOMIC_RELAXED);
                    const std::uint32_t d_u = FloodField::unpack_dist(packed_u);

                    if (d_u < bucket_lower || d_u >= bucket_upper) {
                        continue;
                    }

                    if constexpr (RecordTrace) {
                        res.ff.settle_ns[u] = phase_ns;
                    }

                    const std::uint32_t step_cost = static_cast<std::uint32_t>(weights_ptr[u]);
                    std::int64_t r = base_r;
                    std::int64_t c = base_c + b0;
                    if (c >= n) {
                        r += c / n;
                        c %= n;
                    }

                    for (int dir_i = 0; dir_i < 4; ++dir_i) {
                        const std::int64_t nr = r + DR[dir_i];
                        const std::int64_t nc = c + DC[dir_i];
                        if (nr < 0 || nr >= n || nc < 0 || nc >= n) {
                            continue;
                        }
                        const std::uint32_t v = static_cast<std::uint32_t>(nr * n + nc);
                        if (weights_ptr[v] == 0) {
                            continue;
                        }

                        const std::uint32_t new_dist = d_u + step_cost;
                        bool first_visit = false;
                        if (atomic_relax_packed(&cells_ptr[v], new_dist, FLOW_TO_CURR[dir_i], first_visit)) {
                            if (first_visit) {
                                ++my_newly_reached;
                            }
                            if (new_dist < bucket_upper) {
                                set_bit_and_track_block(nxt_b, v, my_next);
                            } else {
                                set_bit_and_track_block(fut_b, v, my_future);
                            }
                        }
                    }
                }
            }

            thread_stats[tid].reachable += my_newly_reached;

            // Barrier #1: all active_bits cleared, all next/future block lists finalized
            #pragma omp barrier

            // Single + Barrier #2: swap frontier or advance bucket
            #pragma omp single
            {
                const std::size_t total_next = build_offsets(next_chunks, active_offsets);
                need_future_extract = false;

                if (total_next > 0) {
                    active_bits.swap(next_bits);
                    active_chunks.swap(next_chunks);
                    for (int t = 0; t < threads; ++t) {
                        next_chunks[t].data.clear();
                    }
                    total_active = total_next;
                    ++light_phases;
                    if constexpr (RecordTrace) {
                        const auto now = std::chrono::steady_clock::now();
                        phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                    }
                } else {
                    total_future = build_offsets(future_chunks, future_offsets);
                    if (total_future == 0) {
                        done = true;
                    } else if (delta >= max_w) {
                        // O(1) swap: every pending future cell belongs to current_bucket + 1
                        ++current_bucket;
                        bucket_lower = current_bucket * delta;
                        bucket_upper = bucket_lower + delta;
                        active_bits.swap(future_bits);
                        active_chunks.swap(future_chunks);
                        active_offsets = future_offsets;
                        total_active = total_future;
                        for (int t = 0; t < threads; ++t) {
                            future_chunks[t].data.clear();
                        }
                        ++bucket_steps;
                        ++light_phases;
                        if constexpr (RecordTrace) {
                            const auto now = std::chrono::steady_clock::now();
                            phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                        }
                    } else {
                        need_future_extract = true;
                        cand_bucket = current_bucket + 1;
                        cand_lower = cand_bucket * delta;
                        cand_upper = cand_lower + delta;
                        for (int t = 0; t < threads; ++t) {
                            active_chunks[t].data.clear();
                            rem_chunks[t].data.clear();
                        }
                    }
                }
            } // implicit barrier

            if (done) {
                break;
            }

            if (need_future_extract) {
                std::uint64_t* const act_b2 = active_bits.data();
                std::uint64_t* const fut_b2 = future_bits.data();
                auto& my_act = active_chunks[tid].data;
                auto& my_rem = rem_chunks[tid].data;
                std::uint64_t my_min_b = INF_BUCKET;
                int f_src_t = 0;

                #pragma omp for schedule(static, 8) nowait
                for (std::size_t idx = 0; idx < total_future; ++idx) {
                    if (idx < future_offsets[f_src_t]) {
                        f_src_t = 0;
                    }
                    while (idx >= future_offsets[f_src_t + 1]) {
                        ++f_src_t;
                    }
                    const std::uint32_t w0 = future_chunks[f_src_t].data[idx - future_offsets[f_src_t]];

                    std::uint64_t m0 = fut_b2[w0];
                    if (m0 == 0ULL) {
                        continue;
                    }

                    std::uint64_t rem_m0 = 0ULL;
                    std::uint64_t act_m0 = 0ULL;
                    const std::size_t base_u = static_cast<std::size_t>(w0) << 6;

                    while (m0 != 0ULL) {
                        const int b0 = __builtin_ctzll(m0);
                        m0 &= (m0 - 1);
                        const std::uint32_t v = static_cast<std::uint32_t>(base_u + static_cast<std::size_t>(b0));
                        const std::uint32_t d_v = FloodField::unpack_dist(cells_ptr[v]);

                        if (d_v < cand_lower) {
                            continue; // Stale entry already settled in an earlier bucket
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

                    fut_b2[w0] = rem_m0;
                    if (rem_m0 != 0ULL) {
                        my_rem.push_back(w0);
                    }
                    if (act_m0 != 0ULL) {
                        act_b2[w0] = act_m0;
                        my_act.push_back(w0);
                    }
                }

                thread_stats[tid].min_bucket = my_min_b;

                #pragma omp barrier

                #pragma omp single
                {
                    future_chunks.swap(rem_chunks);
                    total_active = build_offsets(active_chunks, active_offsets);
                    total_future = build_offsets(future_chunks, future_offsets);

                    std::uint64_t global_min_b = INF_BUCKET;
                    for (int t = 0; t < threads; ++t) {
                        if (thread_stats[t].min_bucket < global_min_b) {
                            global_min_b = thread_stats[t].min_bucket;
                        }
                    }

                    need_second_extract = false;
                    if (total_active > 0) {
                        current_bucket = cand_bucket;
                        bucket_lower = cand_lower;
                        bucket_upper = cand_upper;
                        ++bucket_steps;
                        ++light_phases;
                        if constexpr (RecordTrace) {
                            const auto now = std::chrono::steady_clock::now();
                            phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                        }
                    } else if (total_future == 0 || global_min_b == INF_BUCKET) {
                        done = true;
                    } else {
                        current_bucket = global_min_b;
                        bucket_lower = current_bucket * delta;
                        bucket_upper = bucket_lower + delta;
                        ++bucket_steps;
                        need_second_extract = true;
                        for (int t = 0; t < threads; ++t) {
                            active_chunks[t].data.clear();
                            rem_chunks[t].data.clear();
                        }
                    }
                }

                if (done) {
                    break;
                }

                if (need_second_extract) {
                    std::uint64_t* const act_b3 = active_bits.data();
                    std::uint64_t* const fut_b3 = future_bits.data();
                    auto& my_act2 = active_chunks[tid].data;
                    auto& my_rem2 = rem_chunks[tid].data;
                    int f_src_t2 = 0;

                    #pragma omp for schedule(static, 8) nowait
                    for (std::size_t idx = 0; idx < total_future; ++idx) {
                        if (idx < future_offsets[f_src_t2]) {
                            f_src_t2 = 0;
                        }
                        while (idx >= future_offsets[f_src_t2 + 1]) {
                            ++f_src_t2;
                        }
                        const std::uint32_t w0 = future_chunks[f_src_t2].data[idx - future_offsets[f_src_t2]];

                        std::uint64_t m0 = fut_b3[w0];
                        if (m0 == 0ULL) {
                            continue;
                        }

                        std::uint64_t rem_m0 = 0ULL;
                        std::uint64_t act_m0 = 0ULL;
                        const std::size_t base_u = static_cast<std::size_t>(w0) << 6;

                        while (m0 != 0ULL) {
                            const int b0 = __builtin_ctzll(m0);
                            m0 &= (m0 - 1);
                            const std::uint32_t v = static_cast<std::uint32_t>(base_u + static_cast<std::size_t>(b0));
                            const std::uint32_t d_v = FloodField::unpack_dist(cells_ptr[v]);

                            if (d_v < bucket_upper) {
                                act_m0 |= (1ULL << b0);
                            } else {
                                rem_m0 |= (1ULL << b0);
                            }
                        }

                        fut_b3[w0] = rem_m0;
                        if (rem_m0 != 0ULL) {
                            my_rem2.push_back(w0);
                        }
                        if (act_m0 != 0ULL) {
                            act_b3[w0] = act_m0;
                            my_act2.push_back(w0);
                        }
                    }

                    #pragma omp barrier

                    #pragma omp single
                    {
                        future_chunks.swap(rem_chunks);
                        total_active = build_offsets(active_chunks, active_offsets);
                        if (total_active == 0) {
                            done = true;
                        } else {
                            ++light_phases;
                            if constexpr (RecordTrace) {
                                const auto now = std::chrono::steady_clock::now();
                                phase_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
                            }
                        }
                    }

                    if (done) {
                        break;
                    }
                }
            }
        }
    }

    std::int64_t total_reachable = 0;
    for (int t = 0; t < threads; ++t) {
        total_reachable += thread_stats[t].reachable;
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