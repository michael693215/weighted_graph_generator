#include "serial_dijkstra.hpp"
#include <chrono>
#include <cmath>
#include <queue>
#include <utility>

namespace {

template <bool RecordTrace>
FloodField run_dijkstra(const Graph& g) {
    FloodField ff;
    if (g.p.size <= 0) {
        return ff;
    }

    ff.n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    ff.dist.assign(g.p.size, FloodField::INF);
    ff.dir.assign(g.p.size, -1);
    if constexpr (RecordTrace) {
        ff.settle_ns.assign(g.p.size, -1);
    }

    const std::int64_t target = g.p.t;
    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return ff;
    }

    using State = std::pair<std::uint64_t, std::int64_t>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;

    const auto t0 = std::chrono::steady_clock::now();

    ff.dist[target] = 0;
    pq.emplace(0, target);

    constexpr std::int64_t dr[4] = {-1, 0, 1, 0};
    constexpr std::int64_t dc[4] = {0, 1, 0, -1};
    constexpr std::int8_t flow_to_curr[4] = {2, 3, 0, 1};

    while (!pq.empty()) {
        const auto [d, curr] = pq.top();
        pq.pop();

        if (d > ff.dist[curr]) {
            continue;
        }
        ++ff.reachable_cells;

        if constexpr (RecordTrace) {
            const auto now = std::chrono::steady_clock::now();
            ff.settle_ns[curr] = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        }

        const std::int64_t r = curr / ff.n;
        const std::int64_t c = curr % ff.n;
        const std::uint64_t step_cost = static_cast<std::uint64_t>(g.weights[curr]);

        for (int i = 0; i < 4; ++i) {
            const std::int64_t nr = r + dr[i];
            const std::int64_t nc = c + dc[i];

            if (nr < 0 || nr >= ff.n || nc < 0 || nc >= ff.n) {
                continue;
            }

            const std::int64_t next = nr * ff.n + nc;
            if (g.weights[next] == 0) {
                continue;
            }

            const std::uint64_t new_dist = d + step_cost;
            if (new_dist < ff.dist[next]) {
                ff.dist[next] = new_dist;
                ff.dir[next] = flow_to_curr[i];
                pq.emplace(new_dist, next);
            }
        }
    }

    return ff;
}

template <bool RecordTrace>
FloodField run_dial(const Graph& g) {
    FloodField ff;
    if (g.p.size <= 0) {
        return ff;
    }

    ff.n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    ff.dist.assign(g.p.size, FloodField::INF);
    ff.dir.assign(g.p.size, -1);
    if constexpr (RecordTrace) {
        ff.settle_ns.assign(g.p.size, -1);
    }

    const std::int64_t target = g.p.t;
    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return ff;
    }

    // Max edge cost = max cell weight (cost of leaving a cell).
    std::uint64_t max_w = 0;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        max_w = std::max(max_w, static_cast<std::uint64_t>(g.weights[i]));
    }

    // Circular bucket queue: all pending distances lie in [d, d + max_w].
    const std::uint64_t num_buckets = max_w + 1;
    std::vector<std::vector<std::int64_t>> buckets(num_buckets);
    std::uint64_t pending = 0; // entries currently stored across all buckets

    const auto t0 = std::chrono::steady_clock::now();

    ff.dist[target] = 0;
    buckets[0].push_back(target);
    ++pending;

    constexpr std::int64_t dr[4] = {-1, 0, 1, 0};
    constexpr std::int64_t dc[4] = {0, 1, 0, -1};
    constexpr std::int8_t flow_to_curr[4] = {2, 3, 0, 1};

    for (std::uint64_t d = 0; pending > 0; ++d) {
        auto& bucket = buckets[d % num_buckets];

        // Edge costs are >= 1, so nothing processed here is ever added back
        // to this same bucket; popping until empty is safe.
        while (!bucket.empty()) {
            const std::int64_t curr = bucket.back();
            bucket.pop_back();
            --pending;

            if (d > ff.dist[curr]) {
                continue; // stale entry
            }
            ++ff.reachable_cells;

            if constexpr (RecordTrace) {
                const auto now = std::chrono::steady_clock::now();
                ff.settle_ns[curr] =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
            }

            const std::int64_t r = curr / ff.n;
            const std::int64_t c = curr % ff.n;
            const std::uint64_t step_cost = static_cast<std::uint64_t>(g.weights[curr]);

            for (int i = 0; i < 4; ++i) {
                const std::int64_t nr = r + dr[i];
                const std::int64_t nc = c + dc[i];

                if (nr < 0 || nr >= ff.n || nc < 0 || nc >= ff.n) {
                    continue;
                }

                const std::int64_t next = nr * ff.n + nc;
                if (g.weights[next] == 0) {
                    continue;
                }

                const std::uint64_t new_dist = d + step_cost;
                if (new_dist < ff.dist[next]) {
                    ff.dist[next] = new_dist;
                    ff.dir[next] = flow_to_curr[i];
                    buckets[new_dist % num_buckets].push_back(next);
                    ++pending;
                }
            }
        }
    }

    return ff;
}

} // namespace

FloodField compute_flood_field_serial(const Graph& g, ALGORITHM a) {
    return a == ALGORITHM::DIJKSTRA ? run_dijkstra<false>(g) 
        : run_dial<false>(g);
}

FloodField compute_flood_field_serial_traced(const Graph& g, ALGORITHM a) {
    return a == ALGORITHM::DIJKSTRA ? run_dijkstra<true>(g) 
        : run_dial<true>(g);
}
