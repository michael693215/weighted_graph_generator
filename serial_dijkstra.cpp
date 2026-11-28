#include "serial_dijkstra.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <bool RecordTrace>
FloodField run_dijkstra(const Graph& g) {
    FloodField ff;
    if (g.p.size <= 0) {
        return ff;
    }
    if (static_cast<std::uint64_t>(g.p.size) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Grid cell count exceeds uint32_t index limit.");
    }

    ff.n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    ff.cells.assign(g.p.size, FloodField::INF_PACKED);
    if constexpr (RecordTrace) {
        ff.settle_ns.assign(g.p.size, -1);
    }

    const std::int64_t target = g.p.t;
    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return ff;
    }

    using State = std::pair<std::uint32_t, std::uint32_t>;
    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;

    const auto t0 = std::chrono::steady_clock::now();

    ff.cells[target] = FloodField::pack(0, -1);
    pq.emplace(0, static_cast<std::uint32_t>(target));

    constexpr std::int64_t dr[4] = {-1, 0, 1, 0};
    constexpr std::int64_t dc[4] = {0, 1, 0, -1};
    constexpr std::int8_t flow_to_curr[4] = {2, 3, 0, 1};

    while (!pq.empty()) {
        const auto [d, curr] = pq.top();
        pq.pop();

        if (d > FloodField::unpack_dist(ff.cells[curr])) {
            continue;
        }
        ++ff.reachable_cells;

        if constexpr (RecordTrace) {
            const auto now = std::chrono::steady_clock::now();
            ff.settle_ns[curr] = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
        }

        const std::int64_t r = static_cast<std::int64_t>(curr) / ff.n;
        const std::int64_t c = static_cast<std::int64_t>(curr) % ff.n;
        const std::uint32_t step_cost = static_cast<std::uint32_t>(g.weights[curr]);

        for (int i = 0; i < 4; ++i) {
            const std::int64_t nr = r + dr[i];
            const std::int64_t nc = c + dc[i];

            if (nr < 0 || nr >= ff.n || nc < 0 || nc >= ff.n) {
                continue;
            }

            const std::uint32_t next = static_cast<std::uint32_t>(nr * ff.n + nc);
            if (g.weights[next] == 0) {
                continue;
            }

            const std::uint32_t new_dist = d + step_cost;
            if (new_dist < FloodField::unpack_dist(ff.cells[next])) {
                ff.cells[next] = FloodField::pack(new_dist, flow_to_curr[i]);
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
    if (static_cast<std::uint64_t>(g.p.size) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Grid cell count exceeds uint32_t index limit.");
    }

    ff.n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(g.p.size))));
    ff.cells.assign(g.p.size, FloodField::INF_PACKED);
    if constexpr (RecordTrace) {
        ff.settle_ns.assign(g.p.size, -1);
    }

    const std::int64_t target = g.p.t;
    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return ff;
    }

    std::uint32_t max_w = 0;
    for (std::int64_t i = 0; i < g.p.size; ++i) {
        max_w = std::max(max_w, static_cast<std::uint32_t>(g.weights[i]));
    }

    const std::uint32_t num_buckets = max_w + 1;
    std::vector<std::vector<std::uint32_t>> buckets(num_buckets);
    std::uint64_t pending = 0;

    const auto t0 = std::chrono::steady_clock::now();

    ff.cells[target] = FloodField::pack(0, -1);
    buckets[0].push_back(static_cast<std::uint32_t>(target));
    ++pending;

    constexpr std::int64_t dr[4] = {-1, 0, 1, 0};
    constexpr std::int64_t dc[4] = {0, 1, 0, -1};
    constexpr std::int8_t flow_to_curr[4] = {2, 3, 0, 1};

    for (std::uint32_t d = 0; pending > 0; ++d) {
        auto& bucket = buckets[d % num_buckets];

        while (!bucket.empty()) {
            const std::uint32_t curr = bucket.back();
            bucket.pop_back();
            --pending;

            if (d > FloodField::unpack_dist(ff.cells[curr])) {
                continue;
            }
            ++ff.reachable_cells;

            if constexpr (RecordTrace) {
                const auto now = std::chrono::steady_clock::now();
                ff.settle_ns[curr] =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
            }

            const std::int64_t r = static_cast<std::int64_t>(curr) / ff.n;
            const std::int64_t c = static_cast<std::int64_t>(curr) % ff.n;
            const std::uint32_t step_cost = static_cast<std::uint32_t>(g.weights[curr]);

            for (int i = 0; i < 4; ++i) {
                const std::int64_t nr = r + dr[i];
                const std::int64_t nc = c + dc[i];

                if (nr < 0 || nr >= ff.n || nc < 0 || nc >= ff.n) {
                    continue;
                }

                const std::uint32_t next = static_cast<std::uint32_t>(nr * ff.n + nc);
                if (g.weights[next] == 0) {
                    continue;
                }

                const std::uint32_t new_dist = d + step_cost;
                if (new_dist < FloodField::unpack_dist(ff.cells[next])) {
                    ff.cells[next] = FloodField::pack(new_dist, flow_to_curr[i]);
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