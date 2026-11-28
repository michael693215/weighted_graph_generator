#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sys/resource.h>

struct ResourceStats {
    double wall_ns = 0.0;
    double user_ns = 0.0;
    double sys_ns = 0.0;
    std::uint64_t peak_rss_bytes = 0;

    double effective_cores() const {
        return (wall_ns > 0.0) ? ((user_ns + sys_ns) / wall_ns) : 0.0;
    }
    double cpu_efficiency_pct(int workers) const {
        return (workers > 0) ? (effective_cores() / static_cast<double>(workers)) * 100.0 : 0.0;
    }
    double peak_rss_mb() const {
        return static_cast<double>(peak_rss_bytes) / (1024.0 * 1024.0);
    }
};

struct ResourceSnapshot {
    double user_ns = 0.0;
    double sys_ns = 0.0;
};

inline std::uint64_t get_peak_rss_bytes() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return static_cast<std::uint64_t>(ru.ru_maxrss); // macOS reports bytes
#else
    return static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ULL; // Linux reports KiB
#endif
}

inline ResourceSnapshot capture_cpu_snapshot() {
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    auto to_ns = [](const struct timeval& tv) {
        return static_cast<double>(tv.tv_sec) * 1e9 + static_cast<double>(tv.tv_usec) * 1e3;
    };
    return {to_ns(ru.ru_utime), to_ns(ru.ru_stime)};
}

inline ResourceStats finalize_resource_stats(const ResourceSnapshot& start, double wall_ns) {
    const ResourceSnapshot end = capture_cpu_snapshot();
    ResourceStats s;
    s.wall_ns = wall_ns;
    s.user_ns = std::max(0.0, end.user_ns - start.user_ns);
    s.sys_ns = std::max(0.0, end.sys_ns - start.sys_ns);
    s.peak_rss_bytes = get_peak_rss_bytes();
    return s;
}

// O(1) extra-memory verification of BOTH Bellman distance optimality and packed flow directions
template <typename GraphT, typename FloodFieldT>
bool verify_bellman_optimality(const GraphT& g, const FloodFieldT& ff) {
    constexpr std::int64_t DR[4] = {-1, 0, 1, 0};
    constexpr std::int64_t DC[4] = {0, 1, 0, -1};
    const std::int64_t n = ff.n;
    const std::int64_t target = g.p.t;
    const std::uint32_t INF_DIST = FloodFieldT::INF_DIST;

    if (target < 0 || target >= g.p.size || g.weights[target] == 0) {
        return true;
    }
    if (ff.dist_at(target) != 0 || ff.dir_at(target) != -1) {
        return false;
    }

    for (std::int64_t u = 0; u < g.p.size; ++u) {
        const std::uint32_t du = ff.dist_at(u);
        const std::int8_t dir = ff.dir_at(u);

        if (g.weights[u] == 0) {
            if (du != INF_DIST || dir != -1) return false;
            continue;
        }
        if (u == target) {
            continue;
        }

        const std::int64_t r = u / n;
        const std::int64_t c = u % n;
        std::uint32_t best_neighbor = INF_DIST;

        for (int i = 0; i < 4; ++i) {
            const std::int64_t nr = r + DR[i];
            const std::int64_t nc = c + DC[i];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
            const std::int64_t v = nr * n + nc;
            const std::uint32_t dv = ff.dist_at(v);
            if (g.weights[v] == 0 || dv == INF_DIST) continue;

            const std::uint32_t cand = dv + static_cast<std::uint32_t>(g.weights[v]);
            if (cand < best_neighbor) {
                best_neighbor = cand;
            }
        }

        if (du != best_neighbor) {
            return false;
        }

        if (du == INF_DIST) {
            if (dir != -1) return false;
        } else {
            if (dir < 0 || dir > 3) return false;
            const std::int64_t nr = r + DR[dir];
            const std::int64_t nc = c + DC[dir];
            if (nr < 0 || nr >= n || nc < 0 || nc >= n) return false;
            const std::int64_t nb = nr * n + nc;
            const std::uint32_t dnb = ff.dist_at(nb);
            if (g.weights[nb] == 0 || dnb == INF_DIST) return false;
            if (dnb + static_cast<std::uint32_t>(g.weights[nb]) != du) return false;
        }
    }
    return true;
}