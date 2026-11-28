#ifndef SERIAL_DIJKSTRA_H
#define SERIAL_DIJKSTRA_H

#include "graph.hpp"
#include <cstdint>
#include <limits>
#include <vector>

enum class ALGORITHM { DIJKSTRA, DIAL };

struct FloodField {
    // Packed 32-bit cell layout: [29-bit distance | 3-bit direction]
    // Direction encoding: 0 = Up, 1 = Right, 2 = Down, 3 = Left, 7 = None (Target / Obstacle / Unreachable)
    static constexpr std::uint32_t DIR_MASK   = 0x7u;
    static constexpr std::uint32_t DIR_NONE   = 0x7u;
    static constexpr std::uint32_t INF_DIST   = 0x1FFFFFFFu; // 536,870,911
    static constexpr std::uint32_t INF_PACKED = (INF_DIST << 3) | DIR_NONE;

    static constexpr std::uint32_t pack(std::uint32_t dist, std::int8_t dir) noexcept {
        const std::uint32_t d_bits = (dir < 0) ? DIR_NONE : (static_cast<std::uint32_t>(dir) & 0x3u);
        return (dist << 3) | d_bits;
    }

    static constexpr std::uint32_t unpack_dist(std::uint32_t packed) noexcept {
        return packed >> 3;
    }

    static constexpr std::int8_t unpack_dir(std::uint32_t packed) noexcept {
        const std::uint32_t d_bits = packed & DIR_MASK;
        return (d_bits == DIR_NONE) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(d_bits);
    }

    std::uint32_t dist_at(std::int64_t idx) const noexcept {
        return unpack_dist(cells[idx]);
    }

    std::int8_t dir_at(std::int64_t idx) const noexcept {
        return unpack_dir(cells[idx]);
    }

    // Single bit-packed array replacing separate `dist` (4B) and `dir` (1B) arrays
    std::vector<std::uint32_t> cells;

    // Populated only by traced solvers: elapsed nanoseconds when cell i was settled (-1 if unset)
    std::vector<std::int64_t> settle_ns;

    std::int64_t n = 0;
    std::int64_t reachable_cells = 0;
};

// Pure serial baseline (zero logging overhead for accurate timing)
FloodField compute_flood_field_serial(const Graph& g, ALGORITHM a);

// Traced serial solver (records per-cell nanosecond settle timestamps for animation)
FloodField compute_flood_field_serial_traced(const Graph& g, ALGORITHM a);

#endif