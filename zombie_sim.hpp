#include "serial_dijkstra.hpp" // FloodField
#include <cstdint>
#include <vector>

#ifndef ZOMBIE_SIM_H
#define ZOMBIE_SIM_H

// Pure simulation: no threads, no rendering, no I/O.
class ZombieSim {
public:
    ZombieSim(std::int64_t n, int max_zombies, std::vector<std::int64_t> portals, std::uint64_t seed)
        : n_(n), pos_(max_zombies, -1), flag_(max_zombies, 0),
          portals_(std::move(portals)),
          rng_(seed ^ 0x9E3779B97F4A7C15ULL) {
        if (rng_ == 0) rng_ = 1;
        free_.reserve(max_zombies);
        for (int i = max_zombies - 1; i >= 0; --i) free_.push_back(i);
    }

    // field_target: the target the field was built for. target: the current target.
    void tick(const FloodField& ff, std::int64_t field_target, std::int64_t target, int spawns) {
        // 1. Despawn zombies flagged at the end of the previous tick.
        for (std::size_t i = 0; i < pos_.size(); ++i) {
            if (flag_[i]) {
                pos_[i] = -1;
                flag_[i] = 0;
                free_.push_back(static_cast<int>(i));
            }
        }

        // 2. Spawn at random reachable portals while under the cap.
        for (int s = 0; s < spawns && !free_.empty() && !portals_.empty(); ++s) {
            const std::size_t start = next_rand() % portals_.size();
            std::int64_t chosen = -1;
            for (std::size_t k = 0; k < portals_.size(); ++k) {
                const std::int64_t p = portals_[(start + k) % portals_.size()];
                if (ff.dist_at(p) != FloodField::INF_DIST) { chosen = p; break; }
            }
            if (chosen < 0) break; // no portal can reach the target
            pos_[free_.back()] = chosen;
            free_.pop_back();
        }

        // 3. Move one cell along the flow field; flag finished/stuck zombies.
        // If the field is stale (target moved), zombies stuck on the old root just
        // wait instead of being flagged as stalled.
        const bool field_current = (field_target == target);
        for (std::size_t i = 0; i < pos_.size(); ++i) {
            const std::int64_t p = pos_[i];
            if (p < 0) continue;

            std::int64_t np = p;
            const int d = static_cast<int>(ff.dir_at(p));
            if (d >= 0) {
                const std::int64_t nr = p / n_ + DR[d];
                const std::int64_t nc = p % n_ + DC[d];
                if (nr >= 0 && nr < n_ && nc >= 0 && nc < n_) np = nr * n_ + nc;
            }
            pos_[i] = np;
            if (np == target || (np == p && field_current)) flag_[i] = 1;
        }
    }

    const std::vector<std::int64_t>& positions() const { return pos_; } // -1 = free slot
    int alive() const { return static_cast<int>(pos_.size() - free_.size()); }
    int cap() const { return static_cast<int>(pos_.size()); }
    int portal_count() const { return static_cast<int>(portals_.size()); }

private:
    static constexpr std::int64_t DR[4] = {-1, 0, 1, 0};
    static constexpr std::int64_t DC[4] = {0, 1, 0, -1};

    std::uint64_t next_rand() { // xorshift64
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 7;
        rng_ ^= rng_ << 17;
        return rng_;
    }

    std::int64_t n_;
    std::vector<std::int64_t> pos_;
    std::vector<std::uint8_t> flag_;
    std::vector<int> free_; // stack of free slot indices
    std::vector<std::int64_t> portals_;
    std::uint64_t rng_;
};

#endif