#include <cstdint>
#include <random>

#ifndef RNG_H
#define RNG_H

class Rng {
public:
    explicit Rng(std::uint64_t seed) : g_(seed) {}

    std::uint64_t next() { return g_(); }

    float u01() {
        return (next() >> 40) * 0x1.0p-24f;
    }

    float uniform(float lo, float hi) {
        return lo + u01() * (hi - lo);
    }

    std::int64_t bounded(std::int64_t range) {
        return (std::int64_t)(((next() >> 32) * (std::uint64_t)range) >> 32);
    }

    std::int64_t uniform_int(std::int64_t lo, std::int64_t hi) {
        return lo + bounded(hi - lo + 1);
    }

private:
    std::mt19937_64 g_;
};
#endif