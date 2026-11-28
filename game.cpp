#include "game.hpp"
#include "zombie_sim.hpp"
#include <raylib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int WIN_PX = 900;
constexpr int MAX_VIEW = 60;

struct Published {
    FloodField ff;
    std::int64_t target = -1;
};

// Computes flood fields on its own thread, on its own copy of the graph.
// The main thread never touches work_, so there are no data races on weights.
class FieldWorker {
public:
    FieldWorker(const Graph& g, const GameConfig& cfg, std::int64_t initial_target)
        : work_(g), cfg_(cfg) {
        auto pub = std::make_shared<Published>();
        pub->target = initial_target;
        pub->ff = compute(initial_target);
        published_ = std::move(pub);
        th_ = std::thread(&FieldWorker::loop, this);
    }

    ~FieldWorker() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_one();
        th_.join(); // waits for any in-flight compute to finish
    }

    // Requests coalesce: only the newest target is computed.
    void request(std::int64_t target) {
        {
            std::lock_guard<std::mutex> lk(m_);
            req_target_ = target;
            has_req_ = true;
        }
        cv_.notify_one();
    }

    std::shared_ptr<const Published> latest() const {
        std::lock_guard<std::mutex> lk(m_);
        return published_;
    }

    double last_ms() const { return last_ms_.load(); }
    std::uint64_t recomputes() const { return recomputes_.load(); }

private:
    FloodField compute(std::int64_t t) {
        work_.p.t = t;
        if (cfg_.use_parallel) {
            ParallelResult r = compute_flood_field_parallel(work_, cfg_.parallel);
            return std::move(r.ff);
        }
        return compute_flood_field_serial(work_, cfg_.serial_algo);
    }

    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        while (true) {
            cv_.wait(lk, [&] { return stop_ || has_req_; });
            if (stop_) return;
            const std::int64_t t = req_target_;
            has_req_ = false;
            lk.unlock();

            const auto t0 = std::chrono::steady_clock::now();
            auto pub = std::make_shared<Published>();
            pub->target = t;
            pub->ff = compute(t);
            const auto t1 = std::chrono::steady_clock::now();
            last_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++recomputes_;

            lk.lock();
            published_ = std::move(pub);
        }
    }

    Graph work_;
    GameConfig cfg_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool has_req_ = false;
    std::int64_t req_target_ = -1;
    std::shared_ptr<const Published> published_;
    std::atomic<double> last_ms_{0.0};
    std::atomic<std::uint64_t> recomputes_{0};
    std::thread th_;
};

struct FrameStats {
    static constexpr std::size_t RING = 1000;
    std::vector<float> ring = std::vector<float>(RING, 0.0f);
    std::size_t head = 0, filled = 0;
    std::uint64_t frames = 0;
    double total = 0.0;
    float best = 1e9f, worst = 0.0f;
    double avg_fps = 0, peak_fps = 0, low1_fps = 0, worst_ms = 0;

    void push(float dt) {
        if (dt <= 0.0f) return;
        ring[head] = dt;
        head = (head + 1) % RING;
        filled = std::min(filled + 1, RING);
        ++frames;
        total += dt;
        best = std::min(best, dt);
        worst = std::max(worst, dt);
        if (frames % 30 == 0) refresh();
    }

    void refresh() {
        avg_fps = frames / total;
        peak_fps = 1.0 / best;
        worst_ms = worst * 1000.0;
        std::vector<float> v(ring.begin(), ring.begin() + filled);
        std::sort(v.begin(), v.end(), std::greater<float>());
        const std::size_t k = std::max<std::size_t>(1, v.size() / 100);
        double sum = 0;
        for (std::size_t i = 0; i < k; ++i) sum += v[i];
        low1_fps = k / sum; // 1% low = mean of the worst 1% of frames
    }
};

std::vector<std::int64_t> place_portals(Graph& g, std::int64_t size, int count, std::int64_t target,
                                        std::uint64_t seed, std::vector<std::uint8_t>& mask) {
    std::vector<std::int64_t> out;
    std::uint64_t s = (seed ^ 0xD1B54A32D192ED03ULL) | 1ULL;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    count = static_cast<int>(std::min<std::int64_t>(count, size - 1));
    for (std::int64_t attempts = 0; static_cast<int>(out.size()) < count && attempts < 1000LL * (count + 1); ++attempts) {
        const std::int64_t idx = static_cast<std::int64_t>(rnd() % static_cast<std::uint64_t>(size));
        if (idx == target || mask[idx]) continue;
        if (g.weights[idx] == 0) g.weights[idx] = 1; // portals may replace obstacles; must stay passable
        mask[idx] = 1;
        out.push_back(idx);
    }
    return out;
}

} // namespace

int run_game(Graph& g, const GameConfig& cfg) {
    const std::int64_t size = g.p.size;
    const std::int64_t n = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(size))));
    const std::int64_t view = std::min<std::int64_t>(n, MAX_VIEW);
    const float cell_px = static_cast<float>(WIN_PX) / static_cast<float>(view);

    std::int64_t target = g.p.t;
    std::vector<std::uint8_t> portal_mask(size, 0);
    std::vector<std::int64_t> portals = place_portals(g, size, cfg.num_portals, target, cfg.seed, portal_mask);

    int wmax = 1;
    for (std::int64_t i = 0; i < size; ++i) wmax = std::max(wmax, static_cast<int>(g.weights[i]));

    // Portals are in the weights now, so the worker's graph copy sees them.
    FieldWorker worker(g, cfg, target);
    ZombieSim sim(n, cfg.max_zombies, portals, cfg.seed);

    InitWindow(WIN_PX, WIN_PX, "Zombies");
    SetTargetFPS(cfg.fps_cap);

    FrameStats stats;
    std::vector<std::uint16_t> counts(static_cast<std::size_t>(view * view));
    std::int64_t requested = target;
    double acc = 0.0, move_cd = 0.0;
    const double tick_dt = 1.0 / std::max(1, cfg.tick_hz);

    while (!WindowShouldClose()) {
        const float dt = GetFrameTime();
        stats.push(dt);

        // --- target input (WASD, cooldown while held) ---
        move_cd -= dt;
        if (move_cd <= 0.0) {
            int dr = 0, dc = 0;
            if (IsKeyDown(KEY_W)) dr = -1;
            else if (IsKeyDown(KEY_S)) dr = 1;
            else if (IsKeyDown(KEY_A)) dc = -1;
            else if (IsKeyDown(KEY_D)) dc = 1;
            if (dr || dc) {
                const std::int64_t nr = target / n + dr, nc = target % n + dc;
                if (nr >= 0 && nr < n && nc >= 0 && nc < n) {
                    const std::int64_t idx = nr * n + nc;
                    if (g.weights[idx] != 0 && !portal_mask[idx]) target = idx;
                }
                move_cd = 0.08;
            }
        }
        if (target != requested) { // per-frame change check
            worker.request(target);
            requested = target;
        }

        // --- fixed-timestep simulation ---
        const auto snap = worker.latest();
        acc = std::min(acc + dt, 0.25);
        while (acc >= tick_dt) {
            sim.tick(snap->ff, snap->target, target, cfg.spawns_per_tick);
            acc -= tick_dt;
        }

        // --- viewport: centered on target, clamped to the grid ---
        const std::int64_t vr0 = std::clamp<std::int64_t>(target / n - view / 2, 0, n - view);
        const std::int64_t vc0 = std::clamp<std::int64_t>(target % n - view / 2, 0, n - view);

        std::fill(counts.begin(), counts.end(), 0);
        for (const std::int64_t p : sim.positions()) {
            if (p < 0) continue;
            const std::int64_t r = p / n, c = p % n;
            if (r >= vr0 && r < vr0 + view && c >= vc0 && c < vc0 + view) {
                auto& x = counts[(r - vr0) * view + (c - vc0)];
                if (x < 65535) ++x;
            }
        }

        // --- draw ---
        BeginDrawing();
        ClearBackground(BLACK);
        for (std::int64_t vr = 0; vr < view; ++vr) {
            for (std::int64_t vc = 0; vc < view; ++vc) {
                const std::int64_t idx = (vr0 + vr) * n + (vc0 + vc);
                const int x0 = static_cast<int>(vc * cell_px), x1 = static_cast<int>((vc + 1) * cell_px);
                const int y0 = static_cast<int>(vr * cell_px), y1 = static_cast<int>((vr + 1) * cell_px);

                Color col;
                const int w = static_cast<int>(g.weights[idx]);
                if (idx == target) col = GOLD;
                else if (portal_mask[idx]) col = PURPLE;
                else if (w == 0) col = Color{25, 25, 25, 255};
                else {
                    const unsigned char v = static_cast<unsigned char>(70 + 130 * w / wmax);
                    col = Color{155, 155, 155, 255};
                }
                DrawRectangle(x0, y0, x1 - x0 - 1, y1 - y0 - 1, col);

                const std::uint16_t cnt = counts[vr * view + vc];
                if (cnt > 0) {
                    DrawCircle((x0 + x1) / 2, (y0 + y1) / 2, cell_px * 0.38f, RED);
                    if (cnt >= 2) {
                        char buf[8];
                        std::snprintf(buf, sizeof buf, "%u", cnt);
                        const int fs = std::max(10, static_cast<int>(cell_px * 0.45f));
                        const int tw = MeasureText(buf, fs);
                        DrawText(buf, (x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - fs / 2, fs, WHITE);
                    }
                }
            }
        }

        char l[4][128];
        std::snprintf(l[0], 128, "FPS avg %.0f  peak %.0f  1%% low %.0f  worst %.1f ms",
                      stats.avg_fps, stats.peak_fps, stats.low1_fps, stats.worst_ms);
        std::snprintf(l[1], 128, "%s | field %.2f ms | recomputes %llu", cfg.algo_label.c_str(),
                      worker.last_ms(), static_cast<unsigned long long>(worker.recomputes()));
        std::snprintf(l[2], 128, "Zombies %d / %d | Portals %d", sim.alive(), sim.cap(), sim.portal_count());
        std::snprintf(l[3], 128, "Target (%lld, %lld)%s", static_cast<long long>(target / n),
                      static_cast<long long>(target % n), snap->target == target ? "" : "  [field stale]");
        DrawRectangle(0, 0, 460, 4 * 20 + 10, Fade(BLACK, 0.65f));
        for (int i = 0; i < 4; ++i) DrawText(l[i], 8, 6 + i * 20, 16, RAYWHITE);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}