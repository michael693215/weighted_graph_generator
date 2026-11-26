# Parallel Flood Field Generation

Benchmarking serial vs. parallel single-source shortest path algorithms (Dijkstra, Dial's, and Delta-Stepping) for 2D grid flood field generation.

## Quick Start

### 1. Compile the Benchmark Driver
```bash
g++ -std=c++17 -O3 -o driver driver.cpp serial_dijkstra.cpp graph.cpp

```

### 2. Run a Benchmark

Default run (`500x500` grid, weights `1..100`, `20%` obstacles, seed `42`):

```bash
./driver

```

Custom parameters (`./driver <n> <w_max> <k> <obstacle_density> <seed>`):

```bash
./driver 2000 100 0 0.25 42

```

### 3. Set Up Python Virtual Environment (One-Time)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

```

*(Re-activate in new terminal sessions with `source .venv/bin/activate`.)*

### 4. Generate Static Flood Field Image (`flood_field.png`)

```bash
./driver 60 100 0 0.20 42 --export --[dijkstra | dial] && python visualize.py

```

### 5. Generate Propagation Animation (`flood_animation.gif` or `.mp4`)

Export the nanosecond settle trace with `--trace` and render the animation:

```bash
./driver 60 100 0 0.20 42 --trace --[dijkstra | dial] && python animate.py

```

Render as an `.mp4` or adjust frame count / FPS:

```bash
python animate.py --out flood_animation.mp4 --frames 150 --fps 30

```

Side-by-side comparison (once `trace_parallel.csv` is generated):

```bash
python animate.py --parallel trace_parallel.csv --out comparison.mp4

```

---

## Original Graph Generator Tests

Uncomment a test block in `main.cpp` and run:

```bash
g++ -std=c++17 -O2 -o exec main.cpp graph.cpp && ./exec

```