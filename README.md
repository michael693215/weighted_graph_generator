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
./driver 1000 100 0 0.25 42

```

### 3. Export CSVs & Generate Visualization

Install Python dependencies (one-time):

```bash
pip install numpy matplotlib

```

Run with `--export` (smaller grids like `60x60` render crisp flow arrows) and generate `flood_field.png`:

```bash
./driver 60 100 0 0.20 42 --export && python3 visualize.py

```

Optional visualization flags:

```bash
python3 visualize.py --out custom_name.png --no-arrows

```

---

## Original Graph Generator Tests

To run the standalone grid generator tests in `main.cpp`, uncomment a test block in `main.cpp` and run:

```bash
g++ -std=c++17 -O2 -o exec main.cpp graph.cpp && ./exec

```

## Project Layout

* `graph.hpp` / `graph.cpp` — Deterministic 2D weighted grid generator with obstacle masking.
* `rng.hpp` — Fast 64-bit Mersenne Twister wrapper.
* `serial_dijkstra.hpp` / `serial_dijkstra.cpp` — Serial Dijkstra flood field baseline.
* `driver.cpp` — Benchmark runner and CSV exporter.
* `visualize.py` — Generates `flood_field.png` from exported CSVs.

```
