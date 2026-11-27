# Parallel Flood Field Generation

Benchmarking serial vs. parallel single-source shortest path algorithms (Dijkstra, Dial's, and Delta-Stepping) for 2D grid flood field generation.

---

## Setup

### 1. Compile the Benchmark Driver
```bash
g++ -std=c++17 -O3 -o driver driver.cpp serial_dijkstra.cpp graph.cpp

```

### 2. Set Up Python Virtual Environment (One-Time)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

```

*(Re-activate in new terminal sessions with `source .venv/bin/activate`.)*

---

## Pure Timing Benchmarks (No File Exports)

Arguments: `./driver <n> <w_max> <k> <obstacle_density> [--dijkstra | --dial] [seed]`

**Run Serial Dijkstra (`2000x2000`):**

```bash
./driver 2000 100 0 0.25 --dijkstra

```

**Run Serial Dial's (`2000x2000`):**

```bash
./driver 2000 100 0 0.25 --dial

```

---

## Static Flood Field Images (`output/*.png`)

**Generate Dijkstra static image:**

```bash
./driver 60 100 0 0.20 --export --dijkstra && python visualize.py --title="Serial Dijkstra" --output=output/flood_field_base_dijkstra.png

```

**Generate Dial's static image:**

```bash
./driver 60 100 0 0.20 --export --dial && python visualize.py --title="Serial Dial's" --output=output/flood_field_base_dial.png

```

---

## Propagation Animations (`output/*.gif` or `.mp4`)

**Generate Dijkstra GIF:**

```bash
./driver 60 100 0 0.20 --trace --dijkstra && python animate.py --trace=output/trace_dijkstra.csv --title="Serial Dijkstra" --output=output/flood_field_base_dijkstra.gif

```

**Generate Dial's GIF:**

```bash
./driver 60 100 0 0.20 --trace --dial && python animate.py --trace=output/trace_dial.csv --title="Serial Dial's" --output=output/flood_field_base_dial.gif

```

**Generate Side-by-Side Comparison (`Dijkstra` vs `Dial's`) as MP4:**

```bash
./driver 150 100 0 0.20 --trace --dijkstra && ./driver 150 100 0 0.20 --trace --dial && python animate.py --trace=output/trace_dijkstra.csv --title="Serial Dijkstra" --compare=output/trace_dial.csv --compare-title="Serial Dial's" --output=output/dijkstra_vs_dial.mp4 --frames=150 --fps=30

```

---

## Original Graph Generator Tests

Uncomment a test block in `main.cpp` and run:

```bash
g++ -std=c++17 -O2 -o exec main.cpp graph.cpp && ./exec

```

```

```