# Parallel Flood Field Generation

Benchmarking serial vs. parallel single-source shortest path algorithms (Dijkstra, Dial's, OpenMP Delta-Stepping, and MPI Delta-Stepping) for 2D grid flood field generation.

---

## 1. One-Time Setup

### System Dependencies (OpenMP & MPI)

**macOS (via Homebrew):**
```bash
brew install libomp open-mpi

```

**Ubuntu / Debian / WSL:**

```bash
sudo apt update && sudo apt install -y build-essential libomp-dev openmpi-bin libopenmpi-dev

```

### Python Virtual Environment (For Visualizations & Animations)

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

```

*(Re-activate in new terminal sessions with `source .venv/bin/activate`.)*

---

## 2. Build the Benchmark Driver

The included `Makefile` automatically configures OpenMP flags for both macOS and Linux.

**Build OpenMP + Serial Solvers (Recommended default):**

```bash
make

```

**Build OpenMP + MPI + Serial Solvers:**

```bash
make mpi

```

**Clean Build Artifacts:**

```bash
make clean

```

---

## 3. Pure Timing Benchmarks (No File Exports)

Arguments:
`./driver <n> <w_max> <k> <obstacle_density> [seed] [--dijkstra | --dial | --omp | --mpi] [--delta=<val>] [--threads=<val>] [--no-verify]`

**Run Serial Dijkstra (`2000x2000`):**

```bash
./driver 2000 100 0 0.25 --dijkstra

```

**Run Serial Dial's (`2000x2000`):**

```bash
./driver 2000 100 0 0.25 --dial

```

**Run OpenMP Delta-Stepping (`2000x2000`, `delta=40`, `8` threads):**

```bash
./driver 2000 100 0 0.25 --omp --delta=40 --threads=8

```

**Run MPI Delta-Stepping (`2000x2000`, `delta=40`, `4` ranks — requires `make mpi`):**

```bash
mpirun -np 4 ./driver 2000 100 0 0.25 --mpi --delta=40

```

**Sweep Bucket Size ($\Delta$) & Thread Counts:**

```bash
for d in 5 15 30 50 100 200; do
  ./driver 2000 100 0 0.25 --omp --delta=$d --threads=8
done

```

---

## 4. Static Flood Field Images (`output/*.png`)

**Generate Dijkstra static image:**

```bash
./driver 60 100 0 0.20 --export --dijkstra && python visualize.py --title="Serial Dijkstra" --output=output/flood_field_base_dijkstra.png

```

**Generate Dial's static image:**

```bash
./driver 60 100 0 0.20 --export --dial && python visualize.py --title="Serial Dial's" --output=output/flood_field_base_dial.png

```

**Generate OpenMP Delta-Stepping static image:**

```bash
./driver 60 100 0 0.20 --export --omp --delta=30 --threads=4 && python visualize.py --title="OpenMP Delta-Stepping (delta=30)" --output=output/flood_field_omp.png

```

---

## 5. Propagation Animations (`output/*.gif` or `.mp4`)

Trace CSVs are automatically named with their backend, $\Delta$, and worker count (e.g., `trace_omp_d30_t8.csv` or `trace_mpi_d30_r4.csv`) so multiple runs do not overwrite each other.

**Generate OpenMP GIF for a specific $\Delta$ and thread count:**

```bash
./driver 80 100 0 0.20 --trace --omp --delta=30 --threads=8 && python animate.py --trace=output/trace_omp_d30_t8.csv --title="OpenMP Delta-Stepping (delta=30, 8T)" --output=output/omp_d30_t8.gif

```

**Compare two $\Delta$ values side-by-side (`delta=10` vs `delta=50`):**

```bash
./driver 120 100 0 0.20 --trace --omp --delta=10 --threads=8 && \
./driver 120 100 0 0.20 --trace --omp --delta=50 --threads=8 && \
python animate.py \
  --trace=output/trace_omp_d10_t8.csv --title="OpenMP (delta=10, 8T)" \
  --compare=output/trace_omp_d50_t8.csv --compare-title="OpenMP (delta=50, 8T)" \
  --output=output/omp_delta_comparison.mp4 --frames=150 --fps=30

```

**Compare Serial Dial's vs. Parallel OpenMP side-by-side:**

```bash
./driver 120 100 0 0.20 --trace --dial && \
./driver 120 100 0 0.20 --trace --omp --delta=40 --threads=8 && \
python animate.py \
  --trace=output/trace_dial.csv --title="Serial Dial's" \
  --compare=output/trace_omp_d40_t8.csv --compare-title="OpenMP Delta-Stepping (8T)" \
  --output=output/dial_vs_omp.mp4 --frames=150 --fps=30

```

```

*(Quick verification after running `brew install libomp open-mpi` and `make`: run `./driver 500 100 0 0.20 --omp --threads=4` and confirm the output prints `Workers        : 4` and `Verification   : PASS`.)*

```