# Rust CPU Shower Particle Simulation

A real-time particle shower simulation implemented in **Rust**, focused on CPU multithreading, collision detection, spatial partitioning and performance profiling.

The simulation models particles emitted from a shower head into a 3D cubicle. Particles are affected by gravity, reflect from the cubicle walls, cool over time, merge when they collide and respawn when they reach the floor.

![Rust Shower Simulation - Temperature Mode](docs/images/Temperature%20RUST.gif)

---

## What I Implemented

- Built the particle simulation and supporting data structures in Rust
- Implemented gravity, movement and reflective wall collisions
- Generated particles from a circular shower-head emitter with randomized downward velocities
- Implemented floor detection and particle respawning
- Implemented temperature decay where higher-mass particles cool more slowly
- Implemented mass- and momentum-preserving particle merging
- Parallelised physics, collision and temperature stages using dedicated thread pools
- Implemented a spatial grid to reduce collision checks
- Added bounded neighbouring-cell searches based on the collision radius
- Reused collision/grid allocations between frames to reduce allocation overhead
- Used atomic particle states to coordinate threaded collision processing
- Implemented real-time OpenGL rendering using `glium`
- Added temperature and mass visualisation modes
- Added stage-level profiling for simulation and rendering performance

---

## Simulation Architecture

The simulation is divided into three main CPU stages:

1. **Physics** – updates particle velocity and position, applies gravity, handles wall reflections and respawns particles that reach the floor.
2. **Collision detection and merging** – builds a spatial grid, assigns occupied cells to collision workers and resolves nearby particle collisions.
3. **Temperature** – reduces particle temperature over time according to particle mass.

Each stage uses its own two-thread `scoped_threadpool` pool. The stages run in sequence, while the work inside each stage is partitioned across its worker threads.

The visualisation currently starts with:

```text
100,000 particles
2 physics workers
2 collision workers
2 temperature workers
60 Hz simulation timestep
```

---

## Particle Simulation

The cubicle represents a **1 m × 2 m × 1 m** space.

Particles are emitted from a circular shower head with a radius of **0.05 m**, positioned at the centre of the ceiling.

Each new particle begins with:

- Mass: `1.0`
- Temperature: `1.0`
- Initial speed: `2.0 m/s`
- Direction: randomized within a **30° downward cone**

Gravity is applied at **-9.81 m/s²**.

Particles reflect when they reach the side walls of the cubicle. When a particle reaches the floor, it is immediately respawned at the shower head with a new randomized position and velocity.

---

## Spatial Grid

Checking every particle against every other particle would become prohibitively expensive as particle counts increase.

To reduce the collision search space, the simulation divides the cubicle into a spatial grid using a cell size of:

```text
0.005 m
```

Each occupied grid cell stores the indices of the particles currently inside it.

The visual simulation uses a reusable grid structure that retains its `HashMap`, bucket vectors and occupied-cell storage between frames. Instead of rebuilding all of these allocations from scratch every update, the existing storage is cleared and reused.

This significantly reduces allocation overhead during grid construction.

---

## Collision Detection and Merging

Collision detection uses squared distances to avoid unnecessary square-root operations.

For each particle, the simulation calculates the minimum and maximum grid coordinates intersected by its collision radius. Only cells inside these bounds are searched.

This replaces a fixed neighbouring-cell search with a search bounded by the actual collision area.

When two particles collide, they merge into a single particle.

The merged particle preserves total mass and calculates its position, velocity and temperature using mass-weighted values. This preserves momentum across the merge.

### Threaded Collision Processing

Occupied grid cells are distributed between collision workers.

Particle state is coordinated using atomic values representing states such as:

```text
ACTIVE
LOCKED
INACTIVE
SURVIVOR
```

Atomic compare-and-exchange operations prevent multiple workers from consuming the same particle during a merge.

Output particle slots are synchronized separately and the surviving particles are compacted into a reusable buffer at the end of the collision stage.

---

## Temperature

Each particle begins at a temperature of `1.0` and cools toward `0.0`.

Cooling is inversely proportional to particle mass:

```text
temperature decrease ∝ 1 / mass
```

As a result, particles that have gained mass through merging cool more slowly than standard particles.

---

## Rendering

The simulation is visualised in 3D using **glium/OpenGL**.

Only a sampled subset of the particle population is rendered so that visualisation does not dominate simulation performance. The current visual mode uses a render stride of `100`.

Static cubicle and shower-head geometry is uploaded once, while particle rendering uses a reusable dynamic vertex buffer.

Two visualisation modes are available.

### Temperature Mode

Particle colour represents temperature:

- **Red** – hotter particles
- **Blue** – colder particles

![Rust Shower Simulation - Temperature Mode](docs/images/Temperature%20RUST.gif)

### Mass Mode

Particle colour represents mass:

- **Blue** – standard / low-mass particles
- **Green** – particles that have increased in mass
- **Yellow** – larger merged particles
- **Red** – highest-mass particles

Higher-mass merged particles are prioritised when selecting the sampled particles for rendering, making the results of collision merging easier to observe.

![Rust Shower Simulation - Mass Mode](docs/images/Mass%20RUST.gif)

---

## Performance Optimisation

Several optimisation passes were applied after profiling the simulation.

### Smaller Spatial Cells

The spatial grid cell size was reduced to better match the small collision radius.

This reduces the number of particles grouped into each grid cell and therefore reduces unnecessary candidate collision checks.

### Collision-Bounded Cell Search

Instead of checking a fixed neighbourhood around each particle, the simulation calculates the exact range of cells intersected by the particle's collision radius.

This substantially reduces the number of grid buckets visited during collision detection.

### Reusable Grid and Collision Storage

The visual simulation reuses:

- the spatial-grid `HashMap`
- per-cell particle buckets
- occupied-cell storage
- per-thread owned-cell vectors
- atomic particle-state storage
- collision output slots
- compacted particle storage

This avoids repeatedly allocating large temporary structures every frame.

### Reusable Rendering Buffers

Static cubicle and shower-head vertex buffers are created once.

Particle rendering uses a preallocated dynamic vertex buffer, and only the currently used slice is updated each frame.

---

## Profiling

The visual simulation includes stage-level profiling.

Every 300 frames it reports mean and maximum timings for:

- Physics
- Grid construction
- Collision / merge
- Temperature
- Total simulation update
- Render-vertex preparation
- Particle-buffer upload
- Rendering and presentation
- Whole-frame time

This profiling was used to identify the grid and collision stages as the main optimisation targets.

### 500,000-Particle Benchmark

Performance profiling was carried out on an **Intel Core i5-13600KF** using a **500,000-particle initial stress test** in a release build.

The threaded simulation used:

- 2 physics workers
- 2 collision workers
- 2 temperature workers
- `0.005 m` spatial-grid cells
- `0.0001 m` collision radius
- `1 / 60 s` simulation timestep
- `-9.81 m/s²` gravity
- render stride of `100`

The final implementation was measured across three separate runs using the same configuration. The table below reports the mean timing over **frames 1-300** for each run.

| Metric | Run 1 | Run 2 | Run 3 | 3-Run Average |
|---|---:|---:|---:|---:|
| Physics | 0.521 ms | 0.484 ms | 0.600 ms | **0.535 ms** |
| Grid construction | 12.799 ms | 12.429 ms | 15.718 ms | **13.648 ms** |
| Collision / merge | 26.064 ms | 25.348 ms | 29.922 ms | **27.112 ms** |
| Temperature | 0.280 ms | 0.274 ms | 0.340 ms | **0.298 ms** |
| Simulation update | 39.711 ms | 38.581 ms | 46.636 ms | **41.642 ms** |
| Whole frame | 39.890 ms | 38.761 ms | 46.836 ms | **41.829 ms** |

Because particles merge during the simulation, **500,000 is the initial particle count** rather than a constant active-particle count throughout the profiling window.

The three-run average is used as the headline result rather than selecting the fastest run. This gives a more representative measurement and makes the benchmark directly comparable with the separate CUDA implementation, which uses the same initial particle count and core simulation parameters.

### Example Optimisation Result

A representative **100,000-particle** profiling window over frames 1–300 showed the effect of reusing grid and collision allocations:

| Metric | Before allocation reuse | After allocation reuse |
|---|---:|---:|
| Grid construction | 5.18 ms | **2.77 ms** |
| Collision / merge | 5.70 ms | **5.29 ms** |
| Simulation update | 12.29 ms | **8.33 ms** |
| Whole frame | 12.43 ms | **8.48 ms** |

This reduced average grid-construction time by approximately **47%** and average simulation-update time by approximately **32%** in that profiling window.

These values are development measurements intended to demonstrate the effect of the optimisation and should not be treated as hardware-independent benchmark results.

---

## Controls

| Key | Action |
|---|---|
| `T` | Temperature visualisation |
| `M` | Mass visualisation |

Close the application window to exit.

---

## Technologies

- **Rust**
- **glium / OpenGL**
- **cgmath**
- **scoped_threadpool**
- **rand**
- Rust atomics and synchronization primitives
- Cargo workspaces

---

## Project Structure

```text
RustShowerSimulation/
├── Cargo.toml
├── Cargo.lock
└── cpu_version/
    ├── Cargo.toml
    └── src/
        ├── main.rs
        ├── simulation.rs
        └── renderer.rs
```

### `main.rs`

Contains the real-time visualisation loop, threaded simulation stages, collision-worker coordination, reusable scratch data, profiling and benchmark entry points.

### `simulation.rs`

Contains the particle model, movement, temperature behaviour, sequential collision implementation, spatial-grid helpers, particle merging and emitter generation.

### `renderer.rs`

Contains rendering data conversion, cubicle and shower-head geometry, temperature colouring, mass colouring and sampled render-particle selection.

---

## Requirements

- **Rust toolchain** with support for the **2024 edition**
- **Cargo**
- **64-bit CPU**
- **OpenGL-capable GPU and drivers** for the real-time visualisation

### Rust Dependencies

- `glium 0.36.0`
- `cgmath 0.18.0`
- `rand 0.10.1`
- `scoped_threadpool 0.1.9`

---

## Building and Running

The project is organised as a Cargo workspace with the Rust implementation in `cpu_version`.

From the repository root:

```bash
cargo run --release -p cpu_version
```

The current source has visual mode enabled by default.

To run the built-in sequential/threaded benchmark path instead, set:

```rust
const VISUAL_MODE: bool = false;
```

in `cpu_version/src/main.rs` and rebuild in release mode.

---

## Configuration

The main simulation parameters can be adjusted in `cpu_version/src/main.rs` and `cpu_version/src/simulation.rs`, including:

- Particle count
- Physics / collision / temperature worker counts
- Simulation timestep
- Gravity
- Collision radius
- Spatial-grid cell size
- Cooling factor
- Render stride

---

## Limitations and Future Work

The current CPU implementation uses a hash-based spatial grid and synchronization around threaded collision processing. Further work could investigate alternative spatial-partitioning structures, different collision-work partitioning strategies and reduced synchronization overhead.

Additional benchmarking across different CPU architectures and thread counts would also help evaluate how effectively the implementation scales beyond the current worker configuration.

---

## Technical Areas

`Rust` `CPU Multithreading` `Concurrent Programming` `Particle Simulation` `Spatial Partitioning` `Collision Detection` `Atomics` `OpenGL` `Performance Profiling`
