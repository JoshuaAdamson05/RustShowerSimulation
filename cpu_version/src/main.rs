use std::{time::Instant};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rand::{random_range};

type Cell = (i32, i32);
type Grid = HashMap<Cell, Vec<usize>>;

const WIDTH: f32 = 1.0;
const HEIGHT: f32 = 2.0;
const EMITTER_RADIUS: f32 = 0.05;

fn main() {
    const DEBUG_LOGGING: bool = true;
    const NUM_PARTICLES: usize = 100_000;
    const NUM_STEPS: usize = 100_000;
    const NUM_RUNS: usize = 3;
    const NUM_THREADS: usize = 2;
    const GRAVITY: f32 = -9.81;
    const DELTA_TIME: f32 = 1.0 / 60.0;
    const COLLISION_RADIUS: f32 = 0.00001;

    let mut ps = ParticleSystem::new();

    for _ in 0..NUM_PARTICLES{
        let (x, y) = random_emitter_position();
        let vx: f32 = random_range(-0.05..=0.05);
        let vy: f32 = random_range(-2.0..=-1.0);
        let p = Particle::new(x, y, vx, vy);
        ps.particles.push(p);
    }

    let ps_sequential = ps.clone();
    let ps_threaded = ps.clone();

    run_sequential_simulation(ps_sequential, NUM_STEPS, NUM_RUNS, DELTA_TIME, GRAVITY, COLLISION_RADIUS, DEBUG_LOGGING);
    run_threaded_simulation(ps_threaded, NUM_STEPS, NUM_THREADS, NUM_RUNS, DELTA_TIME, GRAVITY, COLLISION_RADIUS, DEBUG_LOGGING);
}

pub fn update_particle_chunk(particles : &mut [Particle], delta_time : f32, gravity : f32) -> usize {
    let mut local_floor_hits = 0;
    for particle in particles
    {
        particle.update_position(delta_time, gravity);

        if particle.y <= 0.0 {
            local_floor_hits += 1;
            ParticleSystem::respawn_particle(particle);
        }
    }
    local_floor_hits
}

pub fn resolve_collisions_chunk(particles : &[Particle], grid: &Grid, collision_radius: f32, start: usize, end: usize) -> Vec<Particle> {
    let mut merged = vec![false; particles.len()];
    let mut local_particles = Vec::new();

    for i in start..end {
        if merged[i] {
            continue;
        }

        let mut current = particles[i];
        let mut changed = true;

        while changed {
            changed = false;

            let cell = get_cell(&current, collision_radius * 2.0);

            for dx in -1..=1 {
                for dy in -1..=1 {
                    let neighbour_cell = (cell.0 + dx, cell.1 + dy);

                    if let Some(indices) = grid.get(&neighbour_cell) {
                        for &j in indices {
                            if j == i || merged[j] {
                                continue;
                            }

                            let dx = current.x - particles[j].x;
                            let dy = current.y - particles[j].y;

                            if dx * dx + dy * dy <= collision_radius * collision_radius {
                                current = ParticleSystem::merge(&current, &particles[j]);
                                merged[j] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        local_particles.push(current);
    }

    local_particles
}

fn run_threaded_simulation(ps: ParticleSystem, steps: usize, num_threads: usize, runs: usize, delta_time : f32, gravity : f32, collision_radius: f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();      
        let mut pool = scoped_threadpool::Pool::new(num_threads as u32);

        let start = Instant::now();

        for step in 0..steps {
            let chunk_size: usize = (ps_clone.particles.len() / num_threads).max(1);   
            let floor_results = Arc::new(Mutex::new(Vec::<usize>::new()));


            pool.scoped(|scope| {
                for slice in ps_clone.particles.chunks_mut(chunk_size) 
                {
                    let results = Arc::clone(&floor_results);
                    scope.execute(move || {
                        let hits = update_particle_chunk(slice, delta_time, gravity);
                        results.lock().unwrap().push(hits);
                    });
                }
            });

            //Combine results
            let total_hits: usize = floor_results.lock().unwrap().iter().sum();
            ps_clone.stats.floor_hits += total_hits;

            let old_count = ps_clone.particles.len();

            let grid = ps_clone.build_grid(collision_radius * 2.0);
            let collision_results = Arc::new(Mutex::new(Vec::<Vec<Particle>>::new()));
            
            let len = ps_clone.particles.len();
            let collision_chunk_size = (len / num_threads).max(1);

            pool.scoped(|scope| {
                for t in 0..num_threads {
                    let start = t * collision_chunk_size;
                    let end = ((t + 1) * collision_chunk_size).min(len);

                    if start >= len {
                        continue;
                    }

                    let particles = &ps_clone.particles;
                    let grid = &grid;
                    let results = Arc::clone(&collision_results);

                    scope.execute(move || {
                        let local = resolve_collisions_chunk(particles, grid, collision_radius, start, end);
                        results.lock().unwrap().push(local);
                    });
                }
            });

            let mut new_particles = Vec::new();

            for chunk in collision_results.lock().unwrap().iter() {
                new_particles.extend(chunk.iter().copied());
            }

            ps_clone.particles = new_particles;

            let local_merges = old_count - ps_clone.particles.len();
            ps_clone.stats.collision_count += local_merges;
            ps_clone.stats.merge_count += local_merges;

            // Cross-thread merge
            loop {
                let before = ps_clone.particles.len();
                ps_clone.resolve_collisions(collision_radius);
                let after = ps_clone.particles.len();

                if before == after {
                    break;
                }
            }

            ps_clone.update_temperature();

            if debug_logging && step % 20_000 == 0 {
                println!(
                    "Threaded Step: {}, Particles: {}, Floor Hits: {}, Collisions: {}, Merges: {}",
                    step, ps_clone.particles.len(), ps_clone.stats.floor_hits, ps_clone.stats.collision_count,
                    ps_clone.stats.merge_count
                );
            }
        }

        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }

    println!("Threaded avg time: {:.6} s", total / runs as f64);
}

fn run_sequential_simulation(ps: ParticleSystem, steps: usize, runs: usize, delta_time : f32, gravity : f32, collision_radius: f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();

        let start = Instant::now();
        ps_clone.simulate_particle_movement(steps, delta_time, gravity, collision_radius, debug_logging);
        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }
    
    println!("Sequential avg time: {:.6} s", total / runs as f64);
}

fn get_cell(p: &Particle, cell_size: f32) -> Cell {
    let x = (p.x / cell_size).floor() as i32;
    let y = (p.y / cell_size).floor() as i32;
    (x, y)
}

fn random_emitter_position() -> (f32, f32) {
    let radius = EMITTER_RADIUS;
    let centre_x = 0.5;
    let centre_y = 2.0;

    let r = radius * random_range(0.0..=1.0f32).sqrt();
    let theta = random_range(0.0..=std::f32::consts::TAU);

    let x = centre_x + r * theta.cos();
    let y = centre_y;

    (x, y)
}

#[derive(Debug, Copy, Clone)]
pub struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32
}

impl Particle
{
    pub fn new(x_variable: f32, y_variable: f32, vx_variable: f32, vy_variable: f32) -> Particle{
        Particle{
            x: x_variable,
            y: y_variable,
            vx: vx_variable,
            vy: vy_variable,
            mass: 1.0
        }
    }

    pub fn update_position(&mut self, delta_time: f32, gravity: f32)
    {
        if self.y > 0.0 {
            self.vy += gravity * delta_time;
        }

        self.x += self.vx * delta_time;

        self.y += self.vy * delta_time;

        if self.x <= 0.0 {
            self.x = 0.0;
            self.vx = -self.vx;
        }

        if self.x >= WIDTH {
            self.x = WIDTH;
            self.vx = -self.vx;
        }

        self.x = self.x.clamp(0.0, WIDTH);
        self.y = self.y.clamp(0.0, HEIGHT);
    }
}

#[derive(Clone, Debug)]
pub struct SimulationStats {
    floor_hits: usize,
    collision_count: usize,
    merge_count: usize
}

#[derive(Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
    stats: SimulationStats
}

impl ParticleSystem {
    pub fn new() -> ParticleSystem {
        ParticleSystem {
            particles: Vec::new(),
            stats: SimulationStats { floor_hits: 0, collision_count: 0, merge_count: 0 }
        }
    }

    pub fn update_physics(&mut self, delta_time : f32, gravity : f32)
    {
        for particle in &mut self.particles{
            particle.update_position(delta_time, gravity);

            if particle.y <= 0.0 {
                self.stats.floor_hits += 1;
                Self::respawn_particle(particle);
            }
        }
    }

    fn build_grid(&self, cell_size: f32) -> Grid {
        let mut grid: Grid = HashMap::new();

        for (i, particle) in self.particles.iter().enumerate() {
            let cell = get_cell(particle, cell_size);
            grid.entry(cell).or_default().push(i);
        }

        grid
    }

    fn are_colliding(a: &Particle, b: &Particle, radius: f32) -> bool {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        dx * dx + dy * dy <= radius * radius
    }

    fn merge(a: &Particle, b: &Particle) -> Particle {
        let mass = a.mass + b.mass;
        let vx = (a.mass * a.vx + b.vx * b.mass) / mass;
        let vy = (a.mass * a.vy + b.mass * b.vy) / mass;
        let x = (a.x * a.mass + b.x * b.mass) / mass;
        let y = (a.y * a.mass + b.y * b.mass) / mass;

        Particle { x, y, vx, vy, mass }
    }

    pub fn resolve_collisions(&mut self, collision_radius: f32) {
        let old_count = self.particles.len();

        let grid = self.build_grid(collision_radius * 2.0);
        let mut merged = vec![false; self.particles.len()];
        let mut new_particles: Vec<Particle> = Vec::new();

        for i in 0..self.particles.len() {
            if merged[i] {
                continue;
            }

            let mut current = self.particles[i];
            let mut changed = true;

            while changed {
                changed = false;    

                let cell = get_cell(&current, collision_radius * 2.0);

                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let neighbour_cell = (cell.0 + dx, cell.1 + dy);

                        if let Some(indices) = grid.get(&neighbour_cell) {
                            for &j in indices {
                                if j == i || merged[j] {
                                    continue;
                                }

                                if Self::are_colliding(&current, &self.particles[j], collision_radius)
                                {
                                    current = Self::merge(&current, &self.particles[j]);
                                    merged[j] = true;
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }

            new_particles.push(current);
        }
        self.particles = new_particles;

        let new_count = self.particles.len();
        let merges = old_count - new_count;

        self.stats.collision_count += merges;
        self.stats.merge_count += merges;
    }

    pub fn update_temperature(&mut self) {
        // TODO: cooling formula
    }

    pub fn simulate_particle_movement(&mut self, steps: usize, delta_time : f32, gravity : f32, collision_radius: f32, debug_logging : bool)
    {
        for step in 0..steps{
            self.update_physics(delta_time, gravity);
            self.resolve_collisions(collision_radius);
            self.update_temperature();

            if debug_logging && step % 20_000 == 0 {
                println!("Step: {}, Particles: {}, Floor Hits: {}, Collisions: {}, Merges: {}", 
                step, self.particles.len(), self.stats.floor_hits, self.stats.collision_count, 
                self.stats.merge_count);
            }
        }
    }

    pub fn respawn_particle(p: &mut Particle)
    {
        let (x, y) = random_emitter_position();
        p.x = x;
        p.y = y;

        p.vx = random_range(-0.05..=0.05);
        p.vy = random_range(-2.0..=-1.0);
        p.mass = 1.0;
    }
 }