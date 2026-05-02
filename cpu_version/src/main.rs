use std::{time::Instant};
use std::sync::Arc;
use std::sync::Mutex;
use rand::{random_range};

fn main() {
    const DEBUG_LOGGING: bool = true;
    const NUM_PARTICLES: usize = 1_000;
    const NUM_STEPS: usize = 400_000;
    const NUM_RUNS: usize = 5;
    const PHYSICS_THREADS: usize = 2;
    const GRAVITY: f32 = -9.81;
    const DELTA_TIME: f32 = 1.0 / 60.0;
    let mut ps = ParticleSystem::new();

    for _ in 0..NUM_PARTICLES{
        let x: f32 = random_range(0.0..=10.0);
        let y: f32 = random_range(0.0..=10.0);
        let vx: f32 = random_range(-2.0..=2.0);
        let vy: f32 = random_range(-2.0..=2.0);
        let p = Particle::new(x, y, vx, vy);
        ps.particles.push(p);
    }

    let ps_sequential = ps.clone();
    let ps_threaded = ps.clone();

    run_sequential_simulation(ps_sequential, NUM_STEPS, NUM_RUNS, DELTA_TIME, GRAVITY, DEBUG_LOGGING);
    run_threaded_simulation(ps_threaded, NUM_STEPS, PHYSICS_THREADS, NUM_RUNS, DELTA_TIME, GRAVITY, DEBUG_LOGGING);
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

pub fn detect_collisions_chunk(particles : &[Particle], start: usize, end: usize) -> Vec<CollisionPair> {
    let mut collisions = Vec::new();

    for i in start..end {
        for j in (i + 1)..particles.len() {
            if ParticleSystem::are_colliding(&particles[i], &particles[j], 0.01) {
                collisions.push(CollisionPair { a: i, b: j});
            }
        }
    }
    collisions
}

fn run_threaded_simulation(ps: ParticleSystem, steps: usize, num_threads: usize, runs: usize, delta_time : f32, gravity : f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();      
        let mut pool = scoped_threadpool::Pool::new(num_threads as u32);

        let start = Instant::now();

        for step in 0..steps {
            let chunk_size: usize = (ps_clone.particles.len() / num_threads).max(1);   
            let results = Arc::new(Mutex::new(Vec::<usize>::new()));
            let collision_results = Arc::new(Mutex::new(Vec::<CollisionPair>::new()));
            let len = ps_clone.particles.len();
            pool.scoped(|scope| {
                for slice in ps_clone.particles.chunks_mut(chunk_size) 
                {
                    let results = Arc::clone(&results);
                    scope.execute(move || {
                        let hits = update_particle_chunk(slice, delta_time, gravity);
                        results.lock().unwrap().push(hits);
                    });
                }
            });

            pool.scoped(|scope| {
                for t in 0..num_threads {
                    let start = t * chunk_size;
                    let end = ((t + 1) * chunk_size).min(len);

                    if start >= len {
                        continue;
                    }

                    let particles = &ps_clone.particles;
                    let results = Arc::clone(&collision_results);

                    scope.execute(move || {
                        let local = detect_collisions_chunk(particles, start, end);
                        results.lock().unwrap().extend(local);
                    });
                }
            });

            //Combine results
            let total_hits: usize = results.lock().unwrap().iter().sum();
            ps_clone.stats.floor_hits += total_hits;

            let pairs = collision_results.lock().unwrap().clone();
            
            ps_clone.resolve_collisions_pairs(pairs);
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

fn run_sequential_simulation(ps: ParticleSystem, steps: usize, runs: usize, delta_time : f32, gravity : f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();

        let start = Instant::now();
        ps_clone.simulate_particle_movement(steps, delta_time, gravity, debug_logging);
        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }
    
    println!("Sequential avg time: {:.6} s", total / runs as f64);
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

        if self.x >= 10.0 {
            self.x = 10.0;
            self.vx = -self.vx;
        }

        self.x = self.x.clamp(0.0, 10.0);
        self.y = self.y.clamp(0.0, 10.0);
    }
}

#[derive(Clone, Debug)]
pub struct SimulationStats {
    floor_hits: usize,
    collision_count: usize,
    merge_count: usize
}

#[derive(Clone, Copy)]
pub struct CollisionPair {
    a: usize, 
    b: usize
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

    pub fn resolve_collisions(&mut self) {
        const COLLISION_RADIUS: f32 = 0.01;

        let mut merged = vec![false; self.particles.len()];
        let mut new_particles: Vec<Particle> = Vec::new();

        for i in 0..self.particles.len() {
            if merged[i] {
                continue;
            }

            let mut current = self.particles[i];

            for j in (i + 1)..self.particles.len() {
                if merged[j] {
                    continue;
                }

                if Self::are_colliding(&current, &self.particles[j], COLLISION_RADIUS) {
                    self.stats.collision_count += 1;
                    self.stats.merge_count += 1;
                    current = Self::merge(&current, &self.particles[j]);
                    merged[j] = true;
                }
            }

            new_particles.push(current);
        }
        self.particles = new_particles;
    }

    pub fn resolve_collisions_pairs(&mut self, pairs: Vec<CollisionPair>) {
        let mut merged = vec![false; self.particles.len()];
        let mut new_particles: Vec<Particle> = Vec::new();

        for i in 0..self.particles.len() {
            if merged[i] {
                continue;
            }

            let mut current = self.particles[i];

            for pair in &pairs {
                if pair.a == i && !merged[pair.b] {
                    current = Self::merge(&current, &self.particles[pair.b]);
                    merged[pair.b] = true;

                    self.stats.collision_count += 1;
                    self.stats.merge_count += 1;
                }
            }

            new_particles.push(current);
        }

        self.particles = new_particles;
    }

    pub fn update_temperature(&mut self) {
        // TODO: cooling formula
    }

    pub fn simulate_particle_movement(&mut self, steps: usize, delta_time : f32, gravity : f32, debug_logging : bool)
    {
        for step in 0..steps{
            self.update_physics(delta_time, gravity);
            self.resolve_collisions();
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
        p.x = random_range(0.45..=0.55);
        p.y = 2.0;

        p.vx = random_range(-2.0..=2.0);
        p.vy = random_range(-5.0..=-1.0);
    }
 }