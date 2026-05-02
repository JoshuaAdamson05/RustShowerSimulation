use std::{time::Instant};
use rand::{random_range};

fn main() {
    const NUM_PARTICLES: usize = 100;
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

    run_sequential_simulation(ps_sequential, NUM_STEPS, NUM_RUNS, DELTA_TIME, GRAVITY);
    run_threaded_simulation(ps_threaded, NUM_STEPS, PHYSICS_THREADS, NUM_RUNS, DELTA_TIME, GRAVITY);
}

pub fn update_particle_chunk(particles : &mut [Particle], delta_time : f32, gravity : f32)
{
    for particle in particles
    {
        particle.update_position(delta_time, gravity);
    }
}

fn run_threaded_simulation(ps: ParticleSystem, steps: usize, num_threads: usize, runs: usize, delta_time : f32, gravity : f32)
{
    let mut total = 0.0;
    for _ in 0..runs {
        let mut ps_clone = ps.clone();
        let chunk_size: usize = ps_clone.particles.len() / num_threads;   
        let mut pool = scoped_threadpool::Pool::new(num_threads as u32);

        let start = Instant::now();
        pool.scoped(|scope| {
            for slice in ps_clone.particles.chunks_mut(chunk_size) 
            {
                scope.execute(move || {
                    for _ in 0..steps{
                        update_particle_chunk(slice, delta_time, gravity);
                    }
                });
            }
        });
        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }
    println!("Threaded avg time: {:.6} s", total / runs as f64);
}

fn run_sequential_simulation(ps: ParticleSystem, steps: usize, runs: usize, delta_time : f32, gravity : f32)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();

        let start = Instant::now();
        ps_clone.simulate_particle_movement(steps, delta_time, gravity);
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
    vy: f32
}

impl Particle
{
    pub fn new(x_variable: f32, y_variable: f32, vx_variable: f32, vy_variable: f32) -> Particle{
        Particle{
            x: x_variable,
            y: y_variable,
            vx: vx_variable,
            vy: vy_variable
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

#[derive(Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>,
    floor_hits: usize
}

impl ParticleSystem {
    pub fn new() -> ParticleSystem {
        ParticleSystem {
            particles: Vec::new(),
            floor_hits: 0
        }
    }

    pub fn update_physics(&mut self, delta_time : f32, gravity : f32)
    {
        for particle in &mut self.particles{
            particle.update_position(delta_time, gravity);

            if particle.y <= 0.0 {
                self.floor_hits += 1;
                Self::respawn_particle(particle);
            }
        }
    }

    pub fn resolve_collisions(&mut self) {
        // TODO: collision detection + merging
    }

    pub fn update_temperature(&mut self) {
        // TODO: cooling formula
    }

    pub fn simulate_particle_movement(&mut self, steps: usize, delta_time : f32, gravity : f32)
    {
        for _ in 0..steps{
            self.update_physics(delta_time, gravity);

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