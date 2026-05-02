use std::{time::Instant};
use rand::{random, random_range};

fn main() {
    const NUM_PARTICLES: usize = 100;
    const NUM_STEPS: usize = 400_000;
    const NUM_OF_THREADS: usize = 5;
    let mut ps = ParticleSystem::new();

    for _ in 0..NUM_PARTICLES{
        let x: f32 = random_range(0.0..=10.0);
        let y: f32 = random_range(0.0..=10.0);
        let p = Particle::new(x, y);
        ps.particles.push(p);
    }

    let ps_sequential = ps.clone();
    let ps_threaded = ps.clone();

    run_sequential_simulation(ps_sequential, NUM_STEPS);
    run_threaded_simulation(ps_threaded, NUM_STEPS, NUM_OF_THREADS);
}

pub fn update_particle_chunk(particles : &mut [Particle])
{
    for particle in particles
    {
        particle.move_random_distance();
    }
}

fn run_threaded_simulation(mut ps: ParticleSystem, steps: usize, num_threads: usize)
{
    let chunk_size: usize = ps.particles.len() / num_threads;   
    let mut pool = scoped_threadpool::Pool::new(num_threads as u32);

    let start = Instant::now();
    pool.scoped(|scope| {
        for slice in ps.particles.chunks_mut(chunk_size) 
        {
            scope.execute(move || {
                for _ in 0..steps{
                    update_particle_chunk(slice);
                }
            });
        }
    });
    let duration = start.elapsed();
    println!("Threaded time taken: {:?}", duration);
}

fn run_sequential_simulation(mut ps: ParticleSystem, steps: usize)
{
    let start = Instant::now();
    ps.simulate_particle_movement(steps);
    let duration = start.elapsed();
    println!("Sequential time taken: {:?}", duration);
}

#[derive(Debug, Copy, Clone)]
pub struct Particle {
    x: f32,
    y: f32
}

impl Particle
{
    pub fn new(x_variable: f32, y_variable: f32) -> Particle{
        Particle{
            x: x_variable,
            y: y_variable
        }
    }

    pub fn move_random_distance(&mut self)
    {
        let x = random::<f32>();
        let y = random::<f32>();
        self.x += x;
        self.y +=y;

        self.x = self.x.clamp(0.0, 10.0);
        self.y = self.y.clamp(0.0, 10.0);
    }
}

#[derive(Clone)]
pub struct ParticleSystem {
    particles: Vec<Particle>
}

impl ParticleSystem {
    pub fn new() -> ParticleSystem {
        ParticleSystem {
            particles: Vec::new()
        }
    }

    pub fn move_particles(&mut self)
    {
        for particle in &mut self.particles{
            particle.move_random_distance();
        }
    }

    pub fn simulate_particle_movement(&mut self, steps: usize)
    {
        for _ in 0..steps{
            self.move_particles();
        }
    }
 }