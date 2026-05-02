use std::time::{Instant};

use rand::{random, random_range};

fn main() {
    const PARTICLES_PER_THREAD: usize = 20;
    let mut ps = ParticleSystem::new();

    for _ in 0..100{
        let x: f32 = random_range(0.0..=10.0);
        let y: f32 = random_range(0.0..=10.0);
        let p = Particle::new(x, y);
        ps.particles.push(p);
    }

    const NUM_OF_THREADS: usize = 5;
    let mut pool = scoped_threadpool::Pool::new(NUM_OF_THREADS as u32);

    let start = Instant::now();
    // Limit the scope of the reads to this section of code
    pool.scoped(|scope| {
        for slice in ps.particles.chunks_mut(PARTICLES_PER_THREAD) 
        {
            scope.execute(move || thread_main(slice));
        }
    });
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);
    // Implicit join here, where all threads go out of scope.
}

pub fn thread_main(particles : &mut [Particle])
{
    for particle in particles
    {
        particle.move_random_distance();
    }
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

pub struct ParticleSystem<> {
    particles: Vec<Particle>
}

impl ParticleSystem<> {
    pub fn new() -> ParticleSystem<> {
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

    pub fn simulate_particle_movement(&mut self)
    {
        for _ in 0..400_000{
            self.move_particles();
        }
    }
 }