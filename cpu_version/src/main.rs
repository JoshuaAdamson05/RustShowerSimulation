use std::{time::Instant};
use rand::{random_range};
mod simulation;
use simulation::*;

fn main() {
    const DEBUG_LOGGING: bool = false;
    const NUM_PARTICLES: usize = 100_000;
    const NUM_STEPS: usize = 100_000;
    const NUM_RUNS: usize = 3;
    const PHYSICS_THREADS: usize = 2;
    const COLLISION_THREADS: usize = 2;
    const TEMPERATURE_THREADS: usize = 2;
    const GRAVITY: f32 = -9.81;
    const DELTA_TIME: f32 = 1.0 / 60.0;
    const COLLISION_RADIUS: f32 = 0.0001;
    const COOLING_FACTOR: f32 = 0.5;

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

    run_sequential_simulation(ps_sequential, NUM_STEPS, NUM_RUNS, DELTA_TIME, GRAVITY, COLLISION_RADIUS, COOLING_FACTOR, DEBUG_LOGGING);
    run_threaded_simulation(ps_threaded, NUM_STEPS, PHYSICS_THREADS, COLLISION_THREADS, TEMPERATURE_THREADS, NUM_RUNS, DELTA_TIME, GRAVITY, COLLISION_RADIUS, COOLING_FACTOR, DEBUG_LOGGING);
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

pub fn update_temperature_chunk(particles : &mut [Particle], delta_time : f32, cooling_factor : f32) {
    for particle in particles {
        particle.update_temperature(delta_time, cooling_factor);
    }
}

pub fn resolve_owned_grid_cells(particles : &[Particle], grid: &Grid, owned_cells: &[Cell], collision_radius: f32) -> Vec<Particle> {
    let mut merged = vec![false; particles.len()];
    let mut local_particles = Vec::new();

    for &cell in owned_cells {
        if let Some(indices) = grid.get(&cell) {
            for &i in indices {
                if merged[i] {
                    continue;
                }

                let mut current = particles[i];
                let mut changed = true;

                while changed {
                    changed = false;
                    let current_cell = get_cell(&current, collision_radius * 2.0);

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            let neighbor_cell = (current_cell.0 + dx, current_cell.1 + dy);

                            if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                                for &j in neighbor_indices {
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
        }
    }

    local_particles
}

fn run_threaded_simulation(ps: ParticleSystem, steps: usize, physics_threads: usize, collision_threads: usize, temperature_threads: usize, runs: usize, delta_time : f32, gravity : f32, collision_radius: f32, cooling_factor : f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();      
        let mut physics_pool = scoped_threadpool::Pool::new(physics_threads as u32);
        let mut collision_pool = scoped_threadpool::Pool::new(collision_threads as u32);
        let mut temeperature_pool = scoped_threadpool::Pool::new(temperature_threads as u32);

        let start = Instant::now();

        for step in 0..steps {
            let chunk_size: usize = (ps_clone.particles.len() / physics_threads).max(1);   
            let mut floor_outputs = vec![0usize; physics_threads];

            physics_pool.scoped(|scope| {
                for (slice, output) in ps_clone.particles.chunks_mut(chunk_size).zip(floor_outputs.iter_mut()) {
                    scope.execute(move || {
                        *output = update_particle_chunk(slice, delta_time, gravity);
                    });
                }
            });

            //Combine results
            let total_hits: usize = floor_outputs.iter().sum();
            ps_clone.stats.floor_hits += total_hits;

            let old_count = ps_clone.particles.len();

            let grid = ps_clone.build_grid(collision_radius * 2.0);
            let mut collision_outputs: Vec<Vec<Particle>> = vec![Vec::new(); physics_threads];
            
            let len = ps_clone.particles.len();
            let mut owned_cells: Vec<Vec<Cell>> = vec![Vec::new(); physics_threads];

            for &cell in grid.keys() {
                let owner = cell_owner(cell, physics_threads);
                owned_cells[owner].push(cell);
            }

            collision_pool.scoped(|scope| {
                for (t, output) in collision_outputs.iter_mut().enumerate() {
                    let cells = &owned_cells[t];

                    if cells.is_empty() {
                        continue;
                    }

                    let particles = &ps_clone.particles;
                    let grid = &grid;

                    scope.execute(move || {
                    let local = resolve_owned_grid_cells(particles, grid, cells, collision_radius);

                    *output = local;
                    });
                }
            });

            let mut new_particles = Vec::with_capacity(len);

            for output in collision_outputs {
                new_particles.extend(output);
            }

            ps_clone.particles = new_particles;

            let local_merges = old_count - ps_clone.particles.len();
            ps_clone.stats.collision_count += local_merges;
            ps_clone.stats.merge_count += local_merges;

            // Cross-thread merge
            if step % 100 == 0 {
                loop {
                    let before = ps_clone.particles.len();
                    ps_clone.resolve_collisions(collision_radius);
                    let after = ps_clone.particles.len();

                    if before == after {
                        break;
                    }
                }
            }

            let temp_chunk_size = (ps_clone.particles.len() / physics_threads).max(1);

            temeperature_pool.scoped(|scope| {
                for slice in ps_clone.particles.chunks_mut(temp_chunk_size) {
                    scope.execute(move || {
                        update_temperature_chunk(slice, delta_time, cooling_factor);
                    });
                }
            });

            if debug_logging && step % 20_000 == 0 {
                println!(
                    "Threaded Step: {}, Particles: {}, Floor Hits: {}, Collisions: {}, Merges: {}, Avg Temp: {:.3}, Min Temp: {:.3}",
                    step, ps_clone.particles.len(), ps_clone.stats.floor_hits, ps_clone.stats.collision_count,
                    ps_clone.stats.merge_count, ps_clone.average_temperature(), ps_clone.min_temperature()
                );
            }
        }

        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }

    println!("Threaded avg time: {:.6} s", total / runs as f64);
}

fn run_sequential_simulation(ps: ParticleSystem, steps: usize, runs: usize, delta_time : f32, gravity : f32, collision_radius: f32, cooling_factor : f32, debug_logging : bool)
{
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();

        let start = Instant::now();
        ps_clone.simulate_particle_movement(steps, delta_time, gravity, collision_radius, cooling_factor, debug_logging);
        let duration = start.elapsed();
        total += duration.as_secs_f64();
    }
    
    println!("Sequential avg time: {:.6} s", total / runs as f64);
}