use std::{time::Instant, sync::Mutex, sync::atomic::{AtomicU8, Ordering}};
use cgmath::{perspective, Matrix4, Point3, Rad, Vector3};
use glium::{Surface, backend::glutin::SimpleWindowBuilder};
mod simulation;
mod renderer;
use simulation::*;
use renderer::*;

const INACTIVE: u8 = 0;
const ACTIVE: u8 = 1;
const LOCKED: u8 = 2;

fn main() {
    const VISUAL_MODE: bool = true;

    if VISUAL_MODE {
        run_visualisation();
    } else {
        run_benchmarks();
    }
}

fn run_visualisation() {
    const NUM_PARTICLES: usize = 100_000;
    const PHYSICS_THREADS: usize = 2;
    const COLLISION_THREADS: usize = 4;
    const TEMPERATURE_THREADS: usize = 2;

    const DELTA_TIME: f32 = 1.0 / 60.0;
    const GRAVITY: f32 = -9.81;
    const COLLISION_RADIUS: f32 = 0.0001;
    const COOLING_FACTOR: f32 = 1.0;
    const RENDER_STRIDE: usize = 100;

    let event_loop = glium::winit::event_loop::EventLoop::builder()
        .build()
        .unwrap();

    let (window, display) = SimpleWindowBuilder::new()
        .with_title("Particle Shower Simulation")
        .with_inner_size(900, 700)
        .build(&event_loop);

    let vertex_shader_src = r#"
        #version 140

        in vec3 position;
        in vec3 colour;

        out vec3 v_colour;

        uniform mat4 matrix;

        void main() {
            v_colour = colour;
            gl_Position = matrix * vec4(position, 1.0);
            gl_PointSize = 10.0;
        }
    "#;

    let particle_fragment_shader_src = r#"
        #version 140

        in vec3 v_colour;
        out vec4 color;

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);

            if(length(coord) > 0.5)
                discard;

            color = vec4(v_colour, 1.0);
        }
    "#;

    let line_fragment_shader_src = r#"
        #version 140

        in vec3 v_colour;
        out vec4 color;

        void main() {
            color = vec4(v_colour, 1.0);
        }
    "#;

    let particle_program = glium::Program::from_source(&display, vertex_shader_src, particle_fragment_shader_src, None)
        .unwrap();

    let line_program = glium::Program::from_source(&display, vertex_shader_src, line_fragment_shader_src, None)
        .unwrap();

    let mut ps = ParticleSystem::new();

    for _ in 0..NUM_PARTICLES {
        let (x, y, z) = random_emitter_position();
        let (vx, vy, vz) = random_cone_velocity();

        ps.particles.push(Particle::new(x, y, z, vx, vy, vz));
    }

    let mut physics_pool =
        scoped_threadpool::Pool::new(PHYSICS_THREADS as u32);

    let mut collision_pool =
        scoped_threadpool::Pool::new(COLLISION_THREADS as u32);

    let mut temperature_pool =
        scoped_threadpool::Pool::new(TEMPERATURE_THREADS as u32);

    let mut render_mode = RenderMode::Thermal;
    let mut frame_step: usize = 0;

    #[allow(deprecated)]
    event_loop
        .run(move |event, window_target| {
            match event {
                glium::winit::event::Event::WindowEvent { event, .. } => {
                    match event {
                        glium::winit::event::WindowEvent::CloseRequested => {
                            window_target.exit();
                        }

                        glium::winit::event::WindowEvent::KeyboardInput { event, .. } => {
                            use glium::winit::keyboard::{KeyCode, PhysicalKey};

                            if event.state == glium::winit::event::ElementState::Pressed {
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::KeyT) => {
                                        render_mode = RenderMode::Thermal;
                                    }
                                    PhysicalKey::Code(KeyCode::KeyM) => {
                                        render_mode = RenderMode::Mass;
                                    }
                                    _ => {}
                                }
                            }
                        }

                        glium::winit::event::WindowEvent::RedrawRequested => {
                            threaded_simulation_step(&mut ps, &mut physics_pool, &mut collision_pool, &mut temperature_pool, PHYSICS_THREADS, COLLISION_THREADS, TEMPERATURE_THREADS, DELTA_TIME, GRAVITY, COLLISION_RADIUS, COOLING_FACTOR);

                            if frame_step % 300 == 0 {
                                println!("Floor Hits: {} | Particles {} | Avg Temp: {:.3}",
                                ps.stats.floor_hits, ps.particles.len(), ps.average_temperature())
                            }

                            frame_step += 1;

                            // Particles
                            let particle_vertices = particles_to_vertices(&ps.particles, &render_mode, RENDER_STRIDE);

                            let particles_vertex_buffer = glium::VertexBuffer::new(&display, &particle_vertices).unwrap();

                            let particle_indices = glium::index::NoIndices(glium::index::PrimitiveType::Points);

                            // Cubicle
                            let cubicle_vertices = cubicle_vertices();

                            let cubicle_buffer = glium::VertexBuffer::new(&display, &cubicle_vertices).unwrap();
                            
                            let cubicle_indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);

                            // Shower Head
                            let shower_vertices = shower_head_vertices();

                            let shower_buffer = glium::VertexBuffer::new(&display, &shower_vertices).unwrap();

                            let shower_indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleFan);

                            let mut frame = display.draw();

                        frame.clear_color_and_depth((0.02, 0.02, 0.02, 1.0), 1.0);

                        let particle_draw_params = glium::DrawParameters {
                            point_size: Some(14.0),
                            ..Default::default()
                        };

                        let line_draw_params = glium::DrawParameters {
                            line_width: Some(4.0),
                            ..Default::default()
                        };

                        let perspective_matrix: Matrix4<f32> = perspective(Rad(std::f32::consts::FRAC_PI_4), 900.0 / 700.0, 0.1, 100.0);

                        let view_matrix = Matrix4::look_to_rh(Point3::new(1.5, 1.2, 3.0), Vector3::new(-1.5, -1.2, -3.0), Vector3::unit_y());

                        let matrix: [[f32; 4]; 4] = (perspective_matrix * view_matrix).into();

                        let uniforms = glium::uniform! {
                            matrix: matrix
                        };

                        frame.draw(&particles_vertex_buffer, &particle_indices, &particle_program, &uniforms, &particle_draw_params).unwrap();

                        frame.draw(&cubicle_buffer, &cubicle_indices, &line_program, &uniforms, &line_draw_params).unwrap();

                        frame.draw(&shower_buffer, &shower_indices, &line_program, &uniforms, &line_draw_params).unwrap();

                        frame.finish().unwrap();
                        }

                        _ => {}
                    }
                }

                glium::winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }

                _ => {}
            }
        })
        .unwrap();
}

pub fn threaded_simulation_step(
    ps: &mut ParticleSystem,
    physics_pool: &mut scoped_threadpool::Pool,
    collision_pool: &mut scoped_threadpool::Pool,
    temperature_pool: &mut scoped_threadpool::Pool,
    physics_threads: usize,
    collision_threads: usize,
    temperature_threads: usize,
    delta_time: f32,
    gravity: f32,
    collision_radius: f32,
    cooling_factor: f32,
) {
    let chunk_size = (ps.particles.len() / physics_threads).max(1);
    let mut floor_outputs = vec![0usize; physics_threads];

    physics_pool.scoped(|scope| {
        for (slice, output) in ps
            .particles
            .chunks_mut(chunk_size)
            .zip(floor_outputs.iter_mut())
        {
            scope.execute(move || {
                *output = update_particle_chunk(slice, delta_time, gravity);
            });
        }
    });

    let total_hits: usize = floor_outputs.iter().sum();
    ps.stats.floor_hits += total_hits;

    resolve_collisions_threaded_atomic(
        ps,
        collision_pool,
        collision_threads,
        collision_radius,
    );

    let temp_chunk_size = (ps.particles.len() / temperature_threads).max(1);

    temperature_pool.scoped(|scope| {
        for slice in ps.particles.chunks_mut(temp_chunk_size) {
            scope.execute(move || {
                update_temperature_chunk(slice, delta_time, cooling_factor);
            });
        }
    });
}

fn run_benchmarks(){
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
    const COOLING_FACTOR: f32 = 1.0;

    let mut ps = ParticleSystem::new();

    for _ in 0..NUM_PARTICLES {
        let (x, y, z) = random_emitter_position();
        let (vx, vy, vz) = random_cone_velocity();

        ps.particles.push(Particle::new(x, y, z, vx, vy, vz));
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

pub fn resolve_collisions_threaded_atomic(
    ps: &mut ParticleSystem,
    collision_pool: &mut scoped_threadpool::Pool,
    collision_threads: usize,
    collision_radius: f32,
) {
    let old_count = ps.particles.len();
    let len = ps.particles.len();

    let grid = ps.build_grid(CELL_SIZE);

    let mut owned_cells: Vec<Vec<Cell>> = vec![Vec::new(); collision_threads];

    for &cell in grid.keys() {
        let owner = cell_owner(cell, collision_threads);
        owned_cells[owner].push(cell);
    }

    let states: Vec<AtomicU8> = (0..len)
        .map(|_| AtomicU8::new(ACTIVE))
        .collect();

    let output_slots: Vec<Mutex<Option<Particle>>> = (0..len)
        .map(|_| Mutex::new(None))
        .collect();

    collision_pool.scoped(|scope| {
        for t in 0..collision_threads {
            let cells = &owned_cells[t];

            if cells.is_empty() {
                continue;
            }

            let particles = &ps.particles;
            let grid = &grid;
            let states = &states;
            let output_slots = &output_slots;

            scope.execute(move || {
                resolve_owned_grid_cells_atomic(
                    particles,
                    grid,
                    cells,
                    collision_radius,
                    states,
                    output_slots,
                );
            });
        }
    });

    let mut new_particles = Vec::with_capacity(len);

    for i in 0..len {
        let state = states[i].load(Ordering::Acquire);

        if state != INACTIVE {
            let particle = output_slots[i]
                .lock()
                .unwrap()
                .take()
                .unwrap_or(ps.particles[i]);

            new_particles.push(particle);
        }
    }

    ps.particles = new_particles;

    let merges = old_count - ps.particles.len();

    ps.stats.collision_count += merges;
    ps.stats.merge_count += merges;
}

pub fn resolve_owned_grid_cells_atomic(
    particles: &[Particle],
    grid: &Grid,
    owned_cells: &[Cell],
    collision_radius: f32,
    states: &[AtomicU8],
    output_slots: &[Mutex<Option<Particle>>],
) {
    for &cell in owned_cells {
        if let Some(indices) = grid.get(&cell) {
            for &i in indices {
                // Try to claim this particle as the active survivor.
                // If another thread has already merged it, skip it.
                if states[i]
                    .compare_exchange(ACTIVE, LOCKED, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    continue;
                }

                let mut current = particles[i];
                let current_cell = get_cell(&current, CELL_SIZE);
                let mut has_merged = false;

                for dx in -1..=1 {
                    if has_merged {
                        break;
                    }

                    for dy in -1..=1 {
                        if has_merged {
                            break;
                        }

                        for dz in -1..=1 {
                            if has_merged {
                                break;
                            }

                            let neighbour_cell = (
                                current_cell.0 + dx,
                                current_cell.1 + dy,
                                current_cell.2 + dz,
                            );

                            if let Some(neighbour_indices) = grid.get(&neighbour_cell) {
                                for &j in neighbour_indices {
                                    if j == i {
                                        continue;
                                    }

                                    let dx = current.x - particles[j].x;
                                    let dy = current.y - particles[j].y;
                                    let dz = current.z - particles[j].z;

                                    let distance_sq = dx * dx + dy * dy + dz * dz;
                                    let collision_sq = collision_radius * collision_radius;

                                    if distance_sq <= collision_sq {
                                        // Try to claim particle j.
                                        if states[j]
                                            .compare_exchange(
                                                ACTIVE,
                                                INACTIVE,
                                                Ordering::AcqRel,
                                                Ordering::Acquire,
                                            )
                                            .is_ok()
                                        {
                                            current = ParticleSystem::merge(&current, &particles[j]);
                                            has_merged = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Store this particle's resulting state.
                *output_slots[i].lock().unwrap() = Some(current);

                // Unlock the survivor.
                states[i].store(ACTIVE, Ordering::Release);
            }
        }
    }
}

fn run_threaded_simulation(
    ps: ParticleSystem,
    steps: usize,
    physics_threads: usize,
    collision_threads: usize,
    temperature_threads: usize,
    runs: usize,
    delta_time: f32,
    gravity: f32,
    collision_radius: f32,
    cooling_factor: f32,
    debug_logging: bool,
) {
    let mut total = 0.0;

    for _ in 0..runs {
        let mut ps_clone = ps.clone();

        let mut physics_pool = scoped_threadpool::Pool::new(physics_threads as u32);
        let mut collision_pool = scoped_threadpool::Pool::new(collision_threads as u32);
        let mut temperature_pool = scoped_threadpool::Pool::new(temperature_threads as u32);

        let start = Instant::now();

        for step in 0..steps {
            // Physics stage
            let chunk_size = (ps_clone.particles.len() / physics_threads).max(1);
            let chunk_count = (ps_clone.particles.len() + chunk_size - 1) / chunk_size;

            let mut floor_outputs = vec![0usize; chunk_count];

            physics_pool.scoped(|scope| {
                for (slice, output) in ps_clone
                    .particles
                    .chunks_mut(chunk_size)
                    .zip(floor_outputs.iter_mut())
                {
                    scope.execute(move || {
                        *output = update_particle_chunk(slice, delta_time, gravity);
                    });
                }
            });

            let total_hits: usize = floor_outputs.iter().sum();
            ps_clone.stats.floor_hits += total_hits;

            // Collision / merging stage
            resolve_collisions_threaded_atomic(
                &mut ps_clone,
                &mut collision_pool,
                collision_threads,
                collision_radius,
            );

            // Temperature stage
            let temp_chunk_size = (ps_clone.particles.len() / temperature_threads).max(1);

            temperature_pool.scoped(|scope| {
                for slice in ps_clone.particles.chunks_mut(temp_chunk_size) {
                    scope.execute(move || {
                        update_temperature_chunk(slice, delta_time, cooling_factor);
                    });
                }
            });

            if debug_logging && step % 20_000 == 0 {
                println!(
                    "Threaded Step: {}, Particles: {}, Floor Hits: {}, Collisions: {}, Merges: {}, Avg Temp: {:.3}, Min Temp: {:.3}",
                    step,
                    ps_clone.particles.len(),
                    ps_clone.stats.floor_hits,
                    ps_clone.stats.collision_count,
                    ps_clone.stats.merge_count,
                    ps_clone.average_temperature(),
                    ps_clone.min_temperature()
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