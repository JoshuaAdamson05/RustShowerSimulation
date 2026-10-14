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
const SURVIVOR: u8 = 3;

pub struct CollisionScratch {
    states: Vec<AtomicU8>,
    output_slots: Vec<Mutex<Option<Particle>>>,
}

impl CollisionScratch {
    fn new(capacity: usize) -> Self {
        let states = (0..capacity)
            .map(|_| AtomicU8::new(ACTIVE))
            .collect();

        let output_slots = (0..capacity)
            .map(|_| Mutex::new(None))
            .collect();

        Self {
            states,
            output_slots,
        }
    }

    fn prepare(&mut self, len: usize) {
        while self.states.len() < len {
            self.states.push(AtomicU8::new(ACTIVE));
            self.output_slots.push(Mutex::new(None));
        }

        for state in self.states.iter().take(len) {
            state.store(ACTIVE, Ordering::Relaxed);
        }
    }
}

#[derive(Copy, Clone, Default)]
pub struct CollisionStageTimings {
    pub grid_build_ms: f64,
    pub collision_ms: f64,
}

#[derive(Copy, Clone, Default)]
pub struct SimulationStageTimings {
    pub physics_ms: f64,
    pub grid_build_ms: f64,
    pub collision_ms: f64,
    pub temperature_ms: f64,
}

struct VisualProfileWindow {
    sums: [f64; 9],
    peaks: [f64; 9],
    samples: usize,
}

impl VisualProfileWindow {
    fn new() -> Self {
        Self {
            sums: [0.0; 9],
            peaks: [0.0; 9],
            samples: 0,
        }
    }

    fn add(&mut self, values: [f64; 9]) {
        for (i, value) in values.iter().enumerate() {
            self.sums[i] += *value;
            if *value > self.peaks[i] {
                self.peaks[i] = *value;
            }
        }
        self.samples += 1;
    }

    fn print(&self, last_frame: usize) {
        if self.samples == 0 {
            return;
        }

        const NAMES: [&str; 9] = [
            "Physics",
            "Grid build",
            "Collision/merge",
            "Temperature",
            "Simulation update total",
            "Render vertex preparation",
            "Particle buffer upload",
            "Render+swap",
            "Whole frame",
        ];

        let first_frame = last_frame.saturating_sub(self.samples - 1);
        println!(
            "PROFILE frames {}-{} | samples={} | mean / max ms",
            first_frame,
            last_frame,
            self.samples
        );

        for i in 0..NAMES.len() {
            println!(
                "  {}: {:.4} / {:.4}",
                NAMES[i],
                self.sums[i] / self.samples as f64,
                self.peaks[i]
            );
        }
    }

    fn reset(&mut self) {
        self.sums = [0.0; 9];
        self.peaks = [0.0; 9];
        self.samples = 0;
    }
}

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
    const COLLISION_THREADS: usize = 2;
    const TEMPERATURE_THREADS: usize = 2;

    const DELTA_TIME: f32 = 1.0 / 60.0;
    const GRAVITY: f32 = -9.81;
    const COLLISION_RADIUS: f32 = 0.0001;
    const COOLING_FACTOR: f32 = 1.0;
    const RENDER_STRIDE: usize = 100;
    const MAX_RENDER_VERTICES: usize = (NUM_PARTICLES + RENDER_STRIDE - 1) / RENDER_STRIDE;

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

    let mut physics_pool = scoped_threadpool::Pool::new(PHYSICS_THREADS as u32);
    let mut collision_pool = scoped_threadpool::Pool::new(COLLISION_THREADS as u32);
    let mut temperature_pool = scoped_threadpool::Pool::new(TEMPERATURE_THREADS as u32);
    let mut collision_scratch = CollisionScratch::new(NUM_PARTICLES);

    // Static geometry never changes, so upload it once rather than rebuilding and
    // reallocating its GPU buffers every frame.
    let cubicle_buffer = glium::VertexBuffer::new(&display, &cubicle_vertices()).unwrap();
    let shower_buffer = glium::VertexBuffer::new(&display, &shower_head_vertices()).unwrap();

    // Particle data changes each frame, but the maximum displayed vertex count is
    // fixed by the render stride. Reuse one dynamic GPU buffer and update a slice.
    let particles_vertex_buffer =
        glium::VertexBuffer::<Vertex>::empty_dynamic(&display, MAX_RENDER_VERTICES).unwrap();

    let particle_indices = glium::index::NoIndices(glium::index::PrimitiveType::Points);
    let cubicle_indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);
    let shower_indices = glium::index::NoIndices(glium::index::PrimitiveType::TriangleFan);

    let particle_draw_params = glium::DrawParameters {
        point_size: Some(8.0),
        ..Default::default()
    };

    let line_draw_params = glium::DrawParameters {
        line_width: Some(2.0),
        ..Default::default()
    };

    let perspective_matrix: Matrix4<f32> = perspective(
        Rad(std::f32::consts::FRAC_PI_4),
        900.0 / 700.0,
        0.1,
        100.0,
    );
    let view_matrix = Matrix4::look_to_rh(
        Point3::new(1.5, 1.2, 3.0),
        Vector3::new(-1.5, -1.2, -3.0),
        Vector3::unit_y(),
    );
    let matrix: [[f32; 4]; 4] = (perspective_matrix * view_matrix).into();

    let mut render_mode = RenderMode::Thermal;
    let mut frame_step: usize = 0;
    let mut total_update_ms = 0.0f64;
    let mut profile_window = VisualProfileWindow::new();

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
                                    PhysicalKey::Code(KeyCode::KeyT) => render_mode = RenderMode::Thermal,
                                    PhysicalKey::Code(KeyCode::KeyM) => render_mode = RenderMode::Mass,
                                    _ => {}
                                }
                            }
                        }

                        glium::winit::event::WindowEvent::RedrawRequested => {
                            let frame_start = Instant::now();
                            let update_start = Instant::now();

                            let stage_timings = threaded_simulation_step(
                                &mut ps,
                                &mut physics_pool,
                                &mut collision_pool,
                                &mut temperature_pool,
                                &mut collision_scratch,
                                PHYSICS_THREADS,
                                COLLISION_THREADS,
                                TEMPERATURE_THREADS,
                                DELTA_TIME,
                                GRAVITY,
                                COLLISION_RADIUS,
                                COOLING_FACTOR,
                            );

                            let update_ms = update_start.elapsed().as_secs_f64() * 1000.0;
                            total_update_ms += update_ms;
                            frame_step += 1;

                            let vertex_start = Instant::now();
                            let particle_vertices =
                                particles_to_vertices(&ps.particles, &render_mode, RENDER_STRIDE);
                            let vertex_prep_ms = vertex_start.elapsed().as_secs_f64() * 1000.0;

                            let upload_start = Instant::now();
                            let particle_vertex_count = particle_vertices.len();

                            if particle_vertex_count > 0 {
                                let particle_slice = particles_vertex_buffer
                                    .slice(0..particle_vertex_count)
                                    .expect("particle render slice exceeded preallocated buffer");
                                particle_slice.write(&particle_vertices);
                            }

                            let particle_upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;

                            let render_start = Instant::now();
                            let mut frame = display.draw();
                            frame.clear_color_and_depth((0.02, 0.02, 0.02, 1.0), 1.0);

                            let uniforms = glium::uniform! { matrix: matrix };

                            if particle_vertex_count > 0 {
                                let particle_slice = particles_vertex_buffer
                                    .slice(0..particle_vertex_count)
                                    .expect("particle render slice exceeded preallocated buffer");

                                frame.draw(
                                    particle_slice,
                                    &particle_indices,
                                    &particle_program,
                                    &uniforms,
                                    &particle_draw_params,
                                ).unwrap();
                            }

                            frame.draw(
                                &cubicle_buffer,
                                &cubicle_indices,
                                &line_program,
                                &uniforms,
                                &line_draw_params,
                            ).unwrap();

                            frame.draw(
                                &shower_buffer,
                                &shower_indices,
                                &line_program,
                                &uniforms,
                                &line_draw_params,
                            ).unwrap();

                            frame.finish().unwrap();
                            let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
                            let whole_frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;

                            profile_window.add([
                                stage_timings.physics_ms,
                                stage_timings.grid_build_ms,
                                stage_timings.collision_ms,
                                stage_timings.temperature_ms,
                                update_ms,
                                vertex_prep_ms,
                                particle_upload_ms,
                                render_ms,
                                whole_frame_ms,
                            ]);

                            if frame_step % 300 == 0 {
                                let avg_update_ms = total_update_ms / frame_step as f64;

                                println!(
                                    "Frame: {} | Floor Hits: {} | Particles {} | Rendered Vertices: {} | Avg Temp: {:.3} | Rust Update: {:.3} ms | Avg Rust Update: {:.3} ms",
                                    frame_step,
                                    ps.stats.floor_hits,
                                    ps.particles.len(),
                                    particle_vertex_count,
                                    ps.average_temperature(),
                                    update_ms,
                                    avg_update_ms
                                );

                                profile_window.print(frame_step);
                                profile_window.reset();
                            }
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
    collision_scratch: &mut CollisionScratch,
    physics_threads: usize,
    collision_threads: usize,
    temperature_threads: usize,
    delta_time: f32,
    gravity: f32,
    collision_radius: f32,
    cooling_factor: f32,
) -> SimulationStageTimings {
    let physics_start = Instant::now();
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
    let physics_ms = physics_start.elapsed().as_secs_f64() * 1000.0;

    let collision_timings = resolve_collisions_threaded_atomic(
        ps,
        collision_pool,
        collision_threads,
        collision_radius,
        collision_scratch,
    );

    let temperature_start = Instant::now();
    let temp_chunk_size = (ps.particles.len() / temperature_threads).max(1);

    temperature_pool.scoped(|scope| {
        for slice in ps.particles.chunks_mut(temp_chunk_size) {
            scope.execute(move || {
                update_temperature_chunk(slice, delta_time, cooling_factor);
            });
        }
    });
    let temperature_ms = temperature_start.elapsed().as_secs_f64() * 1000.0;

    SimulationStageTimings {
        physics_ms,
        grid_build_ms: collision_timings.grid_build_ms,
        collision_ms: collision_timings.collision_ms,
        temperature_ms,
    }
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
    for particle in particles {
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
    collision_scratch: &mut CollisionScratch,
) -> CollisionStageTimings {
    let old_count = ps.particles.len();
    let len = ps.particles.len();

    let grid_start = Instant::now();
    let grid = ps.build_grid(CELL_SIZE);
    let grid_build_ms = grid_start.elapsed().as_secs_f64() * 1000.0;

    let collision_start = Instant::now();
    let mut owned_cells: Vec<Vec<Cell>> = vec![Vec::new(); collision_threads];

    for &cell in grid.keys() {
        let owner = cell_owner(cell, collision_threads);
        owned_cells[owner].push(cell);
    }

    collision_scratch.prepare(len);
    let states = &collision_scratch.states[..len];
    let output_slots = &collision_scratch.output_slots[..len];

    collision_pool.scoped(|scope| {
        for t in 0..collision_threads {
            let cells = &owned_cells[t];

            if cells.is_empty() {
                continue;
            }

            let particles = &ps.particles;
            let grid = &grid;

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

    CollisionStageTimings {
        grid_build_ms,
        collision_ms: collision_start.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn resolve_owned_grid_cells_atomic(
    particles: &[Particle],
    grid: &Grid,
    owned_cells: &[Cell],
    collision_radius: f32,
    states: &[AtomicU8],
    output_slots: &[Mutex<Option<Particle>>],
) {
    let collision_sq = collision_radius * collision_radius;

    for &cell in owned_cells {
        if let Some(indices) = grid.get(&cell) {
            for &i in indices {
                if states[i]
                    .compare_exchange(ACTIVE, LOCKED, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    continue;
                }

                let mut current = particles[i];
                let ((min_x, min_y, min_z), (max_x, max_y, max_z)) =
                    collision_cell_bounds(&current, collision_radius, CELL_SIZE);
                let mut has_merged = false;

                for cell_x in min_x..=max_x {
                    if has_merged {
                        break;
                    }

                    for cell_y in min_y..=max_y {
                        if has_merged {
                            break;
                        }

                        for cell_z in min_z..=max_z {
                            if has_merged {
                                break;
                            }

                            let neighbour_cell = (cell_x, cell_y, cell_z);

                            if let Some(neighbour_indices) = grid.get(&neighbour_cell) {
                                for &j in neighbour_indices {
                                    if j == i {
                                        continue;
                                    }

                                    if states[j].load(Ordering::Acquire) != ACTIVE {
                                        continue;
                                    }

                                    let dx = current.x - particles[j].x;
                                    let dy = current.y - particles[j].y;
                                    let dz = current.z - particles[j].z;
                                    let distance_sq = dx * dx + dy * dy + dz * dz;

                                    if distance_sq <= collision_sq {
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

                *output_slots[i].lock().unwrap() = Some(current);
                states[i].store(SURVIVOR, Ordering::Release);
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
        let mut collision_scratch = CollisionScratch::new(ps_clone.particles.len());

        let start = Instant::now();

        for step in 0..steps {
            let _ = threaded_simulation_step(
                &mut ps_clone,
                &mut physics_pool,
                &mut collision_pool,
                &mut temperature_pool,
                &mut collision_scratch,
                physics_threads,
                collision_threads,
                temperature_threads,
                delta_time,
                gravity,
                collision_radius,
                cooling_factor,
            );

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