use std::{collections::HashMap};
use rand::{random_range};

pub type Cell = (i32, i32, i32);
pub type Grid = HashMap<Cell, Vec<usize>>;

pub const WIDTH: f32 = 1.0;
pub const HEIGHT: f32 = 2.0;
pub const DEPTH: f32 = 1.0;
pub const EMITTER_RADIUS: f32 = 0.05;

#[derive(Debug, Copy, Clone)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub mass: f32,
    pub temperature: f32
}

impl Particle
{
    pub fn new(x_variable: f32, y_variable: f32, z_variable: f32, vx_variable: f32, vy_variable: f32, vz_variable: f32) -> Particle{
        Particle{
            x: x_variable,
            y: y_variable,
            z: z_variable,
            vx: vx_variable,
            vy: vy_variable,
            vz: vz_variable,
            mass: 1.0,
            temperature: 1.0
        }
    }

    pub fn update_position(&mut self, delta_time: f32, gravity: f32)
    {
        if self.y > 0.0 {
            self.vy += gravity * delta_time;
        }

        self.x += self.vx * delta_time;

        self.y += self.vy * delta_time;

        self.z += self.vz * delta_time;

        if self.x <= 0.0 {
            self.x = 0.0;
            self.vx = -self.vx;
        }

        if self.x >= WIDTH {
            self.x = WIDTH;
            self.vx = -self.vx;
        }

        if self.z <= 0.0 {
            self.z = 0.0;
            self.vz = -self.vz;
        }

        if self.z >= DEPTH {
            self.z = DEPTH;
            self.vz = -self.vz;
        }

        self.x = self.x.clamp(0.0, WIDTH);
        self.y = self.y.clamp(0.0, HEIGHT);
        self.z = self.z.clamp(0.0, DEPTH);
    }

    pub fn update_temperature(&mut self, delta_time : f32, cooling_factor: f32) {
        self.temperature -= (delta_time * cooling_factor) / self.mass;
        self.temperature = self.temperature.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Debug)]
pub struct SimulationStats {
    pub floor_hits: usize,
    pub collision_count: usize,
    pub merge_count: usize
}

#[derive(Clone)]
pub struct ParticleSystem {
    pub particles: Vec<Particle>,
    pub stats: SimulationStats
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

    pub fn update_temperature(&mut self, delta_time : f32, cooling_factor : f32) {
        for particle in &mut self.particles {
            particle.update_temperature(delta_time, cooling_factor);
        }
    }

    pub fn average_temperature(&self) -> f32 {
        if self.particles.is_empty() {
            return 0.0;
        }

        let total: f32 = self.particles.iter().map(|p| p.temperature).sum();

        total / self.particles.len() as f32
    }

    pub fn min_temperature(&self) -> f32 {
        self.particles.iter().map(|p| p.temperature).fold(1.0, f32::min)
    }

    pub fn build_grid(&self, cell_size: f32) -> Grid {
        let mut grid: Grid = HashMap::new();

        for (i, particle) in self.particles.iter().enumerate() {
            let cell = get_cell(particle, cell_size);
            grid.entry(cell).or_default().push(i);
        }

        grid
    }

    pub fn are_colliding(a: &Particle, b: &Particle, radius: f32) -> bool {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let dz = a.z - b.z;
        dx * dx + dy * dy + dz * dz <= radius * radius
    }

    pub fn merge(a: &Particle, b: &Particle) -> Particle {
        let mass = a.mass + b.mass;

        let vx = (a.mass * a.vx + b.vx * b.mass) / mass;
        let vy = (a.mass * a.vy + b.mass * b.vy) / mass;
        let vz = (a.mass * a.vz + b.mass * b.vz) / mass;

        let x = (a.x * a.mass + b.x * b.mass) / mass;
        let y = (a.y * a.mass + b.y * b.mass) / mass;
        let z = (a.z * a.mass + b.z * b.mass) / mass;

        let temperature = (a.temperature * a.mass + b.temperature * b.mass) / mass;
        Particle { x, y, z, vx, vy, vz, mass, temperature }
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
                        for dz in -1..=1 {
                            let neighbour_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);

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
            }

            new_particles.push(current);
        }
        self.particles = new_particles;

        let new_count = self.particles.len();
        let merges = old_count - new_count;

        self.stats.collision_count += merges;
        self.stats.merge_count += merges;
    }

    pub fn simulate_particle_movement(&mut self, steps: usize, delta_time : f32, gravity : f32, collision_radius: f32, cooling_factor : f32, debug_logging : bool)
    {
        for step in 0..steps{
            self.update_physics(delta_time, gravity);
            self.resolve_collisions(collision_radius);
            self.update_temperature(delta_time, cooling_factor);

            if debug_logging && step % 20_000 == 0 {
                println!("Step: {}, Particles: {}, Floor Hits: {}, Collisions: {}, Merges: {}, Avg Temp: {:.3}, Min Temp: {:.3}", 
                step, self.particles.len(), self.stats.floor_hits, self.stats.collision_count, 
                self.stats.merge_count, self.average_temperature(), self.min_temperature());
            }
        }
    }

    pub fn respawn_particle(p: &mut Particle)
    {
        let (x, y, z) = random_emitter_position();
        let (vx, vy, vz) = random_cone_velocity();

        p.x = x;
        p.y = y;
        p.z = z;
        p.vx = vx;
        p.vy = vy;
        p.vz = vz;
        p.mass = 1.0;
        p.temperature = 1.0;
    }
 }


pub fn random_cone_velocity() -> (f32, f32, f32) {
    let speed: f32 = 2.0;

    let angle = random_range(0.0f32..=30.0f32).to_radians();
    let azimuth = random_range(0.0f32..=std::f32::consts::TAU);

    let horizontal = speed * angle.sin();

    let vx = horizontal * azimuth.cos();
    let vy = -speed * angle.cos();
    let vz = horizontal * azimuth.sin();

    (vx, vy, vz)
}
 
pub fn get_cell(p: &Particle, cell_size: f32) -> Cell {
    let x = (p.x / cell_size).floor() as i32;
    let y = (p.y / cell_size).floor() as i32;
    let z = (p.z / cell_size).floor() as i32;
    (x, y, z)
}

pub fn random_emitter_position() -> (f32, f32, f32) {
    let radius = EMITTER_RADIUS;
    let centre_x = WIDTH / 2.0;
    let centre_y = HEIGHT;
    let centre_z = DEPTH / 2.0;

    let r = radius * random_range(0.0..=1.0f32).sqrt();
    let theta = random_range(0.0..=std::f32::consts::TAU);

    let x = centre_x + r * theta.cos();
    let y = centre_y;
    let z = centre_z + r * theta.sin();

    (x, y, z)
}

pub fn cell_owner(cell: Cell, num_threads: usize) -> usize {
    let hash = cell.0.wrapping_mul(73856093) ^ cell.1.wrapping_mul(19349663);
    (hash.abs() as usize) % num_threads
}