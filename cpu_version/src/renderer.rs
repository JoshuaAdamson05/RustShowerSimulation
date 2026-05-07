use glium::{implement_vertex, Surface};

use crate::simulation::{EMITTER_RADIUS, HEIGHT, Particle, WIDTH};

const VIEW_WIDTH: f32 = 1.4;
const VIEW_HEIGHT: f32 = 2.4;

#[derive(Copy, Clone)]
pub struct Vertex {
    position: [f32; 2],
    colour: [f32; 3]
}

implement_vertex!(Vertex, position, colour);

pub enum RenderMode {
    Thermal,
    Mass
}

pub fn temperature_colour(temp: f32) -> [f32; 3] {
    let t = temp.clamp(0.0, 1.0);

    [t, 0.0, 1.0 - t]
}

pub fn mass_colour(mass: f32) -> [f32; 3] {
    if mass < 2.0 {
        [0.0, 0.5, 1.0]
    } else if mass < 5.0 {
        [0.0, 1.0, 0.0]
    } else if mass < 10.0 {
        [1.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    }
}

fn world_to_screen(x: f32, y: f32) -> [f32; 2] {
    let offset_x = (VIEW_WIDTH - WIDTH) / 2.0;
    let offset_y = (VIEW_HEIGHT - HEIGHT) / 2.0;

    let centred_x = x + offset_x;
    let centred_y = y + offset_y;
    [
        (centred_x / VIEW_WIDTH) * 2.0 - 1.0,
        (centred_y / VIEW_HEIGHT) * 2.0 - 1.0
    ]
}

pub fn cubicle_vertices() -> Vec<Vertex> {
    let left = 0.0;
    let right =  WIDTH;
    let bottom = 0.0;
    let top = HEIGHT;

    vec![
        // left wall
        Vertex { position: world_to_screen(left, bottom), colour: [1.0, 1.0, 1.0] },
        Vertex { position: world_to_screen(left, top), colour: [1.0, 1.0, 1.0] },

        // ceiling
        Vertex { position: world_to_screen(left, top), colour: [1.0, 1.0, 1.0] },
        Vertex { position: world_to_screen(right, top), colour: [1.0, 1.0, 1.0] },

        // right wall
        Vertex { position: world_to_screen(right, top), colour: [1.0, 1.0, 1.0] },
        Vertex { position: world_to_screen(right, bottom), colour: [1.0, 1.0, 1.0] },

        // floor
        Vertex { position: world_to_screen(right, bottom), colour: [1.0, 1.0, 1.0] },
        Vertex { position: world_to_screen(left, bottom), colour: [1.0, 1.0, 1.0] },
    ]
}

pub fn shower_head_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::new();
    let segments = 64;

    for i in 0..segments {
        let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;

        let x_world = WIDTH / 2.0 + EMITTER_RADIUS * theta.cos();
        let y_world = HEIGHT;

        vertices.push(Vertex {
            position: world_to_screen(x_world, y_world),
            colour: [1.0, 1.0, 0.0]
        });
    }

    vertices
}

pub fn particles_to_vertices(particles: &[Particle], render_mode: &RenderMode, render_stride: usize) -> Vec<Vertex> {
    particles.iter().step_by(render_stride).map(|p| {
        let colour = match render_mode {
            RenderMode::Thermal => temperature_colour(p.temperature),
            RenderMode::Mass => mass_colour(p.mass)
        };

        Vertex {
            position: world_to_screen(p.x, p.y),
            colour
        }
    })
    .collect()
}