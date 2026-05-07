use glium::{Surface, implement_vertex};

use crate::simulation::{DEPTH, EMITTER_RADIUS, HEIGHT, Particle, WIDTH};

#[derive(Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
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

fn world_to_3d(x: f32, y: f32, z: f32) -> [f32; 3] {
    [
        x - WIDTH / 2.0,
        y - HEIGHT / 2.0,
        z - DEPTH / 2.0
    ]
}

pub fn cubicle_vertices() -> Vec<Vertex> {
    let c = [1.0, 1.0, 1.0];

    let p000 = world_to_3d(0.0, 0.0, 0.0);
    let p100 = world_to_3d(WIDTH, 0.0, 0.0);
    let p010 = world_to_3d(0.0, HEIGHT, 0.0);
    let p110 = world_to_3d(WIDTH, HEIGHT, 0.0);

    let p001 = world_to_3d(0.0, 0.0, DEPTH);
    let p101 = world_to_3d(WIDTH, 0.0, DEPTH);
    let p011 = world_to_3d(0.0, HEIGHT, DEPTH);
    let p111 = world_to_3d(WIDTH, HEIGHT, DEPTH);

    vec![
        // back rectangle
        Vertex { position: p000, colour: c },
        Vertex { position: p100, colour: c },
        Vertex { position: p100, colour: c },
        Vertex { position: p110, colour: c },
        Vertex { position: p110, colour: c },
        Vertex { position: p010, colour: c },
        Vertex { position: p010, colour: c },
        Vertex { position: p000, colour: c },

        // front rectangle
        Vertex { position: p001, colour: c },
        Vertex { position: p101, colour: c },
        Vertex { position: p101, colour: c },
        Vertex { position: p111, colour: c },
        Vertex { position: p111, colour: c },
        Vertex { position: p011, colour: c },
        Vertex { position: p011, colour: c },
        Vertex { position: p001, colour: c },

        // connecting edges
        Vertex { position: p000, colour: c },
        Vertex { position: p001, colour: c },
        Vertex { position: p100, colour: c },
        Vertex { position: p101, colour: c },
        Vertex { position: p110, colour: c },
        Vertex { position: p111, colour: c },
        Vertex { position: p010, colour: c },
        Vertex { position: p011, colour: c },
    ]
}

pub fn shower_head_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::new();
    let segments = 256;
    let c = [1.0, 1.0, 0.0];

    // Centre
    vertices.push(Vertex { position: world_to_3d(WIDTH / 2.0, HEIGHT, DEPTH / 2.0), colour: (c) });

    for i in 0..segments {
        let theta = (i as f32 / segments as f32) * std::f32::consts::TAU;

        let x_world = WIDTH / 2.0 + EMITTER_RADIUS * theta.cos();
        let y_world = HEIGHT;
        let z_world = DEPTH / 2.0 + EMITTER_RADIUS * theta.sin();

        vertices.push(Vertex {
            position: world_to_3d(x_world, y_world, z_world),
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
            position: world_to_3d(p.x, p.y, p.z),
            colour
        }
    })
    .collect()
}