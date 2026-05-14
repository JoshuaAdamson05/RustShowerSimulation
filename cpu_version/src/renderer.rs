use glium::{Surface, implement_vertex};
use std::collections::HashSet;
use std::cmp::Ordering;

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
    if mass < 1.1 {
        [0.0, 0.5, 1.0]
    } else if mass < 2.1 {
        [0.0, 1.0, 0.0]
    } else if mass < 3.1 {
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

pub fn particles_to_vertices(
    particles: &[Particle],
    render_mode: &RenderMode,
    render_stride: usize,
) -> Vec<Vertex> {
    let stride = render_stride.max(1);

    match render_mode {
        RenderMode::Thermal => {
            particles
                .iter()
                .step_by(stride)
                .map(|p| {
                    Vertex {
                        position: world_to_3d(p.x, p.y, p.z),
                        colour: temperature_colour(p.temperature),
                    }
                })
                .collect()
        }

        RenderMode::Mass => {
            // Keep roughly the same number of rendered particles as the strided render.
            let target_count = (particles.len() + stride - 1) / stride;

            let mut selected_indices: Vec<usize> = Vec::with_capacity(target_count);
            let mut used_indices: HashSet<usize> = HashSet::new();

            // First, prioritise particles that have actually merged/gained mass.
            let mut merged_indices: Vec<usize> = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.mass > 1.05)
                .map(|(index, _)| index)
                .collect();

            // Highest mass first, so the most useful mass examples are selected.
            merged_indices.sort_by(|&a, &b| {
                particles[b]
                    .mass
                    .partial_cmp(&particles[a].mass)
                    .unwrap_or(Ordering::Equal)
            });

            // Use up to half the render budget on merged particles.
            for index in merged_indices.into_iter().take(target_count / 2) {
                if used_indices.insert(index) {
                    selected_indices.push(index);
                }
            }

            // Fill the rest with normal strided particles so the overall particle field still appears.
            for index in (0..particles.len()).step_by(stride) {
                if selected_indices.len() >= target_count {
                    break;
                }

                if used_indices.insert(index) {
                    selected_indices.push(index);
                }
            }

            // Draw lower mass first and higher mass later, making coloured merged particles more visible.
            selected_indices.sort_by(|&a, &b| {
                particles[a]
                    .mass
                    .partial_cmp(&particles[b].mass)
                    .unwrap_or(Ordering::Equal)
            });

            selected_indices
                .iter()
                .map(|&index| {
                    let p = &particles[index];

                    Vertex {
                        position: world_to_3d(p.x, p.y, p.z),
                        colour: mass_colour(p.mass),
                    }
                })
                .collect()
        }
    }
}