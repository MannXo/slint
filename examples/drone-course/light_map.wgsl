// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: MIT

// Pools of light added up into a floor's light map, see `Renderer::draw_light_map`.

struct Pools {
    // Center (xy) and half size (zw) in clip space.
    rects: array<vec4<f32>, 64>,
    // Linear color (rgb).
    colors: array<vec4<f32>, 64>,
}

// Arrays of vectors rather than of structs: NXP's Vivante driver miscompiles reading a
// struct from an array.
@group(0) @binding(0) var<uniform> pools: Pools;

struct Varying {
    @builtin(position) clip: vec4<f32>,
    // From -1 to 1 across the pool.
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
}

// The corners come from an array: the Vivante driver also miscompiles vectors built from
// vertex inputs.
@vertex
fn vs_pool(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Varying {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex];
    let rect = pools.rects[instance];
    var out: Varying;
    out.clip = vec4<f32>(rect.xy + corner * rect.zw, 0.0, 1.0);
    out.local = corner;
    out.color = pools.colors[instance].rgb;
    return out;
}

@fragment
fn fs_pool(in: Varying) -> @location(0) vec4<f32> {
    // The alpha of `textures::radial`.
    let r = min(length(in.local), 1.0);
    let alpha = select(0.45 * (1.0 - (r - 0.3) / 0.7), 1.0 - r / 0.3 * 0.55, r < 0.3);
    return vec4<f32>(in.color * alpha, 1.0);
}
