use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute,
    queue,                              
    style::{Color, SetForegroundColor}, 
    terminal::{self, ClearType},        
};
use rand::Rng;
use std::borrow::Cow;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

// --- 基础配置 ---
const NUM_PARTICLES: u32 = 65536; // 6.5万粒子
const GRID_SIZE: f32 = 4.0;
const WORLD_EXTENT: f32 = 80.0;

const SHADER_SOURCE: &str = r#"
const BLOCK_SIZE: u32 = 256;
const VISUAL_RANGE_SQ: f32 = 12.0;

struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct SimParams {
    time: f32,
    dt: f32,
    num_particles: u32,
    world_extent: f32,
    grid_size: f32,
    grid_dim: u32,
    screen_w: u32,
    screen_h: u32,
};

struct SortKeyValuePair { key: u32, value: u32 };

@group(0) @binding(0) var<uniform> params: SimParams;
@group(1) @binding(0) var<storage, read> particles_src: array<Particle>;
@group(1) @binding(1) var<storage, read_write> particles_dst: array<Particle>;
@group(2) @binding(0) var<storage, read_write> sort_buffers: array<SortKeyValuePair>;
@group(2) @binding(1) var<storage, read_write> cell_start: array<u32>;
@group(2) @binding(2) var<storage, read_write> cell_end: array<u32>;
// Screen: Atomic u32
@group(3) @binding(0) var<storage, read_write> screen_buffer: array<atomic<u32>>;

// --- Helpers ---
fn get_grid_pos(pos: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor((pos + params.world_extent) / params.grid_size));
}
fn get_grid_hash(grid_pos: vec3<i32>) -> u32 {
    let dim = i32(params.grid_dim);
    let x = clamp(grid_pos.x, 0, dim - 1);
    let y = clamp(grid_pos.y, 0, dim - 1);
    let z = clamp(grid_pos.z, 0, dim - 1);
    return u32(x + y * dim + z * dim * dim);
}

// --- Kernels ---
@compute @workgroup_size(256)
fn build_grid_keys(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_particles) { return; }
    sort_buffers[idx].key = get_grid_hash(get_grid_pos(particles_src[idx].pos.xyz));
    sort_buffers[idx].value = idx;
}

struct SortParams { j: u32, k: u32 };
@group(2) @binding(3) var<uniform> sort_params: SortParams;
@compute @workgroup_size(256)
fn bitonic_sort_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_particles) { return; }
    let j = sort_params.j;
    let k = sort_params.k;
    let ixj = i ^ j;
    if (ixj > i) {
        let ki = sort_buffers[i].key;
        let kj = sort_buffers[ixj].key;
        if (((i & k) == 0u && ki > kj) || ((i & k) != 0u && ki < kj)) {
            let temp = sort_buffers[i];
            sort_buffers[i] = sort_buffers[ixj];
            sort_buffers[ixj] = temp;
        }
    }
}

@compute @workgroup_size(256)
fn clear_grid_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.grid_dim * params.grid_dim * params.grid_dim;
    if (idx < total) { cell_start[idx] = 0xFFFFFFFFu; cell_end[idx] = 0xFFFFFFFFu; }
}
@compute @workgroup_size(256)
fn find_cell_start(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_particles) { return; }
    let key = sort_buffers[idx].key;
    if (idx == 0u || key != sort_buffers[idx - 1u].key) { cell_start[key] = idx; }
    if (idx == params.num_particles - 1u || key != sort_buffers[idx + 1u].key) { cell_end[key] = idx + 1u; }
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_particles) { return; }

    let p = particles_src[idx];
    var pos = p.pos.xyz;
    var vel = p.vel.xyz;
    let grid_pos = get_grid_pos(pos);
    let dim = i32(params.grid_dim);

    var force = vec3<f32>(0.0);
    var cohesion = vec3<f32>(0.0);
    var alignment = vec3<f32>(0.0);
    var separation = vec3<f32>(0.0);
    var count = 0u;

    for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
        let n_pos = grid_pos + vec3(x, y, z);
        if (any(n_pos < vec3(0)) || any(n_pos >= vec3(dim))) { continue; }
        let key = get_grid_hash(n_pos);
        let start = cell_start[key];
        if (start == 0xFFFFFFFFu) { continue; }
        let end = cell_end[key];
        let loop_end = min(end, start + 32u); 
        for (var k = start; k < loop_end; k++) {
            let other_idx = sort_buffers[k].value;
            if (other_idx == idx) { continue; }
            let other_p = particles_src[other_idx];
            let diff = pos - other_p.pos.xyz;
            let dist_sq = dot(diff, diff);
            if (dist_sq < 16.0 && dist_sq > 0.001) {
                let dist = sqrt(dist_sq);
                separation += diff / (dist * dist + 0.1);
                cohesion += other_p.pos.xyz;
                alignment += other_p.vel.xyz;
                count++;
            }
        }
    }}}

    if (count > 0u) {
        let fc = f32(count);
        force += (cohesion / fc - pos) * 0.02;
        force += (alignment / fc - vel) * 0.04;
        force += separation * 0.15;
    }

    let center_dist = length(pos.xz);
    let target_r = 35.0;
    let radial_force = (target_r - center_dist) * 0.1;
    let tangent = normalize(vec3(-pos.z, 0.0, pos.x));
    
    force += vec3(pos.x, 0.0, pos.z) * (radial_force / (center_dist + 1.0)); 
    force += tangent * 0.5; 
    let wave = sin(params.time + center_dist * 0.1) * 10.0;
    force.y += (wave - pos.y) * 0.1;

    vel += force * params.dt;
    let speed = length(vel);
    if (speed > 15.0) { vel = normalize(vel) * 15.0; }
    pos += vel * params.dt;

    let bound = 60.0;
    if (pos.x > bound) { pos.x -= bound*2.0; } else if (pos.x < -bound) { pos.x += bound*2.0; }
    if (pos.y > bound) { pos.y -= bound*2.0; } else if (pos.y < -bound) { pos.y += bound*2.0; }
    if (pos.z > bound) { pos.z -= bound*2.0; } else if (pos.z < -bound) { pos.z += bound*2.0; }

    particles_dst[idx].pos = vec4(pos, p.pos.w);
    particles_dst[idx].vel = vec4(vel, 0.0);
}

@compute @workgroup_size(256)
fn clear_screen(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // 动态屏幕大小
    if (idx < params.screen_w * params.screen_h) { atomicStore(&screen_buffer[idx], 0u); }
}

@compute @workgroup_size(256)
fn render(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_particles) { return; }

    let p = particles_dst[idx];
    let pos = p.pos.xyz;

    let t = params.time * 0.3;
    let cam_pos = vec3(sin(t)*45.0, cos(t*0.5)*15.0, sin(t*2.0)*20.0);
    let cam_target = vec3(sin(t+0.5)*45.0, cos((t+0.5)*0.5)*15.0, sin((t+0.5)*2.0)*20.0);

    let f = normalize(cam_target - cam_pos);
    let right = normalize(cross(f, vec3(0.0, 1.0, 0.0)));
    let up = cross(right, f);

    let v_pos = pos - cam_pos;
    let z_cam = dot(v_pos, f);
    let x_cam = dot(v_pos, right);
    let y_cam = dot(v_pos, up);

    if (z_cam < 0.5) { return; }

    let fov_factor = 0.8;
    let proj_x = x_cam / (z_cam * fov_factor);
    let proj_y = y_cam / (z_cam * fov_factor);

    let sw = f32(params.screen_w);
    let sh = f32(params.screen_h);
    let ss_x = i32((proj_x * 0.5 + 0.5) * sw);
    let ss_y = i32((-proj_y * 0.25 + 0.5) * sh);

    if (ss_x >= 0 && ss_x < i32(params.screen_w) && ss_y >= 0 && ss_y < i32(params.screen_h)) {
        let px_idx = u32(ss_y) * params.screen_w + u32(ss_x);

        let focal_dist = 25.0; 
        let coc = abs(z_cam - focal_dist); 
        let blur_level = clamp(coc / 15.0, 0.0, 1.0);

        var char_idx = 0u;
        if (blur_level > 0.8) { char_idx = 3u; }      
        else if (blur_level > 0.5) { char_idx = 2u; }
        else if (blur_level > 0.2) { char_idx = 1u; }
        else { char_idx = 0u; } 

        let fog = clamp(1.0 - (z_cam / 90.0), 0.0, 1.0);
        var palette = 0u;
        let speed = length(p.vel.xyz);
        
        if (fog < 0.3) {
            palette = 0u; char_idx = 3u; 
        } else if (speed > 8.0 || blur_level < 0.2) {
            palette = 2u; 
        } else {
            palette = 1u; 
        }

        let inv_depth = u32((1.0 / z_cam) * 100000.0);
        if (inv_depth == 0u) { return; }

        let packed = (inv_depth << 16u) | (char_idx << 8u) | palette;
        atomicMax(&screen_buffer[px_idx], packed);
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    time: f32,
    dt: f32,
    num_particles: u32,
    world_extent: f32,
    grid_size: f32,
    grid_dim: u32,
    screen_w: u32,
    screen_h: u32,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SortParams {
    j: u32,
    k: u32,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    pos: [f32; 4],
    vel: [f32; 4],
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    // 1. 初始化终端 (Alternate Screen)
    let mut stdout = io::stdout();
    execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide).unwrap();
    terminal::enable_raw_mode().unwrap();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Dev"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .unwrap();

    // 2. 初始化粒子 (修复：random_range -> gen_range)
    let mut rng = rand::rng();
    let mut particles = Vec::with_capacity(NUM_PARTICLES as usize);
    for _ in 0..NUM_PARTICLES {
        particles.push(Particle {
            pos: [
                rng.random_range(-40.0..40.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-40.0..40.0),
                0.0,
            ],
            vel: [0.0; 4],
        });
    }

    let grid_dim = ((WORLD_EXTENT * 2.0) / GRID_SIZE).ceil() as u32;
    let total_cells = grid_dim * grid_dim * grid_dim;

    // 3. 创建静态 Buffer
    let create_buf = |lbl, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: lbl,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let create_buf_init = |lbl, content, usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: lbl,
            contents: content,
            usage,
        })
    };

    let p_a = create_buf_init(
        Some("PA"),
        bytemuck::cast_slice(&particles),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let p_b = create_buf(
        Some("PB"),
        (32 * NUM_PARTICLES) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let sort_kv = create_buf(
        Some("KV"),
        (8 * NUM_PARTICLES) as u64,
        wgpu::BufferUsages::STORAGE,
    );
    let cell_s = create_buf(
        Some("CS"),
        (4 * total_cells) as u64,
        wgpu::BufferUsages::STORAGE,
    );
    let cell_e = create_buf(
        Some("CE"),
        (4 * total_cells) as u64,
        wgpu::BufferUsages::STORAGE,
    );
    let u_sim = create_buf(
        Some("USim"),
        std::mem::size_of::<SimParams>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let u_sort = create_buf(
        Some("USort"),
        8,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );

    // 4. BindLayouts
    let bgl = |entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries,
        })
    };
    let entry_uni = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let entry_sto = |b, r| wgpu::BindGroupLayoutEntry {
        binding: b,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: r },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let bl0 = bgl(&[entry_uni]);
    let bl1 = bgl(&[entry_sto(0, true), entry_sto(1, false)]);
    let bl2 = bgl(&[
        entry_sto(0, false),
        entry_sto(1, false),
        entry_sto(2, false),
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]);
    let bl3 = bgl(&[entry_sto(0, false)]); // Screen Buffer Layout

    let bg = |l, e: &[wgpu::BindGroupEntry]| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: l,
            entries: e,
        })
    };
    let bg0 = bg(
        &bl0,
        &[wgpu::BindGroupEntry {
            binding: 0,
            resource: u_sim.as_entire_binding(),
        }],
    );
    let bg1_a = bg(
        &bl1,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p_b.as_entire_binding(),
            },
        ],
    );
    let bg1_b = bg(
        &bl1,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: p_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: p_a.as_entire_binding(),
            },
        ],
    );
    let bg2 = bg(
        &bl2,
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sort_kv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cell_s.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cell_e.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: u_sort.as_entire_binding(),
            },
        ],
    );

    // Pipelines
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bl0, &bl1, &bl2, &bl3],
        ..Default::default()
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
    });
    let mk_ppl = |ep| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let (p_keys, p_sort, p_clr, p_fnd, p_sim, p_scr, p_rnd) = (
        mk_ppl("build_grid_keys"),
        mk_ppl("bitonic_sort_step"),
        mk_ppl("clear_grid_indices"),
        mk_ppl("find_cell_start"),
        mk_ppl("simulate"),
        mk_ppl("clear_screen"),
        mk_ppl("render"),
    );

    // --- 动态变量 ---
    let mut i = 0;
    let start = Instant::now();
    let mut handle = io::BufWriter::with_capacity(1024 * 1024, stdout.lock());

    // 动态屏幕缓冲区资源 (Option 包装以便重建)
    let mut screen_buffer: Option<wgpu::Buffer> = None;
    let mut stage_buffer: Option<wgpu::Buffer> = None;
    let mut bg3: Option<wgpu::BindGroup> = None;
    let mut current_term_size = (0, 0); // (cols, rows)

    loop {
        // 5. 检测终端大小变化
        let term_size = terminal::size().unwrap();
        let (term_w, term_h) = (term_size.0 as u32, term_size.1 as u32);

        // 计算 80% 渲染区域 + 10% Padding
        let render_w = (term_w as f32 * 0.8) as u32;
        let render_h = (term_h as f32 * 0.8) as u32;
        let pad_x = (term_w - render_w) / 2;
        let pad_y = (term_h - render_h) / 2;

        let screen_size_bytes = (render_w * render_h * 4) as u64;

        // 如果尺寸变了，或者第一次运行
        if term_size != current_term_size {
            // 重建 Buffer
            let new_scr = create_buf(
                Some("Scr"),
                screen_size_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let new_stg = create_buf(
                Some("Stg"),
                screen_size_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );

            // 重建 BindGroup
            let new_bg3 = bg(
                &bl3,
                &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: new_scr.as_entire_binding(),
                }],
            );

            screen_buffer = Some(new_scr);
            stage_buffer = Some(new_stg);
            bg3 = Some(new_bg3);
            current_term_size = term_size;

            // 清屏一次防止残影
            execute!(handle, terminal::Clear(ClearType::All)).unwrap();
        }

        let t = start.elapsed().as_secs_f32();

        // 更新 Uniform (传入当前的 render_w/h)
        queue.write_buffer(
            &u_sim,
            0,
            bytemuck::bytes_of(&SimParams {
                time: t,
                dt: 0.016,
                num_particles: NUM_PARTICLES,
                world_extent: WORLD_EXTENT,
                grid_size: GRID_SIZE,
                grid_dim, 
                screen_w: render_w,
                screen_h: render_h,
            }),
        );

        let bg1 = if i % 2 == 0 { &bg1_a } else { &bg1_b };
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Dispatch Helpers
        let wg_p = NUM_PARTICLES.div_ceil(256);
        let wg_scr = (render_w * render_h).div_ceil(256);

        // --- Compute Passes ---
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bg0, &[]);
            cp.set_bind_group(1, bg1, &[]);
            cp.set_bind_group(2, &bg2, &[]);
            cp.set_bind_group(3, bg3.as_ref().unwrap(), &[]);

            // Keys
            cp.set_pipeline(&p_keys);
            cp.dispatch_workgroups(wg_p, 1, 1);
        }

        // Sort (需要多次 dispatch，每次都要重建 pass 因为要插入 write_buffer)
        let mut k = 2;
        while k <= NUM_PARTICLES {
            let mut j = k >> 1;
            while j > 0 {
                queue.write_buffer(&u_sort, 0, bytemuck::bytes_of(&SortParams { j, k }));
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cp.set_pipeline(&p_sort);
                cp.set_bind_group(0, &bg0, &[]);
                cp.set_bind_group(1, bg1, &[]);
                cp.set_bind_group(2, &bg2, &[]);
                cp.set_bind_group(3, bg3.as_ref().unwrap(), &[]);
                cp.dispatch_workgroups(wg_p, 1, 1);
                j >>= 1;
            }
            k <<= 1;
        }

        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bg0, &[]);
            cp.set_bind_group(1, bg1, &[]);
            cp.set_bind_group(2, &bg2, &[]);
            cp.set_bind_group(3, bg3.as_ref().unwrap(), &[]);

            // Clear Grid & Find Cells
            cp.set_pipeline(&p_clr);
            cp.dispatch_workgroups(total_cells.div_ceil(256), 1, 1);
            cp.set_pipeline(&p_fnd);
            cp.dispatch_workgroups(wg_p, 1, 1);

            // Sim
            cp.set_pipeline(&p_sim);
            cp.dispatch_workgroups(wg_p, 1, 1);

            // Clear Screen & Render
            cp.set_pipeline(&p_scr);
            cp.dispatch_workgroups(wg_scr, 1, 1);
            cp.set_pipeline(&p_rnd);
            cp.dispatch_workgroups(wg_p, 1, 1);
        }

        enc.copy_buffer_to_buffer(
            screen_buffer.as_ref().unwrap(),
            0,
            stage_buffer.as_ref().unwrap(),
            0,
            screen_size_bytes,
        );
        queue.submit(Some(enc.finish()));

        // --- Readback & Print ---
        let slice = stage_buffer.as_ref().unwrap().slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let buf: &[u32] = bytemuck::cast_slice(&data);

            // 优化输出：只在渲染区域内移动光标
            for y in 0..render_h {
                // 移动光标到 (Padding X, Padding Y + Current Row)
                queue!(handle, cursor::MoveTo(pad_x as u16, (pad_y + y) as u16)).unwrap();

                for x in 0..render_w {
                    let v = buf[(y * render_w + x) as usize];
                    if v == 0 {
                        write!(handle, " ").unwrap();
                    } else {
                        let char_idx = (v >> 8) & 0xFF;
                        let palette = v & 0xFF;

                        // Set Color (RGB)
                        match palette {
                            0 => queue!(
                                handle,
                                SetForegroundColor(Color::Rgb {
                                    r: 30,
                                    g: 30,
                                    b: 60
                                })
                            )
                            .unwrap(),
                            1 => queue!(
                                handle,
                                SetForegroundColor(Color::Rgb {
                                    r: 0,
                                    g: 150,
                                    b: 200
                                })
                            )
                            .unwrap(),
                            2 => queue!(
                                handle,
                                SetForegroundColor(Color::Rgb {
                                    r: 255,
                                    g: 100,
                                    b: 200
                                })
                            )
                            .unwrap(),
                            _ => {}
                        }

                        match char_idx {
                            0 => write!(handle, "█").unwrap(),
                            1 => write!(handle, "#").unwrap(),
                            2 => write!(handle, ":").unwrap(),
                            3 => write!(handle, ".").unwrap(),
                            _ => write!(handle, " ").unwrap(),
                        }
                    }
                }
            }
            // 复位颜色
            queue!(handle, SetForegroundColor(Color::White)).unwrap();
            handle.flush().unwrap();
            drop(data);
            stage_buffer.as_ref().unwrap().unmap();
        }

        // 简单的输入检查（按 ESC 或 Q 退出）
        if event::poll(Duration::from_millis(0)).unwrap()
            && let Event::Key(key) = event::read().unwrap()
            && (key.code == KeyCode::Esc || key.code == KeyCode::Char('q'))
        {
            break;
        }

        i += 1;
    }

    // 退出清理：恢复光标，离开 Alternate Screen
    execute!(io::stdout(), cursor::Show, terminal::LeaveAlternateScreen).unwrap();
    terminal::disable_raw_mode().unwrap();
}
