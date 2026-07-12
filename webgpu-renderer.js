/* ============================================================================
   WebGPU renderer for the Schwarzschild glyph ray-tracer.

   Drop-in accelerator for the Canvas-2D `frame()` loop in index.html. It moves
   the ~1.3M geodesic integration steps/frame off the single JS thread onto the
   GPU, in two passes:

     PASS 1 — compute:  one invocation per glyph CELL (cols×rows). Ports the
              exact JS ray-march (bending a = -1.5·h²·r/|r|⁵, symplectic Euler,
              adaptive step, disk crossing, photon-ring) and writes a packed
              (skip, level, glyphColumn) u32 into a cell buffer.
     PASS 2 — render:   a full-screen triangle; the fragment shader maps each
              device pixel to its cell, reads the packed result, and samples the
              pre-rasterised glyph atlas (which already bakes the inferno
              palette per brightness level) — identical output to the 2D blit.

   initWebGPU() returns null if WebGPU is unavailable or init fails, so the
   caller can transparently fall back to the Canvas-2D renderer.

   NOTE: needs a real WebGPU browser (Chrome/Edge 113+). It will NOT run under a
   headless http-server preview.
   ========================================================================== */

// Shared uniform layout (12 vec4 = 192 bytes). Declared in both modules.
const U_STRUCT = /* wgsl */ `
struct U {
  cam   : vec4<f32>,   // xyz camera position, w = fov
  fwd   : vec4<f32>,   // xyz forward,          w = aspect
  rgt   : vec4<f32>,   // xyz right (tilted),   w = rT (reservoir/decode mix)
  up    : vec4<f32>,   // xyz up (tilted),      w = t (seconds)
  geom  : vec4<f32>,   // horizon, diskInner, diskOuter, diskSpan
  phys  : vec4<f32>,   // beam, ringGlow, spin, drift
  bg    : vec4<f32>,   // bgGlow, bgScroll, noiseScale, contrast
  amb   : vec4<f32>,   // ambientLow, ambientHigh, escapeR, bMissSq
  grid  : vec4<f32>,   // cellWDev, cellHDev, Wdev, Hdev
  atlas : vec4<f32>,   // tileWd, tileHd, atlasCols, levels
  misc  : vec4<f32>,   // octaves, fallbackCol, maxSteps, nLines
  dims  : vec4<f32>,   // cols, rows, atlasW, atlasH
};
const PI  : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;
const BACKDROP : vec3<f32> = vec3<f32>(5.0/255.0, 3.0/255.0, 10.0/255.0);
`;

const WGSL_COMPUTE = U_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> codesFlat   : array<u32>;
@group(0) @binding(2) var<storage, read> lineOffsets : array<u32>;
@group(0) @binding(3) var<storage, read> lineLens    : array<u32>;
@group(0) @binding(4) var<storage, read> charCol     : array<i32>;   // ascii -> atlas col, -1 = none
@group(0) @binding(5) var<storage, read_write> cellBuf : array<u32>; // packed per-cell result

fn vhash(ix: i32, iy: i32, iz: i32) -> f32 {
  var n : u32 = u32(ix) * 374761393u + u32(iy) * 668265263u + u32(iz) * 1274126177u;
  n = n ^ (n >> 13u);
  n = n * 1274126177u;
  return f32(n & 0xffffffu) / f32(0x1000000u);
}
fn smoothf(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }
fn valueNoise(x: f32, y: f32, z: f32) -> f32 {
  let xi = floor(x); let yi = floor(y); let zi = floor(z);
  let xf = smoothf(x - xi); let yf = smoothf(y - yi); let zf = smoothf(z - zi);
  let ix = i32(xi); let iy = i32(yi); let iz = i32(zi);
  let c000 = vhash(ix,   iy,   iz);   let c100 = vhash(ix+1, iy,   iz);
  let c010 = vhash(ix,   iy+1, iz);   let c110 = vhash(ix+1, iy+1, iz);
  let c001 = vhash(ix,   iy,   iz+1); let c101 = vhash(ix+1, iy,   iz+1);
  let c011 = vhash(ix,   iy+1, iz+1); let c111 = vhash(ix+1, iy+1, iz+1);
  return mix(mix(mix(c000,c100,xf), mix(c010,c110,xf), yf),
             mix(mix(c001,c101,xf), mix(c011,c111,xf), yf), zf);
}
fn fbm(x: f32, y: f32, z: f32, octaves: i32) -> f32 {
  var sum = 0.0; var amp = 0.5; var freq = 1.0; var norm = 0.0;
  for (var o = 0; o < octaves; o = o + 1) {
    sum  = sum + amp * valueNoise(x * freq, y * freq, z * freq);
    norm = norm + amp; amp = amp * 0.5; freq = freq * 2.0;
  }
  return sum / norm;
}
fn getCode(tx: i32, ty: i32) -> i32 {
  let n  = i32(u.misc.w);
  let li = ((ty % n) + n) % n;
  let m  = i32(lineLens[u32(li)]);
  let ci = ((tx % m) + m) % m;
  return i32(codesFlat[lineOffsets[u32(li)] + u32(ci)]);
}
fn pack(skip: u32, lvl: u32, col: u32) -> u32 { return (skip << 24u) | (lvl << 16u) | (col & 0xffffu); }

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cols = i32(u.dims.x); let rows = i32(u.dims.y);
  let gx = i32(gid.x); let gy = i32(gid.y);
  if (gx >= cols || gy >= rows) { return; }
  let idx = u32(gy * cols + gx);

  let fov = u.cam.w; let asp = u.fwd.w; let rT = u.rgt.w; let t = u.up.w;
  let Rhor = u.geom.x; let diskIn = u.geom.y; let diskOut = u.geom.z; let diskSpan = u.geom.w;
  let beam = u.phys.x; let ringGlow = u.phys.y; let spin = u.phys.z; let drift = u.phys.w;
  let bgGlow = u.bg.x; let bgScroll = u.bg.y; let noiseScale = u.bg.z; let contrast = u.bg.w;
  let ambLow = u.amb.x; let ambHigh = u.amb.y; let escapeR = u.amb.z; let bMissSq = u.amb.w;
  let levels = u.atlas.w; let octaves = i32(u.misc.x);
  let fallbackCol = u32(u.misc.y); let maxSteps = i32(u.misc.z);
  let cam = u.cam.xyz; let fwd = u.fwd.xyz; let rgt = u.rgt.xyz; let up = u.up.xyz;

  // Primary ray through the cell centre.
  let cw = u.grid.x; let chh = u.grid.y; let Wd = u.grid.z; let Hd = u.grid.w;
  let px = f32(gx) * cw; let py = f32(gy) * chh;
  let ndcx = ((px + cw * 0.5) / Wd * 2.0 - 1.0) * asp;
  let ndcy = -((py + chh * 0.5) / Hd * 2.0 - 1.0);
  var d = normalize(fwd + rgt * (ndcx * fov) + up * (ndcy * fov));

  let hvec = cross(cam, d);           // h2 = |cam x d|^2 (conserved)
  let h2 = dot(hvec, hvec);

  var pos = cam; var vel = d;
  var captured = false; var hit = false;
  var rmin = 1e9; var rd = 0.0; var phi = 0.0; var beamDot = 0.0;

  var steps = 0;
  if (h2 <= bMissSq) { steps = i32(round(f32(maxSteps) * (1.0 - rT))); }
  if (steps == 0) { rmin = sqrt(h2); }

  for (var s = 0; s < maxSteps; s = s + 1) {
    if (s >= steps) { break; }
    let r2 = dot(pos, pos);
    let r = sqrt(r2);
    if (r < Rhor) { captured = true; break; }
    if (r < rmin) { rmin = r; }

    let fpull = -1.5 * h2 / (r2 * r2 * r);
    let acc = pos * fpull;

    var dl = r * 0.10;
    if (dl > 0.6) { dl = 0.6; } else if (dl < 0.02) { dl = 0.02; }
    if (pos.y < 1.0 && pos.y > -1.0 && dl > 0.15) { dl = 0.15; }

    vel = vel + acc * dl;
    let npos = pos + vel * dl;

    if ((pos.y > 0.0) != (npos.y > 0.0)) {
      let tc = pos.y / (pos.y - npos.y);
      let ipos = pos + (npos - pos) * tc;
      let rr = sqrt(ipos.x * ipos.x + ipos.z * ipos.z);
      if (rr >= diskIn && rr <= diskOut) {
        hit = true; rd = rr; phi = atan2(ipos.z, ipos.x);
        let losx = cam.x - ipos.x; let losz = cam.z - ipos.z;
        let ll = 1.0 / max(sqrt(losx * losx + losz * losz), 1e-6);
        beamDot = (-ipos.z / rr * losx + ipos.x / rr * losz) * ll;
        break;
      }
    }
    pos = npos;
    if (r > escapeR && dot(pos, vel) > 0.0) { break; }
  }

  if (captured) { cellBuf[idx] = pack(1u, 0u, 0u); return; }   // shadow -> backdrop

  var lvl = 0; var col : i32 = -1; var skip = false;

  if (hit) {
    let tt = (diskOut - rd) / diskSpan;
    let flick = 0.72 + 0.6 * fbm(rd * 0.55, phi * 1.6 + 10.0, t * drift + 5.0, octaves);
    let dopp = 1.0 + beam * beamDot;
    let b = (0.55 + 0.5 * tt) * flick * dopp;
    lvl = clamp(i32(b * levels), 0, i32(levels));
    let tx = i32((phi / TAU) * 220.0 + t * spin * 60.0);
    let ty = i32((rd - diskIn) / diskSpan * 64.0);
    let code = getCode(tx, ty);
    if (code == 32) { col = i32(fallbackCol); } else { col = charCol[code]; }
    if (col < 0) { col = i32(fallbackCol); }
  } else {
    let uvel = normalize(vel);
    let azi = atan2(uvel.x, uvel.z);
    let el  = asin(clamp(uvel.y, -1.0, 1.0));
    let ambn = fbm((azi + 3.0) * noiseScale, (el + 3.0) * noiseScale, t * drift, octaves);
    let dr = rmin - 1.5;
    let rg = ringGlow * (1.0 - rT) * exp(-dr * dr / 0.4);
    let ambv = ambLow + (ambHigh - ambLow) * pow(ambn, contrast);
    let b = ambv * (bgGlow * (1.0 + rT * 1.5)) + rg;
    lvl = clamp(i32(b * levels), 0, i32(levels));
    let tx = i32((azi / TAU + 0.5) * 200.0 + t * bgScroll);
    let ty = i32((el / PI + 0.5) * 120.0 + t * bgScroll * 0.3);
    let code = getCode(tx, ty);
    if (code == 32) {
      if (rg < 0.25) { skip = true; } else { col = i32(fallbackCol); }
    } else {
      col = charCol[code];
      if (col < 0) { skip = true; }
    }
  }

  if (skip) { cellBuf[idx] = pack(1u, 0u, 0u); }
  else { cellBuf[idx] = pack(0u, u32(lvl), u32(col)); }
}
`;

const WGSL_RENDER = U_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(5) var<storage, read> cellBuf : array<u32>;
@group(0) @binding(6) var atlasTex : texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4<f32> };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VSOut; o.pos = vec4(p[vi], 0.0, 1.0); return o;
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let cw = u.grid.x; let chh = u.grid.y;
  let cols = i32(u.dims.x); let rows = i32(u.dims.y);
  let gx = i32(floor(frag.x / cw));
  let gy = i32(floor(frag.y / chh));
  if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) { return vec4(BACKDROP, 1.0); }
  let packed = cellBuf[u32(gy * cols + gx)];
  if (((packed >> 24u) & 0xffu) != 0u) { return vec4(BACKDROP, 1.0); }
  let lvl = f32((packed >> 16u) & 0xffu);
  let col = f32(packed & 0xffffu);

  let tileWd = u.atlas.x; let tileHd = u.atlas.y;
  let lx = frag.x - f32(gx) * cw;
  let ly = frag.y - f32(gy) * chh;
  let ax = i32(col * tileWd + lx * (tileWd / cw));
  let ay = i32(lvl * tileHd + ly * (tileHd / chh));
  let texel = textureLoad(atlasTex, vec2<i32>(ax, ay), 0);
  // atlas glyphs sit on a transparent background -> composite over the backdrop
  return vec4(mix(BACKDROP, texel.rgb, texel.a), 1.0);
}
`;

export async function initWebGPU(opts) {
  const { fieldCanvas, atlasCanvas, codeData } = opts;
  if (!navigator.gpu) { console.info("[webgpu] navigator.gpu unavailable — using Canvas 2D"); return null; }

  let device, ctx, format;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) { console.info("[webgpu] no adapter — using Canvas 2D"); return null; }
    device = await adapter.requestDevice();
    ctx = fieldCanvas.getContext("webgpu");
    if (!ctx) { console.info("[webgpu] no webgpu canvas context — using Canvas 2D"); return null; }
    format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: "opaque" });
  } catch (e) {
    console.warn("[webgpu] init failed — using Canvas 2D:", e);
    return null;
  }

  device.addEventListener?.("uncapturederror", (ev) =>
    console.error("[webgpu] device error:", ev.error?.message || ev.error));

  const mkStorage = (typedArr) => {
    const buf = device.createBuffer({ size: Math.max(4, typedArr.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, typedArr);
    return buf;
  };
  const codesBuf   = mkStorage(new Uint32Array(codeData.codesFlat));
  const offsetsBuf = mkStorage(new Uint32Array(codeData.lineOffsets));
  const lensBuf    = mkStorage(new Uint32Array(codeData.lineLens));
  const charColBuf = mkStorage(new Int32Array(codeData.charCol));

  const uniformBuf = device.createBuffer({ size: 12 * 4 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const computeModule = device.createShaderModule({ code: WGSL_COMPUTE });
  const renderModule  = device.createShaderModule({ code: WGSL_RENDER });
  const computePipeline = device.createComputePipeline({ layout: "auto", compute: { module: computeModule, entryPoint: "cs" } });
  const renderPipeline  = device.createRenderPipeline({
    layout: "auto",
    vertex:   { module: renderModule, entryPoint: "vs" },
    fragment: { module: renderModule, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  let cellBuf = null, atlasTex = null, computeBG = null, renderBG = null;
  let gridCols = 0, gridRows = 0;
  let lastUni = null;                       // remembered for benchGPU()
  const uniformArr = new Float32Array(12 * 4);

  function setGrid(g) {
    gridCols = g.cols; gridRows = g.rows;

    cellBuf?.destroy?.();
    cellBuf = device.createBuffer({ size: Math.max(4, g.cols * g.rows * 4), usage: GPUBufferUsage.STORAGE });

    atlasTex?.destroy?.();
    atlasTex = device.createTexture({
      size: [atlasCanvas.width, atlasCanvas.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: atlasCanvas, flipY: false },
      { texture: atlasTex, premultipliedAlpha: false },
      [atlasCanvas.width, atlasCanvas.height]
    );

    computeBG = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: codesBuf } },
        { binding: 2, resource: { buffer: offsetsBuf } },
        { binding: 3, resource: { buffer: lensBuf } },
        { binding: 4, resource: { buffer: charColBuf } },
        { binding: 5, resource: { buffer: cellBuf } },
      ],
    });
    renderBG = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 5, resource: { buffer: cellBuf } },
        { binding: 6, resource: atlasTex.createView() },
      ],
    });
  }

  function render(uni) {
    if (!cellBuf) return;
    lastUni = uni;
    const a = uniformArr;
    a[0]=uni.camx; a[1]=uni.camy; a[2]=uni.camz; a[3]=uni.fov;
    a[4]=uni.fx;   a[5]=uni.fy;   a[6]=uni.fz;   a[7]=uni.asp;
    a[8]=uni.rx;   a[9]=uni.ry;   a[10]=uni.rz;  a[11]=uni.rT;
    a[12]=uni.upx; a[13]=uni.upy; a[14]=uni.upz; a[15]=uni.t;
    a[16]=uni.Rhor; a[17]=uni.diskIn; a[18]=uni.diskOut; a[19]=uni.diskSpan;
    a[20]=uni.beam; a[21]=uni.ringGlow; a[22]=uni.spin; a[23]=uni.drift;
    a[24]=uni.bgGlow; a[25]=uni.bgScroll; a[26]=uni.noiseScale; a[27]=uni.contrast;
    a[28]=uni.ambientLow; a[29]=uni.ambientHigh; a[30]=uni.escapeR; a[31]=uni.bMissSq;
    a[32]=uni.cellWDev; a[33]=uni.cellHDev; a[34]=uni.Wdev; a[35]=uni.Hdev;
    a[36]=uni.tileWd; a[37]=uni.tileHd; a[38]=uni.atlasCols; a[39]=uni.levels;
    a[40]=uni.octaves; a[41]=uni.fallbackCol; a[42]=uni.maxSteps; a[43]=uni.nLines;
    a[44]=gridCols; a[45]=gridRows; a[46]=atlasCanvas.width; a[47]=atlasCanvas.height;
    device.queue.writeBuffer(uniformBuf, 0, a);

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(computePipeline);
    cp.setBindGroup(0, computeBG);
    cp.dispatchWorkgroups(Math.ceil(gridCols / 8), Math.ceil(gridRows / 8));
    cp.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        loadOp: "clear", storeOp: "store",
        clearValue: { r: 5/255, g: 3/255, b: 10/255, a: 1 },
      }],
    });
    rp.setPipeline(renderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    device.queue.submit([enc.finish()]);
  }

  // Throughput benchmark: submit `iterations` frames back-to-back (no rAF, so
  // it isn't vsync-capped) and wait for the GPU to finish them all. Returns
  // average GPU wall-clock ms per frame (compute + render pass).
  async function benchGPU(iterations = 120) {
    if (!lastUni) return null;
    await device.queue.onSubmittedWorkDone();     // drain anything pending
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) render(lastUni);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) / iterations;
  }

  console.info("[webgpu] renderer active");
  return { setGrid, render, benchGPU, device };
}
