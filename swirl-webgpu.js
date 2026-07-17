/* ============================================================================
   WebGPU renderer for the reservoir ASCII swirl flow-field.

   Drop-in accelerator for the Canvas-2D swirl loop in index.html. The 2D path
   cost thousands of per-cell drawImage() calls per frame on the main thread
   (plus 2 more per cell for the chromatic-aberration ghosts), which is what
   kept the reservoir under 60fps. This moves all of it onto the GPU:

     PASS 1 — compute: one invocation per glyph CELL (cols×rows). Ports the exact
              JS math — inverse-mapped swirl, cursor magnifier lens, fBm ambient
              + cursor heat — and writes a packed (skip, level, glyphColumn) u32.
     PASS 2 — render: a full-screen triangle. Each device pixel maps to its cell,
              reads the packed result and samples the pre-rasterised glyph atlas
              (which already bakes the inferno palette per brightness level).
              The chromatic aberration is done here as a per-pixel GATHER: the
              red/cyan ghosts are sampled at ±(warped offset) instead of being
              re-blitted per cell — same look, a fraction of the cost, and it is
              smoother because it is evaluated per pixel rather than per glyph.

   initSwirlGPU() returns null if WebGPU is unavailable or init fails, so the
   caller transparently falls back to the Canvas-2D renderer.
   ========================================================================== */

// Shared uniform layout (12 vec4 = 192 bytes). Declared in both modules.
const U_STRUCT = /* wgsl */ `
struct U {
  grid  : vec4<f32>,   // cellWDev, cellHDev, Wdev, Hdev
  dims  : vec4<f32>,   // cols, rows, atlasW, atlasH
  atlas : vec4<f32>,   // tileWd, tileHd, atlasCols, levels
  time  : vec4<f32>,   // t (swirl seconds), tCA (real seconds), aspect, nLines
  swrl  : vec4<f32>,   // swirl, swirlPower, lensMag, lensRadiusDev
  cur   : vec4<f32>,   // cursorXDev, cursorYDev, heatInv2Sig2Dev, heatGain
  noise : vec4<f32>,   // noiseScale, noiseTime, octaves, contrast
  amb   : vec4<f32>,   // ambientLow, ambientHigh, panCols, panRows
  pan   : vec4<f32>,   // panNX, panNY, -, -
  ca1   : vec4<f32>,   // caStrengthDev, caInv2Dev, caRadiusDev, caSmoke
  ca2   : vec4<f32>,   // caSmokeFreqDev, caDrift, caEdge, caAlpha
  ca3   : vec4<f32>,   // caCullSqDev, -, -, -
};
const PI : f32 = 3.14159265359;
const BACKDROP : vec3<f32> = vec3<f32>(27.0/255.0, 18.0/255.0, 34.0/255.0);
const CA_RED   : vec3<f32> = vec3<f32>(255.0/255.0, 42.0/255.0,  90.0/255.0);
const CA_CYAN  : vec3<f32> = vec3<f32>(42.0/255.0, 200.0/255.0, 255.0/255.0);

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
`;

const WGSL_COMPUTE = U_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> u : U;
@group(0) @binding(1) var<storage, read> codesFlat   : array<u32>;
@group(0) @binding(2) var<storage, read> lineOffsets : array<u32>;
@group(0) @binding(3) var<storage, read> lineLens    : array<u32>;
@group(0) @binding(4) var<storage, read> charCol     : array<i32>;   // ascii -> atlas col, -1 = none
@group(0) @binding(5) var<storage, read_write> cellBuf : array<u32>;

fn getCode(tx: i32, ty: i32) -> i32 {
  let n  = i32(u.time.w);
  let li = ((ty % n) + n) % n;
  let m  = i32(lineLens[u32(li)]);
  let ci = ((tx % m) + m) % m;
  return i32(codesFlat[lineOffsets[u32(li)] + u32(ci)]);
}
// bits: col 0-15, lvl 16-23, skip 24
fn pack(skip: u32, lvl: u32, col: u32) -> u32 {
  return (skip << 24u) | (lvl << 16u) | (col & 0xffffu);
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cols = i32(u.dims.x); let rows = i32(u.dims.y);
  let gx = i32(gid.x); let gy = i32(gid.y);
  if (gx >= cols || gy >= rows) { return; }
  let idx = u32(gy * cols + gx);

  let cw = u.grid.x; let chh = u.grid.y;
  let asp = u.time.z; let t = u.time.x;
  let swirl = u.swrl.x; let swirlPow = u.swrl.y; let lensMag = u.swrl.z; let lensR = u.swrl.w;
  let cx = u.cur.x; let cy = u.cur.y;

  let px = f32(gx) * cw; let py = f32(gy) * chh;
  let ldx = px - cx; let ldy = py - cy;
  let ld2 = ldx * ldx + ldy * ldy;

  // Geometry: plain grid outside the lens, warped (magnified) inside it.
  var dist: f32; var ang0: f32; var distPow: f32;
  if (ld2 < lensR * lensR) {
    let r = sqrt(ld2) / lensR;
    let scale = 1.0 - lensMag * (1.0 - smoothf(r));
    let uu = (cx + ldx * scale) / cw / f32(cols);
    let vv = (cy + ldy * scale) / chh / f32(rows);
    let nx = (uu - 0.5) * asp; let ny = (vv - 0.5);
    dist = sqrt(nx * nx + ny * ny); ang0 = atan2(ny, nx); distPow = pow(dist, swirlPow);
  } else {
    let nx = (f32(gx) / f32(cols) - 0.5) * asp; let ny = (f32(gy) / f32(rows) - 0.5);
    dist = sqrt(nx * nx + ny * ny); ang0 = atan2(ny, nx); distPow = pow(dist, swirlPow);
  }

  // Animated inverse-mapped swirl -> sample the source text.
  let ang = ang0 + swirl * distPow * t;
  let su = cos(ang) * dist; let sv = sin(ang) * dist;
  let tx = i32(floor((su / asp + 0.5) * f32(cols) + u.amb.z));
  let ty = i32(floor((sv + 0.5) * f32(rows) + u.amb.w));
  let code = getCode(tx, ty);
  if (code == 32) { cellBuf[idx] = pack(1u, 0u, 0u); return; }   // space -> backdrop

  // Ambient fBm brightness, lifted toward yellow by the cursor heat spot.
  let af = fbm(su * u.noise.x + 100.0 + u.pan.x, sv * u.noise.x + 100.0 + u.pan.y,
               t * u.noise.y, i32(u.noise.z));
  let amb = u.amb.x + (u.amb.y - u.amb.x) * pow(af, u.noise.w);
  let heat = u.cur.w * exp(-ld2 * u.cur.z);
  let b = 1.0 - (1.0 - amb) * (1.0 - heat);

  let levels = u.atlas.w;
  let lvl = clamp(i32(b * levels), 0, i32(levels));
  var col = 0;
  if (code >= 0 && code < 128) { col = charCol[u32(code)]; } else { col = -1; }
  if (col < 0) { cellBuf[idx] = pack(1u, 0u, 0u); return; }
  cellBuf[idx] = pack(0u, u32(lvl), u32(col));
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

// Sample the glyph covering an arbitrary device-pixel position (rgb = palette
// colour baked in the atlas, a = glyph coverage). Used for the base pixel AND
// for the chromatic-aberration ghost gathers.
fn glyphAt(q: vec2<f32>) -> vec4<f32> {
  let cw = u.grid.x; let chh = u.grid.y;
  let cols = i32(u.dims.x); let rows = i32(u.dims.y);
  let gx = i32(floor(q.x / cw)); let gy = i32(floor(q.y / chh));
  if (gx < 0 || gy < 0 || gx >= cols || gy >= rows) { return vec4<f32>(0.0); }
  let packed = cellBuf[u32(gy * cols + gx)];
  if (((packed >> 24u) & 1u) != 0u) { return vec4<f32>(0.0); }   // skipped cell
  let lvl = f32((packed >> 16u) & 0xffu);
  let col = f32(packed & 0xffffu);
  let tileWd = u.atlas.x; let tileHd = u.atlas.y;
  let lx = q.x - f32(gx) * cw;
  let ly = q.y - f32(gy) * chh;
  let ax = i32(col * tileWd + lx * (tileWd / cw));
  let ay = i32(lvl * tileHd + ly * (tileHd / chh));
  return textureLoad(atlasTex, vec2<i32>(ax, ay), 0);
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = frag.xy;
  let base = glyphAt(p);
  var color = mix(BACKDROP, base.rgb, base.a);

  // Smoky chromatic aberration, gathered per pixel near the cursor. The split
  // direction is bent by drifting noise (the wispy smoke) exactly as in the 2D
  // pass; here we sample the ghosts instead of re-blitting them.
  let d = p - vec2<f32>(u.cur.x, u.cur.y);
  let d2 = dot(d, d);
  if (d2 < u.ca3.x) {
    let dd = max(sqrt(d2), 1e-4);
    let fall = exp(-d2 * u.ca1.y);
    let r = dd / u.ca1.z;
    var edge = 0.0;
    if (r < 1.0) { edge = u.ca2.z * 4.0 * r * (1.0 - r); }   // glassy refracted rim
    let mag = fall + edge;
    let n = fbm(p.x * u.ca2.x, p.y * u.ca2.x, u.time.y * u.ca2.y, 2);
    let rot = (n - 0.5) * PI * u.ca1.w;
    let cr = cos(rot); let sr = sin(rot);
    let ux = d.x / dd; let uy = d.y / dd;
    let w = vec2<f32>(ux * cr - uy * sr, ux * sr + uy * cr) * (u.ca1.x * mag * (0.4 + n));
    let a = u.ca2.w * min(1.0, mag);
    color = color + CA_RED * (glyphAt(p - w).a * a) + CA_CYAN * (glyphAt(p + w).a * a);
  }
  return vec4<f32>(color, 1.0);
}
`;

export async function initSwirlGPU(opts) {
  // getAtlasCanvas() is a GETTER, not the canvas itself: buildAtlas() creates a
  // brand-new atlas canvas on every resize, so a captured reference goes stale
  // (wrong tile size => garbled glyphs) or is undefined if the first resize bailed.
  const { canvas, getAtlasCanvas, codeData } = opts;
  if (!navigator.gpu) { console.info("[swirl-gpu] navigator.gpu unavailable — using Canvas 2D"); return null; }

  let device, ctx, format;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) { console.info("[swirl-gpu] no adapter — using Canvas 2D"); return null; }
    device = await adapter.requestDevice();
    ctx = canvas.getContext("webgpu");
    if (!ctx) { console.info("[swirl-gpu] no webgpu canvas context — using Canvas 2D"); return null; }
    format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: "opaque" });
  } catch (e) {
    console.warn("[swirl-gpu] init failed — using Canvas 2D:", e);
    return null;
  }

  device.addEventListener?.("uncapturederror", (ev) =>
    console.error("[swirl-gpu] device error:", ev.error?.message || ev.error));

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
  let gridCols = 0, gridRows = 0, atlasW = 0, atlasH = 0;
  const uniformArr = new Float32Array(12 * 4);

  function setGrid(g) {
    // Never build 0-size GPU resources (transient 0-size viewport / mid-zoom).
    const atlasCanvas = getAtlasCanvas();
    if (!canvas.width || !canvas.height || g.cols < 1 || g.rows < 1) return;
    if (!atlasCanvas || !atlasCanvas.width || !atlasCanvas.height) return;
    gridCols = g.cols; gridRows = g.rows;
    atlasW = atlasCanvas.width; atlasH = atlasCanvas.height;

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

  function render(u) {
    if (!cellBuf || !canvas.width || !canvas.height) return;
    const a = uniformArr;
    a[0]=u.cellWDev; a[1]=u.cellHDev; a[2]=u.Wdev;   a[3]=u.Hdev;
    a[4]=gridCols;   a[5]=gridRows;   a[6]=atlasW;            a[7]=atlasH;
    a[8]=u.tileWd;   a[9]=u.tileHd;   a[10]=u.atlasCols;      a[11]=u.levels;
    a[12]=u.t;       a[13]=u.tCA;     a[14]=u.asp;            a[15]=u.nLines;
    a[16]=u.swirl;   a[17]=u.swirlPower; a[18]=u.lensMag;     a[19]=u.lensRDev;
    a[20]=u.curXDev; a[21]=u.curYDev;   a[22]=u.heatInv2Dev;  a[23]=u.heatGain;
    a[24]=u.noiseScale; a[25]=u.noiseTime; a[26]=u.octaves;   a[27]=u.contrast;
    a[28]=u.ambientLow; a[29]=u.ambientHigh; a[30]=u.panCols; a[31]=u.panRows;
    a[32]=u.panNX;   a[33]=u.panNY;   a[34]=0; a[35]=0;
    a[36]=u.caStrengthDev; a[37]=u.caInv2Dev; a[38]=u.caRadiusDev; a[39]=u.caSmoke;
    a[40]=u.caSmokeFreqDev; a[41]=u.caDrift; a[42]=u.caEdge; a[43]=u.caAlpha;
    a[44]=u.caCullSqDev; a[45]=0; a[46]=0; a[47]=0;
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
        clearValue: { r: 27/255, g: 18/255, b: 34/255, a: 1 },
      }],
    });
    rp.setPipeline(renderPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();

    device.queue.submit([enc.finish()]);
  }

  console.info("[swirl-gpu] renderer active");
  return { setGrid, render, device };
}
