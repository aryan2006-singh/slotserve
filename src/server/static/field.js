// 3D slot field shared by the home page and the chat page.
// A grid of instanced blocks rolls like a slow wave; blocks light up in the
// accent colour for live requests ("slotserve:stats" events), and the wave
// speeds up with tokens/s. Decorative only: returns null if WebGL is missing.
import * as THREE from "./vendor/three.module.min.js";

export function mountField(canvas, main, opts = {}) {
  const gsap = window.gsap;
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
  } catch (e) {
    canvas.remove();
    return null;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 200);
  const COLS = 30, ROWS = 18, GAP = 1.08;
  const COUNT = COLS * ROWS;

  const geo = new THREE.BoxGeometry(0.86, 1, 0.86);
  geo.translate(0, 0.5, 0); // grow from the floor
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.0 });
  const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-8, 18, 10);
  scene.add(hemi, sun);

  const cells = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = (c - (COLS - 1) / 2) * GAP;
      const z = (r - (ROWS - 1) / 2) * GAP;
      cells.push({ x, z, d: Math.hypot(x, z), lit: 0, target: 0, seed: Math.random() * 6.28 });
    }
  }
  const maxD = Math.max(...cells.map((c) => c.d));

  // State that GSAP tweens: intro rise, energy (tokens/s), dim when chatting.
  const state = { rise: REDUCED ? 1 : 0, energy: 0.15, opacity: 0, mx: 0, my: 0, pmx: 0, pmy: 0 };

  const base = new THREE.Color(), hi = new THREE.Color(), tmp = new THREE.Color(), fogC = new THREE.Color();
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    base.set(cs.getPropertyValue("--box").trim() || "#e6dfcd");
    hi.set(cs.getPropertyValue("--box-hi").trim() || "#f2d64b");
    fogC.set(cs.getPropertyValue("--bg").trim() || "#f3efe4");
    scene.fog = new THREE.Fog(fogC, 22, 46);
    const dark = cs.colorScheme === "dark" || cs.getPropertyValue("color-scheme").trim() === "dark";
    hemi.intensity = dark ? 0.55 : 1.1;
    sun.intensity = dark ? 1.1 : 1.6;
  }
  readTheme();
  new MutationObserver(readTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readTheme);

  const view = { lookY: opts.lookY ?? 7.5, lookYNarrow: opts.lookYNarrow ?? 9 };
  function resize() {
    const w = main.clientWidth, h = main.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Keep the field in the lower part of the view so the headline stays readable.
    const narrow = w < 640;
    camera.position.set(0, narrow ? 21 : 15, narrow ? 30 : 24);
    camera.lookAt(0, narrow ? view.lookYNarrow : view.lookY, 0);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(main);
  resize();

  // Light up one random cell per running request (several per slot so it reads).
  function setLive(running, waiting) {
    const wantLit = Math.min(COUNT, running * 7 + waiting * 2);
    const litNow = cells.filter((c) => c.target > 0);
    if (litNow.length > wantLit) {
      litNow.sort(() => Math.random() - 0.5).slice(0, litNow.length - wantLit).forEach((c) => (c.target = 0));
    } else {
      const dark = cells.filter((c) => c.target === 0).sort(() => Math.random() - 0.5);
      dark.slice(0, wantLit - litNow.length).forEach((c) => (c.target = 1));
    }
  }

  window.addEventListener("slotserve:stats", (e) => {
    const s = e.detail || {};
    setLive(s.running || 0, s.waiting || 0);
    const energy = 0.15 + Math.min(1, (s.tokens_per_second || 0) / 120) * 0.85;
    if (gsap) gsap.to(state, { energy, duration: 1.2, ease: "power2.out" });
    else state.energy = energy;
  });
  const dimTo = (v) => gsap ? gsap.to(state, { opacity: v, duration: 0.8, ease: "power2.out" }) : (state.opacity = v);
  window.addEventListener("slotserve:chat", () => dimTo(0.28));
  window.addEventListener("slotserve:reset", () => dimTo(1));
  window.addEventListener("pointermove", (e) => {
    state.mx = (e.clientX / window.innerWidth) * 2 - 1;
    state.my = (e.clientY / window.innerHeight) * 2 - 1;
  });

  // Intro: fade in, then blocks rise outward from the centre.
  if (gsap && !REDUCED) {
    gsap.to(state, { opacity: document.body.classList.contains("chatting") ? 0.28 : 1, duration: 1.2, ease: "power2.out" });
    gsap.to(state, { rise: 1, duration: 2.2, ease: "power3.out", delay: 0.15 });
  } else {
    state.opacity = 1;
  }

  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), sc = new THREE.Vector3();
  const clock = new THREE.Clock();
  let t = 0;
  function frame() {
    const dt = Math.min(clock.getDelta(), 0.05);
    t += dt * (0.35 + state.energy * 1.6);
    state.pmx += (state.mx - state.pmx) * 0.04;
    state.pmy += (state.my - state.pmy) * 0.04;

    for (let i = 0; i < COUNT; i++) {
      const c = cells[i];
      c.lit += (c.target - c.lit) * 0.08;
      // Rise ripples outward from the centre during the intro.
      const local = THREE.MathUtils.clamp(state.rise * 1.6 - (c.d / maxD) * 0.6, 0, 1);
      const wave = Math.sin(c.x * 0.32 + t) * Math.cos(c.z * 0.38 - t * 0.8) * 0.5 + 0.5;
      const h = (0.25 + wave * (0.6 + state.energy * 1.8) + c.lit * (2.2 + Math.sin(t * 3 + c.seed) * 0.4)) * local + 0.02;
      p.set(c.x, 0, c.z);
      sc.set(1, h, 1);
      m.compose(p, q, sc);
      mesh.setMatrixAt(i, m);
      tmp.copy(base).lerp(hi, c.lit);
      mesh.setColorAt(i, tmp);
    }
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;

    mesh.rotation.y = -0.18 + state.pmx * 0.12 + Math.sin(t * 0.05) * 0.04;
    mesh.rotation.x = state.pmy * 0.04;
    canvas.style.opacity = state.opacity;
    renderer.render(scene, camera);
  }

  // On pages without a live engine, light a few random blocks now and then.
  if (opts.autoPulse) {
    const pulse = () => { setLive(1 + Math.floor(Math.random() * 4), Math.floor(Math.random() * 3)); state.energy = 0.3 + Math.random() * 0.5; };
    pulse();
    setInterval(pulse, 2600);
  }

  if (REDUCED) {
    frame();
    window.addEventListener("slotserve:stats", () => requestAnimationFrame(frame));
  } else {
    // Only render while the field is on screen and visible: saves battery and
    // keeps scrolling smooth on long pages.
    let onScreen = true;
    const sync = () => renderer.setAnimationLoop(onScreen && state.opacity > 0.01 ? frame : null);
    new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; sync(); }).observe(canvas);
    state.wake = () => sync();
    renderer.setAnimationLoop(frame);
  }
  return state;
}
