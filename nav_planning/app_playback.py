"""
Flask playback app for pre-generated TRUMANS SMPL motions.

Loads a scene GLB + a motion .npz and animates the SMPL skeleton in 3D.

Usage:
    python app_playback.py --port 5001
"""

import argparse
import glob
import json
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__,
            template_folder=os.path.join(_PROJ, "templates"),
            static_folder=os.path.join(_PROJ, "static"))

# Global caches
_smpl_model = None
_vertex_cache = {}  # npz_path → [T, N, 3] vertices
_scan_dirs = ["output", "artifacts"]
_glb_dirs = ["static", "assets", "."]


# ---------------------------------------------------------------------------
# SMPL vertex computation
# ---------------------------------------------------------------------------


def _get_smpl_model(device="cpu"):
    """Lazy-load SMPL-X model for FK computation."""
    global _smpl_model
    if _smpl_model is None:
        import smplx

        _smpl_model = smplx.create(
            "./smpl_models",
            model_type="smplx",
            gender="male",
            ext="npz",
            num_betas=10,
            use_pca=False,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_transl=True,
            batch_size=1,
        ).to(device)
        _smpl_model.eval()
    return _smpl_model


def _zup_to_yup_points(pts):
    """Convert 3D points from z-up to y-up: (-x, z, y)."""
    return np.stack([-pts[..., 0], pts[..., 2], pts[..., 1]], axis=-1)


def compute_vertices(npz_path, subsample=10, device="cpu"):
    """Compute SMPL-X vertices from .npz (z-up on disk → y-up FK → y-up output)."""
    if npz_path in _vertex_cache:
        return _vertex_cache[npz_path]

    data = np.load(npz_path)
    poses = data["poses"].astype(np.float32)  # (T, 156) — z-up
    trans_zup = data["trans"].astype(np.float32)  # (T, 3) — z-up

    # ---- convert from z-up back to y-up for SMPL-X FK ----
    from scipy.spatial.transform import Rotation as R

    R_zup_to_yup = R.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])  # det=+1, self-inverse

    # global_orient: axis-angle in z-up world frame → y-up world frame
    global_orient_zup = poses[:, :3]
    global_orient_yup = (R_zup_to_yup * R.from_rotvec(global_orient_zup)).as_rotvec().astype(np.float32)

    # transl: z-up point → y-up point
    transl_yup = _zup_to_yup_points(trans_zup)

    T = poses.shape[0]
    model = _get_smpl_model(device)

    # Re-batch SMPL model for full sequence
    import smplx

    model_seq = smplx.create(
        "./smpl_models",
        model_type="smplx",
        gender="male",
        ext="npz",
        num_betas=10,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=T,
    ).to(device)
    model_seq.eval()

    with torch.no_grad():
        global_orient = torch.from_numpy(global_orient_yup).to(device)
        body_pose = torch.from_numpy(poses[:, 3:66]).to(device)  # local frame — unchanged
        transl = torch.from_numpy(transl_yup).to(device)
        # Other params: zeros
        zeros_T_3 = torch.zeros(T, 3, device=device)
        zeros_T_6 = torch.zeros(T, 6, device=device)
        zeros_T_10 = torch.zeros(T, 10, device=device)
        zeros_T_45 = torch.zeros(T, 45, device=device)

        output = model_seq(
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose,
            betas=zeros_T_10,
            left_hand_pose=zeros_T_45,
            right_hand_pose=zeros_T_45,
            jaw_pose=zeros_T_3,
            leye_pose=zeros_T_3,
            reye_pose=zeros_T_3,
            expression=zeros_T_10,
            return_verts=True,
        )
        verts_yup = output.vertices[:, ::subsample].cpu().numpy()  # (T, N, 3) — y-up

    _vertex_cache[npz_path] = verts_yup
    return verts_yup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("playback.html")


@app.route("/api/scenes")
def list_scenes():
    """List available GLB files under scan dirs."""
    glb_files = []
    for d in _glb_dirs:
        if not os.path.isdir(d):
            continue
        glb_files += glob.glob(os.path.join(d, "**/*.glb"), recursive=True)
    scenes = []
    seen = set()
    for f in sorted(glb_files):
        name = os.path.basename(f).replace(".glb", "")
        if name in seen:
            continue
        seen.add(name)
        scenes.append({"name": name, "path": f})
    return jsonify(scenes)


@app.route("/api/motions")
def list_motions():
    """List available .npz files under scan dirs."""
    motions = []
    seen_paths = set()
    for d in _scan_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "**/*.npz"), recursive=True)):
            if "smpl_models" in f:
                continue
            real = os.path.realpath(f)
            if real in seen_paths:
                continue
            seen_paths.add(real)
            motions.append({"name": f, "path": f})
    return jsonify(motions)


@app.route("/api/load_motion", methods=["POST"])
def load_motion():
    """Compute and return SMPL vertices for a given .npz path."""
    data = request.json
    npz_path = data.get("npz_path", "")
    if not npz_path or not os.path.exists(npz_path):
        return jsonify({"error": f"File not found: {npz_path}"}), 404

    try:
        verts = compute_vertices(npz_path)
        result = verts.tolist()

        # Extract path waypoints if stored in the .npz (convert z-up → y-up)
        path_waypoints = None
        planner = "unknown"
        try:
            raw = np.load(npz_path, allow_pickle=True)
            if "path" in raw:
                path_zup = raw["path"]  # (N, 3) in z-up
                path_waypoints = _zup_to_yup_points(path_zup).tolist()
            if "planner" in raw:
                planner = str(raw["planner"])
        except Exception:
            pass

        return jsonify(
            {
                "frames": len(result),
                "vertices_per_frame": len(result[0]),
                "data": result,
                "path": path_waypoints,
                "planner": planner,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/path", methods=["POST"])
def get_path():
    """Compute and return path waypoints for a given scene and start/goal.

    POST body JSON:
    {
        "scene_path": "Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy",
        "start_x": 0.0, "start_z": 0.0,
        "goal_x": 2.0, "goal_z": 0.0,
        "planner": "linear",
        "clearance": 0.25,
        "height_min": 0.6, "height_max": 0.8
    }
    """
    import traceback

    from path_planner import plan_path

    data = request.json
    scene_path = data.get("scene_path", "")
    if not scene_path or not os.path.exists(scene_path):
        return jsonify({"error": f"Scene not found: {scene_path}"}), 404

    start_2d = (data.get("start_x", 0.0), data.get("start_z", 0.0))
    goal_2d = (data.get("goal_x", 2.0), data.get("goal_z", 0.0))
    planner = data.get("planner", "linear")
    clearance = data.get("clearance", 0.25)
    height_min = data.get("height_min", 0.6)
    height_max = data.get("height_max", 0.8)

    try:
        if planner == "astar":
            # Load grid metadata sidecar
            scene_basename = os.path.basename(scene_path).replace(".npy", "")
            meta_path = os.path.join("grid_meta", f"{scene_basename}_grid.json")
            grid_meta = None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    grid_meta = json.load(f)

            if grid_meta is None:
                return jsonify({"error": f"No grid_meta found for {scene_path}"}), 404

            occu_3d = np.load(scene_path).astype(bool)
            waypoints = plan_path(
                start_2d, goal_2d,
                occu_3d=occu_3d,
                grid_meta=grid_meta,
                mode="astar",
                clearance=clearance,
                height_range=(height_min, height_max),
                ground_y=0.1,
            )
        else:
            waypoints = plan_path(start_2d, goal_2d, mode="linear", ground_y=0.1)

        return jsonify({
            "waypoints": waypoints.tolist(),
            "planner": planner,
            "num_waypoints": len(waypoints),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/load_glb_meta")
def load_glb_meta():
    """Return available GLB paths for the frontend to load directly."""
    scenes = []
    seen = set()
    for d in _glb_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "**/*.glb"), recursive=True)):
            name = os.path.basename(f).replace(".glb", "")
            if name in seen:
                continue
            seen.add(name)
            scenes.append({"name": name, "path": f})
    return jsonify(scenes)


@app.route("/<path:filename>")
def serve_file(filename):
    """Serve files from the working directory for GLTFLoader."""
    if os.path.isfile(filename):
        return send_file(filename)
    return jsonify({"error": f"Not found: {filename}"}), 404


# ---------------------------------------------------------------------------
# Playback HTML template
# ---------------------------------------------------------------------------

PLAYBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TRUMANS Motion Playback</title>
<style>
  body { margin: 0; overflow: hidden; font-family: monospace; }
  #panel {
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: rgba(0,0,0,0.8); color: #fff; padding: 12px;
    border-radius: 6px; max-width: 320px;
  }
  #panel select, #panel button, #panel input {
    display: block; width: 100%; margin: 4px 0; padding: 6px;
    font-size: 13px;
  }
  #info { position: absolute; bottom: 10px; left: 10px; color: #fff;
          background: rgba(0,0,0,0.6); padding: 6px 10px; border-radius: 4px; }
</style>
</head>
<body>
<div id="panel">
  <label>Scene GLB:</label>
  <select id="sceneSelect"></select>
  <label>Motion .npz:</label>
  <select id="motionSelect"></select>
  <button id="loadBtn">Load & Play</button>
  <hr>
  <label>Frame:</label>
  <input type="range" id="frameSlider" min="0" max="100" value="0">
  <button id="playBtn">Play</button>
  <button id="pauseBtn">Pause</button>
  <span id="frameLabel" style="color:#aaa">0 / 0</span>
</div>
<div id="info">TRUMANS Motion Playback</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222);
scene.add(new THREE.AmbientLight(0xffffff, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);
const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x3f3f3f, 0.6);
scene.add(hemiLight);
scene.add(new THREE.GridHelper(10, 20));

const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 100);
camera.position.set(5, 8, 5);
camera.lookAt(0, 0.5, 0);

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

// Point cloud for SMPL body
let pointCloud = null;
let motionData = null;
let currentFrame = 0;
let playing = false;
let gltfScene = null;
const gltfLoader = new GLTFLoader();

const dummyGeo = new THREE.BufferGeometry();
dummyGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
const material = new THREE.PointsMaterial({size: 0.03, color: 0x4080ff});
pointCloud = new THREE.Points(dummyGeo, material);
scene.add(pointCloud);

// ---- UI bindings ----
const sceneSelect = document.getElementById('sceneSelect');
const motionSelect = document.getElementById('motionSelect');
const frameSlider = document.getElementById('frameSlider');
const frameLabel = document.getElementById('frameLabel');

async function loadScenes() {
  const res = await fetch('/api/scenes');
  const scenes = await res.json();
  scenes.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.path;
    opt.text = s.name;
    sceneSelect.appendChild(opt);
  });
}

async function loadMotions() {
  const res = await fetch('/api/motions');
  const motions = await res.json();
  motions.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.text = m.name;
    motionSelect.appendChild(opt);
  });
}

document.getElementById('loadBtn').addEventListener('click', async () => {
  // Remove old GLB scene (with full disposal)
  if (gltfScene) {
    scene.remove(gltfScene);
    gltfScene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
    gltfScene = null;
  }

  // Load scene GLB
  const glbPath = sceneSelect.value;
  if (glbPath) {
    try {
      const gltf = await gltfLoader.loadAsync(glbPath);
      gltfScene = gltf.scene;
      scene.add(gltfScene);
    } catch(e) {
      console.warn('GLB load failed:', e);
    }
  }

  // Load motion
  const npzPath = motionSelect.value;
  const info = document.getElementById('info');
  info.textContent = 'Loading motion...';
  const res = await fetch('/api/load_motion', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({npz_path: npzPath})
  });
  const result = await res.json();
  if (result.error) {
    info.textContent = 'Error: ' + result.error;
    return;
  }
  motionData = result.data;
  frameSlider.max = motionData.length - 1;
  frameSlider.value = 0;
  currentFrame = 0;

  // Auto-render path if included
  if (result.path && result.path.length >= 2) {
    updatePathLine(result.path);
    const p0 = result.path[0], pN = result.path[result.path.length - 1];
    updateMarkers(p0[0], p0[2], pN[0], pN[2]);
    info.textContent = `Loaded ${motionData.length} frames, ${result.vertices_per_frame} verts (${result.planner})`;
  } else {
    info.textContent = `Loaded ${motionData.length} frames, ${result.vertices_per_frame} verts`;
  }
  updateFrame();
});

document.getElementById('playBtn').addEventListener('click', () => { playing = true; });
document.getElementById('pauseBtn').addEventListener('click', () => { playing = false; });
frameSlider.addEventListener('input', () => {
  currentFrame = parseInt(frameSlider.value);
  updateFrame();
});

function updateFrame() {
  if (!motionData) return;
  const frame = motionData[currentFrame];
  if (!pointCloud.geometry || pointCloud.geometry.attributes.position.count !== frame.length) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(frame.length * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (pointCloud.geometry) pointCloud.geometry.dispose();
    pointCloud.geometry = geometry;
  }
  const pos = pointCloud.geometry.attributes.position.array;
  for (let i = 0, j = 0; i < frame.length; i++, j += 3) {
    pos[j] = frame[i][0];
    pos[j+1] = frame[i][1];
    pos[j+2] = frame[i][2];
  }
  pointCloud.geometry.attributes.position.needsUpdate = true;
  frameSlider.value = currentFrame;
  frameLabel.textContent = `${currentFrame + 1} / ${motionData.length}`;
}

// ---- Path line rendering ----
let pathLine = null;
let startMarker = null;
let goalMarker = null;

function updatePathLine(waypoints) {
  if (pathLine) {
    scene.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
    pathLine = null;
  }
  if (!waypoints || waypoints.length < 2) return;

  const positions = new Float32Array(waypoints.length * 3);
  for (let i = 0; i < waypoints.length; i++) {
    positions[i * 3] = waypoints[i][0];
    positions[i * 3 + 1] = waypoints[i][1];
    positions[i * 3 + 2] = waypoints[i][2];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: 0x0000ff });
  pathLine = new THREE.Line(geometry, material);
  scene.add(pathLine);
}

function updateMarkers(sx, sz, gx, gz) {
  if (startMarker) scene.remove(startMarker);
  if (goalMarker) scene.remove(goalMarker);

  const sphereGeo = new THREE.SphereGeometry(0.08, 16, 16);
  const startMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const goalMat = new THREE.MeshBasicMaterial({ color: 0xff0000 });

  startMarker = new THREE.Mesh(sphereGeo, startMat);
  startMarker.position.set(sx, 0.1, sz);
  scene.add(startMarker);

  goalMarker = new THREE.Mesh(sphereGeo, goalMat);
  goalMarker.position.set(gx, 0.1, gz);
  scene.add(goalMarker);
}

let lastFrameTime = 0;
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;

function animate(timestamp) {
  requestAnimationFrame(animate);
  if (playing && motionData && timestamp - lastFrameTime >= FRAME_INTERVAL) {
    lastFrameTime = timestamp;
    currentFrame = (currentFrame + 1) % motionData.length;
    updateFrame();
  }
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

loadScenes();
loadMotions();
animate();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask playback app for TRUMANS SMPL motions"
    )
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--scan",
        nargs="*",
        default=["output", "artifacts", "."],
        help="Directories to scan for .npz files",
    )
    args = parser.parse_args()

    _scan_dirs = args.scan

    # Switch to project root — all paths (templates, GLB dirs, NPZ dirs) are relative to it.
    os.chdir(_PROJ)

    # Write the HTML template
    os.makedirs("templates", exist_ok=True)
    with open("templates/playback.html", "w") as f:
        f.write(PLAYBACK_HTML)

    print(f"Playback server: http://127.0.0.1:{args.port}")
    print(f"Scanning for .npz in: {_scan_dirs}")
    app.run(host=args.host, port=args.port, debug=False)
