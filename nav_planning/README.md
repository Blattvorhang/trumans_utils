# Nav Planning — TRUMANS → G1 Humanoid Motion Pipeline

This folder contains the full pipeline for generating humanoid motion from a MuJoCo scene description: **scene XML → voxel grid → path planning → TRUMANS diffusion → SMPL motion → 3D visualization**.

Designed for Unitree G1 humanoid robot baseline testing. The pipeline supports both straight-line trajectories and obstacle-aware A* path planning over voxel occupancy grids.

## Pipeline Overview

```
MuJoCo XML
  ├── mujoco_scene_to_occ.py  →  voxel occupancy .npy + _grid.json
  ├── mujoco_scene_to_glb.py  →  simplified GLB mesh (for visualization)
  │
  └── trumans_infer.py        ←  start/goal + scene .npy
        ├── path_planner.py   ←  linear or A* path planning
        └── TRUMANS model     →  SMPL motion .npz
              │
              └── app_playback.py  →  3D web visualization
```

## Files

### `path_planner.py`
Global path planning on TRUMANS voxel occupancy grids. Two modes:

- **`linear`** — Straight-line interpolation between start and goal (fast, default).
- **`astar`** — A\* search with C-space dilation, path smoothing (line-of-sight pruning), and densification. Routes around obstacles at waist height (~0.6–0.8 m). Falls back to linear if start/goal are blocked or no path exists.

Key functions: `plan_path()`, `project_occupancy()`, `dilate_obstacles()`, `astar()`, `smooth_path()`, `densify_path()`.

### `mujoco_scene_to_occ.py`
Convert MuJoCo XML scenes to TRUMANS-format voxel occupancy grids.

- Collects box, cylinder, and plane geoms from XML (traverses `<body>` hierarchy).
- Converts MuJoCo z-up → TRUMANS y-up via `(-x, z, y)` (preserves handedness, det=+1).
- Applies scale factor: X/Z always scaled; Y only scaled if `--scale_height` is set.
- Outputs `.npy` (bool voxel grid) + `grid_meta/<name>_grid.json` sidecar.

**Usage:**
```bash
python nav_planning/mujoco_scene_to_occ.py \
    --xml assets/scene_asset/room_hanyi/room.xml \
    --scale 1.4 --margin 2.0 \
    --out_scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy
```

### `mujoco_scene_to_glb.py`
Convert MuJoCo XML scenes to simplified GLB meshes (glTF 2.0 binary) for 3D visualization.

- Produces box-only geometry with vertex colors from MuJoCo materials.
- Uses `KHR_materials_unlit` for flat rendering.
- Matches the same scale as `mujoco_scene_to_occ.py`.

**Usage:**
```bash
python nav_planning/mujoco_scene_to_glb.py \
    --xml assets/scene_asset/room_hanyi/room.xml \
    --scale 1.4 \
    --out static/room_hanyi_s1.4.glb
```

### `trumans_infer.py`
TRUMANS inference: start/goal points → SMPL motion sequence (.npz).

- Loads TRUMANS diffusion model (Transformer backbone) + JointsToSMPLX converter.
- Auto-detects scale from scene filename (e.g., `room_hanyi_s1.4.npy` → scale=1.4).
- Runs the full autoregressive generation pipeline with trajectory guidance.
- Converts output from TRUMANS y-up to AMASS/MuJoCo z-up.
- Saves two `.npz` files: main (shifted so start is at original position) + `_unshifted` reference.
- Output format: `poses(T,156)`, `trans(T,3)`, `betas(16,)`, `dmpls(T,8)`, `mocap_framerate`, plus custom `path` and `planner` fields.

**Usage:**
```bash
python nav_planning/trumans_infer.py \
    --start 0 0 --goal 2 0 \
    --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
    --output output/smpl_motion.npz \
    --planner astar
```

### `grid_search.py`
Batch grid search over multiple scale factors. For each scale:

1. Builds voxel occupancy (.npy) from MuJoCo XML via `mujoco_scene_to_occ`.
2. Runs TRUMANS inference via `trumans_infer`.
3. Saves results in per-scale output directories + a `results_summary.json`.

**Usage:**
```bash
python nav_planning/grid_search.py \
    --xml assets/scene_asset/room_hanyi/room.xml \
    --start 0 0 --goal 2 0 \
    --scales 1.0,1.1,1.2,1.3,1.4,1.5,1.6 \
    --output_dir output/
```

### `app_playback.py`
Flask web server for 3D visualization of pre-generated motions.

- Loads GLB scene + SMPL .npz and animates the body as a point cloud.
- Computes SMPL-X forward kinematics on-the-fly.
- Displays planned path waypoints and start/goal markers.
- Auto-scans directories for available scenes and motions.

**Usage:**
```bash
python nav_planning/app_playback.py --port 5001
```

## `grid_meta/` — Scene Voxel Grid Metadata

Each `.json` file in `grid_meta/` is a **sidecar** for a corresponding `.npy` voxel occupancy grid, storing its physical bounds and generation parameters. Downstream tools read these files to correctly interpret the raw 3D arrays.

### Naming Convention

```
grid_meta/{scene_name}_s{scale}_grid.json
```

Examples: `room_hanyi_s1.0_grid.json`, `room_zixuan_s1.4_grid.json`

### Scene Artifact Triplet

Every `grid_meta/*.json` belongs to a matching set of three files:

```
grid_meta/room_hanyi_s1.4_grid.json               ← bounds + generation params
Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy   ← voxel occupancy grid (3D bool)
static/room_hanyi_s1.4.glb                         ← GLB scene mesh (visualization)
```

Tools discover the pairing by basename at runtime — all three share the `{scene_name}_s{scale}` stem.

> **Why `grid_meta/` is separate from `Data_blocks_motion_all/Scene/`:** The TRUMANS `TrumansDataset` scans all `.npy` files under the scene directory at startup. Placing JSON sidecars there would cause scan errors (non-`.npy` files). `grid_meta/` is kept outside to avoid this.

### JSON Fields

| Field | Type | Description |
|-------|------|-------------|
| `scene_grid` | `float[9]` | `[x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]` — TRUMANS y-up bounds + grid shape |
| `resolution` | `float` | Voxel edge length in metres, always `0.02` (TRUMANS standard) |
| `scale` | `float` | G1-to-SMPL scale applied during voxelisation (e.g., `1.4` = 1.4× enlarged) |
| `scale_height` | `bool` | Whether vertical (Y) was also scaled; `false` means furniture heights stay at original TRUMANS values for interaction recognition |
| `margin` | `float` | Extra free-space padding in metres added beyond geometry bounds (default `2.0`) |

### Produced by

[`mujoco_scene_to_occ.py`](mujoco_scene_to_occ.py) writes a sidecar for every `.npy` it generates.

### Consumed by

| Consumer | How it uses grid_meta |
|----------|----------------------|
| [`trumans_infer.py`](trumans_infer.py) | `load_scene_meta()` reads `scene_grid` and overrides the model's default grid bounds — without this, scaled scenes would infer with wrong coordinates |
| [`path_planner.py`](path_planner.py) | A\* planning uses `scene_grid` + `resolution` for world↔grid coordinate conversion and 2D projection |
| [`app_playback.py`](app_playback.py) | Discovers scale factor from grid_meta to correctly align GLB + motion when loading via the API |

### Example

```json
{
  "scene_grid": [-5.74, -2.94, -5.74, 5.74, 5.6, 5.74, 574, 427, 574],
  "resolution": 0.02,
  "scale": 1.4,
  "scale_height": false,
  "margin": 2.0
}
```

A `room_hanyi` scene at 1.4× scale. Grid spans ~11.5×8.5×11.5 m (574×427×574 cells at 2 cm/voxel). Height was not scaled — furniture stays at original TRUMANS interaction heights.

## Coordinate Systems

| System | Up Axis | Convention |
|--------|---------|------------|
| MuJoCo | z-up | (x=right, y=forward, z=up) |
| TRUMANS / glTF | y-up | (x=left, y=up, z=forward) |
| Conversion | — | MuJoCo→TRUMANS: `(-x, z, y)` — right-handed (det=+1). Negating x is necessary: a naive `(x, z, y)` swap has det=-1, which mirrors the scene. |

## Scale Factor

The G1 robot (~1.27 m) is shorter than SMPL (~1.8 m), so the scene is enlarged by a scale factor during generation. Horizontal dimensions (X, Z) always scale; vertical (Y) only scales with `--scale_height`. Post-generation, the motion is rigidly translated so the start point returns to its original (unscaled) position.

## Key Limitation

TRUMANS is a **trajectory-following motion generator**, not a navigation planner. It generates natural body motions along given waypoints with local scene awareness (side-stepping, ducking), but does NOT handle global obstacle avoidance. Use A\* mode in `path_planner.py` for navigation around obstacles.

## Dependencies

- Python 3.x, PyTorch, NumPy, SciPy
- TRUMANS model checkpoints + SMPL-X models (see [main README](../README.md))
- Flask (for `app_playback.py`)
- PyTorch3D, Kaolin, smplx (for inference)
