# Nav Planning — TRUMANS → G1 Humanoid Motion Pipeline

This folder contains the full pipeline for generating humanoid motion from a MuJoCo scene description: **scene XML → voxel grid → path planning → TRUMANS diffusion → SMPL motion → 3D visualization**.

Designed for Unitree G1 humanoid robot baseline testing. The pipeline supports both straight-line trajectories and obstacle-aware A* path planning over voxel occupancy grids, plus **closed-loop replanning** with G1 state feedback.

## Pipeline Overview

```
MuJoCo XML
  ├── mujoco_scene_to_occ.py  →  voxel occupancy .npy + _grid.json
  ├── mujoco_scene_to_glb.py  →  simplified GLB mesh (for visualization)
  │
  └── trumans_infer.py        ←  start/goal + scene .npy
        ├── path_planner.py   ←  linear or A* path planning
        ├── replan_state.py   ←  autoregressive state save/load/adapt
        └── TRUMANS model     →  SMPL motion .npz + _state.npz
              │
              └── app_playback.py  →  3D web visualization

Closed-loop replanning:
  trumans_infer.py (round N) → motion.npz + _state.npz
       → G1 controller tracks a short horizon
       → G1 real (x, y, heading) fed back
  trumans_infer.py (round N+1) ← --resume-from _state.npz --robot-heading dx dy
       → new plan continues seamlessly from G1's actual pose
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
- Supports **short-horizon generation** (`--max-steps`) and **closed-loop replanning** (`--resume-from`, `--robot-heading`).

**Usage (cold start):**
```bash
python nav_planning/trumans_infer.py \
    --start 0 0 --goal 2 0 \
    --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
    --output output/smpl_motion.npz \
    --planner astar
# → smpl_motion.npz + smpl_motion_state.npz
```

**Usage (replan — resume from previous state):**
```bash
python nav_planning/trumans_infer.py \
    --start <g1_real_x> <g1_real_y> --goal 5 0 \
    --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
    --output output/smpl_motion_r2.npz \
    --planner astar \
    --max-steps 5 \
    --resume-from output/smpl_motion_r1_state.npz \
    --robot-heading <g1_dx> <g1_dy>
# → smpl_motion_r2.npz + smpl_motion_r2_state.npz
```

**Key CLI flags for replanning:**

| Flag | Description |
|------|-------------|
| `--resume-from PATH` | Load `_state.npz` from a previous run for hot-start generation |
| `--robot-heading DX DY` | G1's actual heading vector in z-up ground plane (used with `--resume-from`) |
| `--max-steps N` | Limit diffusion steps. Short horizons (5-8 steps ≈ 1.5-3s) keep the state close to the tracking cutoff for true head-to-tail continuity |

### `replan_state.py`
Autoregressive state management for closed-loop replanning.

- **`GenerationState`** — dataclass holding `mat` (world transform) + `fixed_points` (last 2 frames of 24-joint positions in world space). Both in TRUMANS y-up.
- **`save_state()` / `load_state()`** — serialise to/from `.npz`.
- **`adapt_state(state, new_start_zup, heading_zup)`** — the core replanning operation. Applies a **rigid transform T = [R_yaw \| t_pos]** to the history buffer so that:

  - The pelvis root aligns with G1's actual ground-plane position.
  - The body faces G1's actual heading.
  - All 24 joints across all history frames receive the **same** transform — body pose (relative joint positions) is perfectly preserved.

  **Why rigid, not translation-only:** The TRUMANS model operates in **egocentric** coordinates — it expects the root at the origin, with history frames showing body pose relative to that root. If the history buffer were only translated but not rotated, the body orientation would mismatch the new coordinate frame's yaw, producing out-of-distribution input to the model. Applying `T` to both `mat` and `fixed_points` ensures the egocentric view is perfectly continuous: root at origin, body pose unchanged, no jump.

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

## Closed-Loop Replanning

TRUMANS generates motion by feeding the last `fixed_frame=2` frames of each diffusion episode as fixed context to the next — this is the **autoregressive state**. The replanning system exposes this state explicitly, enabling independent script invocations to chain together seamlessly.

### Workflow

```
Round 1 (cold start):
  trumans_infer.py --start 0 0 --goal 5 0 --max-steps 5
    → path_planner → trajectory → get_guidance → sample_step(fixed_points_init=None)
    → motion_r1.npz + motion_r1_state.npz

  G1 controller tracks motion_r1.npz for ~1.5s
    → reads G1 real position (x, y) and heading vector (dx, dy) from simulation

Round 2 (replan):
  trumans_infer.py --start <g1_x> <g1_y> --goal 5 0 --max-steps 5
    --resume-from motion_r1_state.npz --robot-heading <dx> <dy>
    → path_planner (new trajectory from G1 to goal)
    → adapt_state(prev_state, g1_pos, g1_heading)  ← rigid T=[R|t]
    → sample_step(fixed_points_init=adapted.fixed_points)
    → motion_r2.npz + motion_r2_state.npz

  ... repeat until goal reached ...
```

### Design Rationale: Egocentric Coordinate Continuity

The TRUMANS diffusion model was trained on **egocentric** human motion — it expects the root (pelvis) at the coordinate origin, with history frames representing body pose relative to that root.

When G1's real position and heading differ from the planned trajectory (which they always will), simply using the old history buffer as-is would create an egocentric jump: the root position in the history frames would not match the new start position, producing out-of-distribution (OOD) input to the model.

**The fix**: apply the **same rigid transform `T = [R_yaw | t_pos]`** to both:
- The generation coordinate frame (`mat`) — pulled to G1's position + heading
- The entire history buffer (`fixed_points`) — all 24 joints × 2 frames shifted+rotated by `T`

This way, in the egocentric view of `adapted.mat`, the root is at the origin and the body pose is perfectly continuous. The model sees exactly what it was trained on: smooth, continuous egocentric motion.

**Key insight**: The SONIC controller only consumes body pose (relative joint rotations), not root position. Root position jumps in the SMPL output do not directly cause controller jitter. The jitter comes from the **planner** producing OOD motion when it receives discontinuous egocentric input.

### State File Format (`_state.npz`)

| Key | Shape | Description |
|-----|-------|-------------|
| `mat` | `(1, 4, 4)` float32 | SE(2)-in-SE(3) world transform: yaw rotation + (x, 0, z) translation, y-up |
| `fixed_points` | `(1, 2, 72)` float32 | Last 2 frames of 24-joint 3D positions, y-up world space |

### Related Documents

- [plan/README.md](plan/README.md) — detailed implementation plan for the replanning feature
- [replan_state.py](replan_state.py) — `GenerationState` dataclass, serialization, and `adapt_state()`

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
