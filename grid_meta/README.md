# grid_meta — Scene Voxel Grid Metadata

Each `.json` file in this directory is a **sidecar** for a corresponding `.npy` voxel occupancy grid. It stores the physical bounds and generation parameters of the voxel grid, so downstream tools know how to interpret the raw 3D array.

## File Naming Convention

```
grid_meta/{scene_name}_s{scale}_grid.json
```

| Segment | Meaning |
|---------|---------|
| `{scene_name}` | Scene identifier (e.g. `room_hanyi`, `room_zixuan`) |
| `s{scale}` | G1-to-SMPL scale factor (e.g. `s1.0`, `s1.4`) |
| `_grid.json` | Suffix marking this as grid metadata |

## Relationship to Other Files

Each grid_meta JSON belongs to a **triplet** of scene artifacts:

```
grid_meta/room_hanyi_s1.4_grid.json               ← bounds + generation params (this file)
Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy   ← voxel occupancy grid (3D bool array)
static/room_hanyi_s1.4.glb                         ← GLB scene mesh (for visualization)
```

The GLB and its matching grid_meta share the same basename — this is how `app_playback.py` and `trumans_infer.py` auto-discover the pairing at runtime.

> **Why `grid_meta/` is separate from `Data_blocks_motion_all/Scene/`:** The TRUMANS `TrumansDataset` scans all `.npy` files under the scene directory at startup. Placing JSON sidecars there would cause scan errors. `grid_meta/` is kept outside to avoid this.

## JSON Fields

### `scene_grid`
9-element array: `[x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]`

This matches the `TrumansDataset.scene_grid_np` format used internally by the TRUMANS model.

| Index | Name | Description |
|-------|------|-------------|
| 0 | `x_min` | Minimum X bound (m) |
| 1 | `y_min` | Minimum Y / height bound (m) |
| 2 | `z_min` | Minimum Z bound (m) |
| 3 | `x_max` | Maximum X bound (m) |
| 4 | `y_max` | Maximum Y / height bound (m) |
| 5 | `z_max` | Maximum Z bound (m) |
| 6 | `nx` | Grid cells along X axis |
| 7 | `ny` | Grid cells along Y axis |
| 8 | `nz` | Grid cells along Z axis |

Resolution can be verified: `nx × resolution ≈ (x_max - x_min)`.

**Coordinate system is y-up** (TRUMANS convention).

### `resolution`
Voxel edge length in meters. Always `0.02` (TRUMANS standard).

### `scale`
G1-to-SMPL scaling factor applied during voxelisation:

| `scale` | Meaning |
|---------|---------|
| `1.0` | Native SMPL scale — scene geometry unchanged |
| `1.4` | Enlarged 1.4× — compensates for G1 (~1.27 m) vs SMPL (~1.8 m) height ratio |

### `scale_height`
Boolean. If `true`, vertical (Y) dimensions were also scaled — tables, beds, etc. become taller. Default is `false` (keep original heights so TRUMANS recognizes interaction affordances like sitting/leaning).

### `margin`
Extra free-space padding (meters) added beyond the scene's geometric bounding box. Default `2.0`.

## How These Files Are Used

### Produced by
[`nav_planning/mujoco_scene_to_occ.py`](../nav_planning/mujoco_scene_to_occ.py) writes grid_meta alongside every `.npy` it generates (lines 333–340).

### Consumed by

| Consumer | What it does with grid_meta |
|----------|---------------------------|
| [`trumans_infer.py`](../nav_planning/trumans_infer.py) | `load_scene_meta()` reads `scene_grid` and overrides the model's default scene bounds — without this, scaled scenes would use wrong coordinate ranges during inference |
| [`path_planner.py`](../nav_planning/path_planner.py) | A* path planning uses `scene_grid` + `resolution` for voxel-based collision checking |
| [`app_playback.py`](../nav_planning/app_playback.py) | Auto-discovers scale factor from grid_meta to correctly display GLB + motion |

## Example

```json
{
  "scene_grid": [-5.74, -2.94, -5.74, 5.74, 5.6, 5.74, 574, 427, 574],
  "resolution": 0.02,
  "scale": 1.4,
  "scale_height": false,
  "margin": 2.0
}
```

This describes a `room_hanyi` scene enlarged 1.4× for G1. The voxel grid spans ~11.5×8.5×11.5 m (574×427×574 cells at 2 cm/voxel). Height was NOT scaled, so furniture heights remain at their original TRUMANS-trained values.
