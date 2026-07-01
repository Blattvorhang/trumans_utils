# TRUMANS — Codebase & Paper Knowledge

## Project Overview

TRUMANS (Tracking Human Actions in Scenes) is a CVPR 2024 Highlight paper. It provides:
1. A large-scale MoCap HSI dataset (15 hours, 100 indoor scenes, 1.6M frames at 30Hz)
2. A diffusion-based autoregressive model for generating human-scene interaction motions of arbitrary length

Our goal: test TRUMANS as a baseline motion generator for Unitree G1 humanoid robot. Pipeline: `room.xml → scaled voxel (.npy) → TRUMANS → SMPL motion (.npz) → Sonic + MuJoCo G1 simulation`.

## Architecture (Paper §4 + Code)

### Problem Formulation
Given 3D scene S, goal location G, and action labels A → synthesize human motion `{H_i}` of arbitrary length L.

### Two-Stage Pipeline

**Stage 1 — Diffusion Model (joint positions)**
- Predicts **24 joint 3D positions** (72-dim) directly, NOT SMPL parameters
- Why joints: (a) diffusion noise is meaningful in Euclidean space, (b) scene occupancy queries need 3D coords, (c) trajectory goal clamping directly overwrites joint xz coordinates, (d) easy min-max normalization to [-1,1]
- Backbone: Transformer encoder (8 layers, 16 heads, dim=512, dropout 0.1, FFN dim 1024) in UNet-like architecture
- Conditioned on: scene occupancy (ViT-encoded 32³ local voxel grid), action labels (Transformer-encoded with progress indicator), diffusion timestep
- 100 DDPM timesteps, linear beta schedule (0.0001–0.02)

**Stage 2 — Joints → SMPL (JointsToSMPLX)**
- MLP: Linear(72→64)→BN→ReLU→Linear(64→64)→BN→ReLU→Linear(64→132), outputs 22×6D rotation representation
- 6D rotation → axis-angle via PyTorch3D
- 100-step Adam optimization (lr=0.05): run SMPL-X forward model, compute MSE between predicted joints and SMPL-X joints, backprop to refine pose params
- Subsampled vertices (every 10th → ~1048 verts) returned for visualization

### Autoregressive Generation (Paper §4.2)

Motion is generated in **episodes** (segments), each = `seq_len` (16) frames. Episodes are chained via overlap:

```
Episode 1:  [帧0  帧1  ... 帧13  帧14  帧15]
                                    └──┬──┘
                               fixed_points (最后2帧)
                                        ↓
Episode 2:  [█帧14 █帧15  帧16  帧17  ...  帧29]
              └─固定─┘  └────────生成────────┘
```

- **Transition masking (`M_trans`)**: first `k = fixed_frame` (2) frames copied from previous episode, diffusion noise zeroed out — the model only denoises the remaining 14 frames.
- **Subgoals (`G_i`)**: overall goal G segmented per episode. For navigation: subgoal = pelvis xy at episode's conclusion (z inferred by model from scene).
- **Goal masking (`M_goal`)**: pelvis xy in last frame clamped to subgoal, diffusion noise zeroed out.

#### `fixed_points` — the autoregressive state

`fixed_points` is the **history buffer** passed between episodes. It carries the last `fixed_frame` (2) frames of 24-joint 3D positions.

| Property | Value |
|----------|-------|
| Shape | `(batch, fixed_frame, nb_joints × 3)` = `(1, 2, 72)` |
| Content | 24 joint xyz positions × 2 frames |
| Space | World coordinates (TRUMANS y-up) when stored; converted to egocentric before feeding to model |
| Role | Provides temporal context so the model generates motion continuous with the past |

#### `sample_step()` — the core diffusion loop

`sample_step()` in `sample_hsi.py` is the innermost loop that drives autoregressive generation:

```python
def sample_step(cfg, mat, obj_locs, goal_list, action_label_list, sampler_list,
                fixed_points_init=None):
    fixed_points = fixed_points_init   # None for cold start, or loaded from _state.npz
    for s in range(max_step):
        # 1. Transform fixed_points from WORLD → EGOCENTRIC (inverse of mat)
        if s != 0:
            fixed_points = normalize(transform_points(fixed_points, inverse(mat)))

        # 2. Run DDPM: first 2 frames pinned (fixed_points), last 14 denoised
        samples, occs = sampler.p_sample_loop(fixed_points, obj_locs, mat, ...)

        # 3. Extract new pelvis → compute new mat (world transform) for next episode
        pelvis_new = samples[-1, -fixed_frame:, :9]
        mat = rigid_transform_3D(pelvis_new, rest_pelvis)  # SE(2)-in-SE(3)

        # 4. Capture last 2 frames as fixed_points for next episode
        fixed_points = samples[-1, -fixed_frame:]

    return points_all, mat, fixed_points   # final state for save/resume
```

**Key detail — egocentric coordinate convention**: The model was trained on egocentric motion (root at origin). Each episode, `fixed_points` is transformed from world space into the current egocentric frame via `inverse(mat)`. The model sees: "I'm standing at the origin facing forward, and 2 frames ago my body was in *this* pose relative to me." This makes the diffusion task translation/rotation-invariant.

**`mat` (world transform)**: An SE(2)-in-SE(3) 4×4 matrix. Contains only yaw rotation + (x, 0, z) translation (y-up). No pitch/roll — the model doesn't tip over. Reconstructed each episode from the generated pelvis position relative to `rest_pelvis`.

### Local Scene Perceiver (Paper §4.3)
- Global occupancy grid `S ∈ {0,1}^(Nx×Ny×Nz)`
- Local grid extracted around current subgoal: xy-centered, z ∈ [0, 1.8m], oriented by pelvis yaw
- Encoded by ViT: patches along xy-plane, z-axis as feature channels
- Grid discretization is a necessary trade-off — real-time mesh querying via Kaolin would be ~300× slower

### Frame-wise Action Embedding (Paper §4.4)
- Action labels with **progress indicator** `A_ind ∈ [0,1]`: linearly progresses from start to finish of action
- Final label = original action + progress (values in [0,2] range)
- Processed by Transformer encoder, last token's output → MLP → action embedding

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Frames per episode (`seq_len`) | 16 | `config_sample_synhsi.yaml:30` |
| Overlap frames (`fixed_frame`) | 2 | `config_sample_synhsi.yaml:47,86` |
| Preparation steps (`len_pre`) | 4 | `config_sample_synhsi.yaml:8` |
| Action steps (`len_act`) | 0 or 2 | `config_sample_synhsi.yaml:9` |
| Joint count | 24 | `config_sample_synhsi.yaml:39` |
| Channel dim (24×3) | 72 | All model configs |
| Diffusion timesteps | 100 | `config_sample_synhsi.yaml:85` |
| Interpolation scale (`interp_s`) | 3 | `config_sample_synhsi.yaml:6` |
| Data subsampling (`step`) | 3 | `config_sample_synhsi.yaml:31` |
| Local voxel grid | 32³ | `config_sample_synhsi.yaml:32` |
| Local grid bounds (m) | [-0.6,0.6, 0,1.2, -0.6,0.6] | `config_sample_synhsi.yaml:33` |
| Global voxel resolution | 0.02 m/voxel | `mujoco_scene_to_occ.py` default |
| Output framerate | 30 FPS | `trumans_infer.py:208` |
| SMPL-H pose dim | 156 | root_orient(3)+body_pose(63)+zeros(90) |
| SMPL-X vertices (full) | 10475 | Paper §4.1 |
| Training steps | 500k | Paper §A.2 |
| Training batch size | 512 | Paper §A.2 |
| Training hardware/time | 4×A800, 48h | Paper §A.2 |

### Per-step frame math
- Diffusion generates 16 frames → interp1d ×3 = 48 output frames
- Consecutive segments overlap by 2 (fixed_frame)
- Total frames = `(num_steps × 16 - cumulative_overlap) × interp_s`

## Coordinate Systems

**TRUMANS internal is y-up — confirmed by multiple evidence sources:**

| Evidence | File | Detail |
|----------|------|--------|
| Scene grid | `datasets/trumans.py:61` | `[-3,0,-4, 3,2,4, 300,100,400]` — y∈[0,2] = height axis |
| Local mesh_grid | config | `[-0.6,0.6, 0,1.2, -0.6,0.6]` — y = vertical |
| Ground plane extraction | `sample_hsi.py:29` | `trajectory_layer[:, [0, 2]]` — xz = ground |
| Frontend Three.js | `static/index.js:152` | `point.y = 0.1` — Three.js default y-up |
| Backend receives y-up | `sample_hsi.py:271` | `get_base_speed(cfg, trajectory, is_zup=False)` — no conversion |

**Full data flow (original Flask demo):**
```
Three.js frontend (y-up) → Flask /move_cube → sample_wrapper()
  → get_base_speed(is_zup=False)  ← no conversion
  → get_guidance()                 ← zup_to_yup commented out
  → TRUMANS model (y-up internal)
```

- **MuJoCo**: z-up, (x, y, z) = (right, forward, up)
- **TRUMANS / glTF**: y-up, (x, y, z) = (left, up, forward)
- **Conversion**: MuJoCo(x, y, z) → TRUMANS**(-x, z, y)** — preserves right-handedness (det=+1).
  Naive `(x, z, y)` swap has det=-1 (reflection), which mirrors the scene and breaks the model's internal handedness, causing left/right to be swapped in generated motions.
- **`mujoco_scene_to_occ.py`** `mj_zup_to_trumans_yup`: `(-x, z, y)`
- **`mujoco_scene_to_glb.py`** `geom_to_meta`: `center = [-mj_pos[0], mj_pos[2], mj_pos[1]] * scale`
- **`trumans_infer.py`** `build_trajectory_waypoints`: start/goal in MuJoCo convention `(x=right, y=forward)`, negated to `(-x, 0.1, y)`
- **Paper's z-up mention** (Appendix A.1): "start and end points, given as (x, y) coordinates in a z-up world coordinate system" — this describes evaluation input convention only (ground-plane 2D coords). z-up (x,y) on ground ≡ y-up (x,z) on ground. TRUMANS internal is y-up.
- **`zup_to_yup()` in utils.py**: comment contradicts function name ("change from yup to zup"). NOT used in main demo flow — only in specific act_type handlers. Ignore for coordinate system understanding.
- **Storage voxels**: 0.02 m/voxel (TRUMANS standard)
- **Scale factor**: G1(~1.27m) / SMPL(~1.8m) ratio. Scene, margin, start, and goal all scale from origin during generation. After generation, the entire motion is rigidly translated so the start point lands back at its original (unscaled) position — this way the robot starts from the same place across all scales, but the goal distance changes with scale. `ground_y` (0.1m) does NOT scale.
  - **Post-shift**: `transl[:,0:2] += start[0:2] * (1 - scale)` — all frames shifted equally, poses unchanged.
  - **Two outputs saved**: `smpl_motion.npz` (shifted, start at original position) + `smpl_motion_unshifted.npz` (original scaled positions, reference).

## Trajectory & Waypoints

### Original Demo (Flask → sample_hsi.py)
- User draws trajectory on Three.js canvas → dense points
- `get_base_speed()`: sampling step = num_points / total_distance
- `get_guidance()`: subsamples trajectory at speed stride → midpoints list
- Each midpoint becomes goal for one diffusion episode
- Paper §A.1 reports using A* path planning for evaluation midpoints between start/goal. NOT implemented in our codebase or the public Flask demo — the demo uses user-drawn trajectories.

### Our CLI (trumans_infer.py)
- `build_trajectory_waypoints()` linearly interpolates 20 waypoints between start and goal (XZ plane, Y=0.1)
- `get_base_speed()`: speed = num_waypoints / distance (roughly)
- Actual speed = `int(0.6 × base_speed)`, controls waypoint subsampling stride

### Key Limitation
TRUMANS is a **trajectory-following motion generator**, not a navigation planner. The model was trained to generate motions along given waypoints — local scene awareness handles body-level collision avoidance (side-stepping, ducking) but NOT global path planning (routing around walls). For obstacle-avoiding navigation, waypoints must come from an external path planner (A*, RRT, etc.) or be user-specified.

## Closed-Loop Replanning

TRUMANS was originally designed for one-shot start→goal generation. Our extension enables **incremental generation with G1 state feedback** — the robot tracks a short horizon, reports its actual pose, and the next segment is generated from that real state.

### Motivation

In open-loop mode, `trumans_infer.py` generates the full motion from start to goal in one shot. If the G1 robot deviates from the planned trajectory (which it always will, due to physics and controller error), the remaining motion is stale. Closed-loop replanning solves this by:

1. Generate a **short horizon** (e.g. 5-8 diffusion steps ≈ 1.5–3s)
2. G1 controller tracks that segment
3. Read G1's **actual** (x, y, heading) from simulation
4. Generate the **next** segment starting from G1's real pose
5. Repeat until goal reached

### `GenerationState` — serializable autoregressive state

Defined in `nav_planning/replan_state.py`. Captures everything needed to resume generation seamlessly:

| Field | Shape | Space | Description |
|-------|-------|-------|-------------|
| `mat` | `(1, 4, 4)` | y-up world | SE(2)-in-SE(3): yaw rotation + (x, 0, z) translation |
| `fixed_points` | `(1, 2, 72)` | y-up world | Last 2 frames of 24-joint 3D positions |

Saved as `_state.npz` alongside each motion `.npz`.

### `adapt_state()` — aligning history to G1's actual pose

The core operation for replanning continuity. When G1's real position and heading differ from the planned trajectory, the history buffer must be adjusted:

```python
def adapt_state(state, new_start_zup, heading_zup):
    T = [R_yaw | t_pos]   # rigid transform: yaw rotation + translation
    # Apply SAME transform to both:
    state.mat           @= T    # generation frame moves to G1's pose
    state.fixed_points  @= T    # all joints × 2 frames move with it
```

**Why rigid transform, not translation-only**: The TRUMANS model works in **egocentric** coordinates. If only translation were applied, the body orientation in the history buffer would mismatch the new coordinate frame's yaw — producing out-of-distribution input. By applying `T = [R_yaw | t_pos]` to both `mat` and all `fixed_points`, the egocentric view is perfectly continuous: root at origin, body pose unchanged, no jump. The model sees exactly what it was trained on.

**Why body pose is preserved**: A single rigid transform applied to all 24 joints × 2 frames preserves all relative joint positions (body pose). The transform only changes where the body is and which way it faces — exactly what we want.

### Replanning workflow

```
Round 1 (cold start):
  trumans_infer.py --start 0 0 --goal 5 0 --max-steps 5
    → sample_step(fixed_points_init=None)
    → motion_r1.npz + motion_r1_state.npz

  G1 controller tracks ~1.5s → reads real (x, y, heading)

Round 2 (replan):
  trumans_infer.py --start <g1_x> <g1_y> --goal 5 0 --max-steps 5
    --resume-from motion_r1_state.npz --robot-heading <dx> <dy>
    → adapt_state(prev_state, g1_pos, g1_heading)
    → sample_step(fixed_points_init=adapted.fixed_points, mat=adapted.mat)
    → motion_r2.npz + motion_r2_state.npz

  ... repeat until goal reached ...
```

### Key design insight: planner vs. controller jitter

The SONIC controller only consumes body pose (relative joint rotations), not root position. Root position jumps in the SMPL output do **not** directly cause controller jitter. The jitter comes from the **planner** producing out-of-distribution motion when it receives discontinuous egocentric input. `adapt_state()` fixes the input, so the planner produces smooth, in-distribution motion.

### Related files

| File | Role |
|------|------|
| `nav_planning/replan_state.py` | `GenerationState` dataclass, `save_state()`, `load_state()`, `adapt_state()` |
| `nav_planning/closed_loop_plan.py` | CLI bridge for occHIPC orchestrator → `run_inference()` |
| `nav_planning/plan/README.md` | Detailed implementation plan for the replanning feature |

## Our Tools

| Tool | Purpose |
|------|---------|
| `mujoco_scene_to_occ.py` | MuJoCo XML → TRUMANS voxel occupancy (.npy + _grid.json). `--scale_height` flag controls whether Y (height) is also scaled (default False — keeps original heights for interaction) |
| `mujoco_scene_to_glb.py` | MuJoCo XML → glTF 2.0 binary (.glb) with manual buffer construction |
| `trumans_infer.py` | Start→Goal → SMPL motion .npz via TRUMANS inference. Supports `--resume-from` + `--robot-heading` for closed-loop replanning |
| `replan_state.py` | `GenerationState` save/load/adapt for autoregressive state continuity across replan cycles |
| `closed_loop_plan.py` | CLI adapter for occHIPC orchestrator → TRUMANS native API |
| `grid_search.py` | Batch grid search over scale factors |
| `app_playback.py` | Flask server for 3D visualization of .npz motions on .glb scenes |
| `templates/playback.html` | Standalone browser-based 3D motion playback (no Flask needed) |

### mujoco_scene_to_glb.py notes
- Uses manual GLB binary builder (NOT trimesh export — trimesh doesn't bind materials to primitives)
- KHR_materials_unlit for vertex-color rendering (no PBR)
- CCW winding, doubleSided: true
- Interleaved vertex buffer: position(float32×3) + color(uint8×4), 24-byte stride

### app_playback.py notes
- SMPL-X FK: `compute_vertices()` runs smplx.create with batch_size=T, subsamples every 10th vertex
- Three.js PointCloud rendering (NOT mesh faces)
- Flask serves files from root via catch-all `/<path:filename>` route
- Scans `static/`, `assets/`, `.` for GLB; `output/`, `artifacts/`, `.` for NPZ

## Key Files

```
trumans_utils/
├── config/config_sample_synhsi.yaml     # Main inference config
├── trumans/config/                      # Hydra config hierarchy
│   ├── dataset/trumans.yaml             # nb_joints=24, seq_len=16, step=3
│   ├── model/synhsi_body.yaml           # Transformer: 8L/16H/d512
│   ├── model/model_smplx.yaml           # JointsToSMPLX: 72→132
│   └── sampler/pelvis.yaml              # 100 timesteps, fixed_frame=2
├── sample_hsi.py                        # Core inference loop: get_guidance() + sample_step() (the autoregressive episode loop)
├── models/synhsi.py                     # Unet + Sampler (DDPM)
├── models/joints_to_smplx.py            # Joints→SMPL MLP + optimize_smpl()
├── datasets/trumans.py                  # TrumansDataset (scene loading, occupancy queries)
├── constants.py                         # rest_pelvis, joint names, relaxed_hand_pose
├── utils.py                             # rigid_transform_3D, transform_points, zup_to_yup
├── trumans/train_synhsi.py              # Training script
├── app.py                               # Original Flask demo
├── static/index.js                      # Original Three.js frontend
├── templates/playback.html              # Standalone 3D playback page
├── nav_planning/
│   ├── trumans_infer.py                 # CLI inference: start→goal with A*/linear planning
│   ├── replan_state.py                  # GenerationState: save/load/adapt for replanning
│   ├── closed_loop_plan.py              # occHIPC orchestrator bridge
│   ├── path_planner.py                  # A* and linear path planning over voxel grids
│   └── plan/README.md                   # Detailed implementation plan for replanning
└── paper/2403.08629v2.pdf              # CVPR 2024 paper
```

## G1-Specific Design Decisions

1. **Output format = Sonic SMPL mode**: .npz with `poses(T,156)`, `trans(T,3)`, `mocap_framerate`
2. **Scale factor concept**: G1 is shorter than SMPL, so scene is enlarged. Scale applied to: geometry horizontal positions/dims (X, Z) always; vertical (Y) only if `--scale_height` is set (default False). Margin, start, and goal also scale proportionally (origin-centered). After generation, the entire motion is rigidly translated so start lands back at its original position — this keeps the start fixed across scales while the goal distance varies. NOT applied to: ground_y. Two .npz files are saved: shifted (main) + unshifted (reference).
3. **Grid metadata sidecar**: `grid_meta/<scene>_s<scale>_grid.json` stores scene_grid bounds — loaded by `trumans_infer.py` to override TrumansDataset defaults
4. **Hand poses zeroed**: G1 doesn't have articulated hands → last 90 dims of poses are zeros
5. **Anisotropic height scaling** (`mujoco_scene_to_occ.py`): Horizontal dimensions (X, Z) are always scaled to compensate for G1's smaller body, but vertical (Y) is NOT scaled by default (`scale_height=False`). This keeps objects at their original heights so TRUMANS recognises them as interactable (e.g., bed height for sitting). Passing `--scale_height` restores uniform scaling if the scene has no interaction-critical objects.
