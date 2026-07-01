"""
TRUMANS inference: start/goal point → SMPL motion sequence (.npz).

start / goal are in MuJoCo convention: (x, y) = (right, forward) on the ground plane.
Converted to TRUMANS y-up via (-x, z, y) — a proper rotation (det=+1) that preserves
right-handedness, unlike a naive y↔z swap (det=-1, reflection).

The output .npz matches AMASS format (z-up, float64):
    poses:            (T, 156)  root_orient(3) + body_pose(63) + zeros(90)
    trans:            (T, 3)    root world translation
    betas:            (16,)     body shape parameters (zeros)
    dmpls:            (T, 8)    dynamic blend shapes (zeros)
    gender:           str       'male'
    mocap_framerate:  float64
    path:             (N, 3)    z-up waypoints (custom metadata)
    planner:          str       planner used (custom metadata)

Usage:
    python trumans_infer.py \
        --start 0 0 --goal 2 0 \
        --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \
        --output output/smpl_result.npz
"""

import argparse
import json
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class dotDict(dict):
    """dict with attribute access (matching sample_hsi.py convention)."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj):
    """Recursively convert dict → dotDict for nested access."""
    if isinstance(obj, dict):
        return dotDict({k: _to_dotdict(v) for k, v in obj.items()})
    return obj


def load_scene_meta(scene_path):
    """Load grid metadata sidecar from grid_meta/ directory."""
    scene_basename = os.path.basename(scene_path).replace(".npy", "")
    json_path = os.path.join(_PROJ, "grid_meta", f"{scene_basename}_grid.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def run_inference(
    start,
    goal,
    scene_path,
    output_path,
    config_path="config/config_sample_synhsi.yaml",
    model_joints_ckpt=None,
    model_body_ckpt=None,
    scale=None,
    planner="linear",
    clearance=0.25,
    height_range=(0.6, 0.8),
    history_path=None,
    init_fdir=None,
):
    """Load models, run TRUMANS inference, save SMPL-H .npz."""

    # Auto-detect scale from scene filename if not explicitly provided.
    # Filenames are like "room_hanyi_s1.4.npy" — extract the "s<number>" part.
    if scale is None:
        import re

        basename = os.path.basename(scene_path).replace(".npy", "")
        m = re.search(r"_s(\d+(?:\.\d+)?)", basename)
        if m:
            scale = float(m.group(1))
            print(f"[trumans_infer] Auto-detected scale={scale} from {basename}")
        else:
            scale = 1.0
            print(f"[trumans_infer] Could not detect scale from filename, using default={scale}")

    # ---- load config ----
    with open(config_path) as f:
        cfg = _to_dotdict(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    # Override checkpoint paths if provided
    if model_joints_ckpt:
        cfg.model.model_smplx.ckpt = model_joints_ckpt
    if model_body_ckpt:
        cfg.model.synhsi_body.ckpt = model_body_ckpt

    # ---- load scene metadata and override dataset grid ----
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    grid_meta = load_scene_meta(scene_path)
    if grid_meta is None:
        print(
            f"[WARN] No _grid.json found for {scene_path}; "
            f"using default TRUMANS grid bounds"
        )

    # ---- build trajectory ----
    from path_planner import plan_path

    start_scaled = [start[0] * scale, start[1] * scale]
    goal_scaled = [goal[0] * scale, goal[1] * scale]

    if planner == "astar":
        print(f"[trumans_infer] Loading occupancy grid from {scene_path} for A* planning")
        occu_3d = np.load(scene_path).astype(bool)
        trajectory = plan_path(
            start_scaled,
            goal_scaled,
            occu_3d=occu_3d,
            grid_meta=grid_meta,
            mode="astar",
            clearance=clearance,
            height_range=height_range,
        )
    else:
        trajectory = plan_path(start_scaled, goal_scaled, mode="linear")
    # trajectory is already in TRUMANS y-up. Pass to sample_hsi with is_zup=False.
    trajectory_frontend = [{"x": p[0], "y": p[1], "z": p[2]} for p in trajectory]

    # ---- import TRUMANS pipeline (lazy import to avoid CUDA init issues) ----
    from sample_hsi import (
        convert_trajectory,
        get_base_speed,
        get_guidance,
        sample_step,
    )
    from datasets.trumans import TrumansDataset
    from models.synhsi import Unet
    from models.joints_to_smplx import JointsToSMPLX, joints_to_smpl

    # ---- load models ----
    print("[trumans_infer] Loading models...")
    model_joints_to_smplx = JointsToSMPLX(**cfg.model.model_smplx)
    model_joints_to_smplx.load_state_dict(
        torch.load(cfg.model.model_smplx.ckpt, map_location=device)
    )
    model_joints_to_smplx.to(device)
    model_joints_to_smplx.eval()

    model_body = Unet(**cfg.model.synhsi_body)
    model_body.load_state_dict(
        torch.load(cfg.model.synhsi_body.ckpt, map_location=device)
    )
    model_body.to(device)
    model_body.eval()

    # ---- load dataset ----
    # Set scene_name to match the custom scene file (dataset uses substring match)
    scene_basename = os.path.basename(scene_path).replace(".npy", "")
    cfg.scene_name = scene_basename

    synhsi_dataset = TrumansDataset(**cfg.dataset)

    # Override scene grid if metadata available
    if grid_meta is not None:
        sg = grid_meta["scene_grid"]
        synhsi_dataset.scene_grid_np = np.array(sg, dtype=np.float64)
        synhsi_dataset.scene_grid_torch = torch.tensor(sg, dtype=torch.float32).to(
            device
        )
        print(f"[trumans_infer] Overrode scene_grid: {sg}")

    # ---- setup sampler ----
    import hydra

    sampler_body = hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler_body.set_dataset_and_model(synhsi_dataset, model_body)
    samplers = {"body": sampler_body, "hand": None}

    # ---- run inference pipeline ----
    trajectory_np = convert_trajectory(trajectory_frontend)
    obj_locs = {}  # no dynamic objects for now

    base_speed = get_base_speed(cfg, trajectory_np, is_zup=False)
    mat, goal_list, action_label_list, sampler_list = get_guidance(
        cfg,
        trajectory_np,
        samplers,
        act_type=cfg.action_type,
        speed=int(0.6 * base_speed),
    )
    print(f"[trumans_infer] Diffusion steps: {len(goal_list)}")

    # ---- load and adapt autoregressive state for seamless replanning ----
    fixed_points_init = None
    if history_path is not None and os.path.exists(history_path):
        from nav_planning.replan_state import load_state, adapt_state
        prev_state = load_state(history_path)
        heading = init_fdir if init_fdir is not None else (0.0, 1.0)
        adapted = adapt_state(prev_state, (float(start[0]), float(start[1])), heading)
        fixed_points_init = torch.from_numpy(adapted.fixed_points).float().to(device)
        # Override mat with the adapted transform so the generation frame
        # matches the rigidly-transformed history buffer.
        mat = torch.from_numpy(adapted.mat).float().to(device)
        print(f"[trumans_infer] Loaded state, adapted: start={start}, heading={heading}")
    else:
        print("[trumans_infer] Cold start (no history)")

    points_all, final_mat, final_fixed_points = sample_step(
        cfg, mat, obj_locs, goal_list, action_label_list, sampler_list,
        fixed_points_init=fixed_points_init,
    )

    # ---- save state for next replan cycle ----
    from nav_planning.replan_state import GenerationState, save_state as save_gs
    base_out, _ext = os.path.splitext(output_path)
    state_path = f"{base_out}_state.npz"
    save_gs(GenerationState(
        mat=final_mat.cpu().numpy().astype(np.float32),
        fixed_points=final_fixed_points.cpu().numpy().astype(np.float32),
    ), state_path)
    print(f"[trumans_infer] Saved state → {state_path}")

    # ---- convert to SMPL params ----
    all_poses = []
    all_transl = []

    for i in range(cfg.batch_size):
        keypoint_gene_torch = (
            torch.from_numpy(points_all[i])
            .reshape(-1, cfg.dataset.nb_joints * 3)
            .to(device)
        )
        pose, transl, _, _, _ = joints_to_smpl(
            model_joints_to_smplx,
            keypoint_gene_torch,
            cfg.dataset.joints_ind,
            cfg.interp_s,
        )
        all_poses.append(pose)
        all_transl.append(transl)

    pose_out = all_poses[0]  # (T, 66): global_orient(3) + body_pose(63)
    transl_out = all_transl[0]  # (T, 3) — TRUMANS y-up

    # ---- convert from TRUMANS y-up to MuJoCo z-up ----
    # y-up (x=left, y=up, z=forward) → z-up (x=right, y=forward, z=up)
    # Point mapping: (x, y, z) → (-x, z, y)  — pure rotation, det=+1
    from scipy.spatial.transform import Rotation as R

    R_yup_to_zup = R.from_matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Convert global_orient: axis-angle rotation in y-up world frame → z-up world frame
    global_orient_yup = pose_out[:, :3]  # (T, 3) axis-angle
    global_orient_zup = (R_yup_to_zup * R.from_rotvec(global_orient_yup)).as_rotvec()
    pose_out[:, :3] = global_orient_zup

    # Convert transl: TRUMANS y-up point → z-up point
    transl_out_zup = np.stack(
        [-transl_out[:, 0], transl_out[:, 2], transl_out[:, 1]], axis=1
    )

    # Convert path waypoints: y-up → z-up
    trajectory_zup = np.stack(
        [-trajectory[:, 0], trajectory[:, 2], trajectory[:, 1]], axis=1
    )

    T = pose_out.shape[0]
    fps = 30.0  # TRUMANS dataset base FPS, interp_s compensates

    # Assemble SMPL-H format: poses (T, 156) = root_orient(3) + body_pose(63) + zeros(90)
    poses_full = np.zeros((T, 156), dtype=np.float64)
    poses_full[:, 0:3] = pose_out[:, :3]  # global_orient
    poses_full[:, 3:66] = pose_out[:, 3:]  # body_pose
    # poses_full[:, 66:156] stays zero (hand poses — unused by G1)

    # ---- build AMASS-format arrays (all z-up, float64) ----
    betas = np.zeros(16, dtype=np.float64)          # body shape params
    dmpls = np.zeros((T, 8), dtype=np.float64)      # dynamic blend shapes
    gender = np.array("male", dtype="<U7")            # SMPL gender label

    # Ensure transl is float64
    transl_out_zup = transl_out_zup.astype(np.float64)

    # ---- post-generation shift: undo scale-induced offset on start point ----
    # Generation uses origin-centered scaling: start_scaled = start * scale.
    # We want the start to land back at the original (unscaled) start position,
    # so we rigidly translate the entire motion by the offset.
    # MuJoCo (sx, sy) → z-up output: (sx, sy, ground_y)
    # Shift in z-up: [start_x*(1-scale), start_y*(1-scale), 0]
    shift_x = start[0] * (1.0 - scale)
    shift_y = start[1] * (1.0 - scale)

    # Save unshifted version first (reference, old behavior)
    base, ext = os.path.splitext(output_path)
    unshifted_path = f"{base}_unshifted{ext}"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        unshifted_path,
        poses=poses_full,
        trans=transl_out_zup,  # original scaled positions
        betas=betas,
        dmpls=dmpls,
        gender=gender,
        mocap_framerate=np.array(fps, dtype=np.float64),
        path=trajectory_zup,  # original waypoints
        planner=np.array(planner),
    )
    print(
        f"[trumans_infer] Saved unshifted {T} frames → {unshifted_path} "
        f"(transl range z-up: {transl_out_zup.min(axis=0)} → {transl_out_zup.max(axis=0)})"
    )

    # Apply shift: bring start back to original position (all frames + path)
    transl_out_zup[:, 0] += shift_x
    transl_out_zup[:, 1] += shift_y
    trajectory_zup[:, 0] += shift_x
    trajectory_zup[:, 1] += shift_y

    # ---- save shifted version (start at original position, main output) ----
    np.savez(
        output_path,
        poses=poses_full,
        trans=transl_out_zup,
        betas=betas,
        dmpls=dmpls,
        gender=gender,
        mocap_framerate=np.array(fps, dtype=np.float64),
        path=trajectory_zup,  # shifted waypoints
        planner=np.array(planner),
    )
    print(
        f"[trumans_infer] Saved shifted {T} frames → {output_path} "
        f"(transl range z-up: {transl_out_zup.min(axis=0)} → {transl_out_zup.max(axis=0)})"
    )

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRUMANS inference: start/goal → SMPL motion .npz"
    )
    parser.add_argument(
        "--start", nargs=2, type=float, required=True, help="Start point in MuJoCo convention: (x, y) = (right, forward)"
    )
    parser.add_argument(
        "--goal", nargs=2, type=float, required=True, help="Goal point in MuJoCo convention: (x, y) = (right, forward)"
    )
    parser.add_argument(
        "--scene", required=True, help="Path to scene .npy (with _grid.json sidecar)"
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--config",
        default="config/config_sample_synhsi.yaml",
        help="TRUMANS config YAML",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale factor. Generation uses origin-centered scaling for scene + start/goal. "
             "After generation, the entire motion is rigidly translated so start "
             "lands back at its original (unscaled) position. "
             "Auto-detected from scene filename if not set (e.g. room_hanyi_s1.4.npy → 1.4)",
    )
    parser.add_argument(
        "--planner",
        choices=["linear", "astar"],
        default="linear",
        help="Global path planner: linear (straight line) or astar (A* on voxel grid)",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.25,
        help="Robot clearance in metres for A* C-space dilation (default: 0.25)",
    )
    parser.add_argument(
        "--height_min",
        type=float,
        default=0.6,
        help="Min height in metres for 2D occ projection (default: 0.6, waist)",
    )
    parser.add_argument(
        "--height_max",
        type=float,
        default=0.8,
        help="Max height in metres for 2D occ projection (default: 0.8)",
    )

    args = parser.parse_args()

    run_inference(
        start=args.start,
        goal=args.goal,
        scene_path=args.scene,
        output_path=args.output,
        config_path=args.config,
        scale=args.scale,
        planner=args.planner,
        clearance=args.clearance,
        height_range=(args.height_min, args.height_max),
    )
