#!/usr/bin/env python3
"""TRUMANS planner wrapper for closed-loop control.

Thin CLI → ``run_inference()`` adapter.  Accepts standard args from the
occHIPC orchestrator (via ``planner_bridge.py``) and converts them to the
TRUMANS native API.

Usage::

    conda run -n trumans python nav_planning/closed_loop_plan.py \\
        --init_pos 1.2,3.4,0.78 --init_fdir 0.0,1.0,0.0 \\
        --init_path /tmp/start.npz --init_frame 7 \\
        --tgt_limb /tmp/tgt_limb_abs.npy \\
        --output_npz /tmp/seq_new.npz \\
        --history_path /tmp/prev_state.npz \\
        --scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy \\
        --planner astar \\
        --config config/config_sample_synhsi.yaml
"""

import argparse
import os
import sys
import traceback

# Ensure trumans_utils is on sys.path so that nav_planning imports work.
_project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_vec3(s: str):
    """Parse comma-separated float string → numpy array [3]."""
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return np.array(parts, dtype=np.float64)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TRUMANS planner — closed-loop wrapper"
    )

    # ---- Standard args (from planner_bridge.py unified interface) ----
    parser.add_argument("--init_pos", type=str, required=True,
                        help="Start root position x,y,z (Z-up)")
    parser.add_argument("--init_fdir", type=str, required=True,
                        help="Start forward direction x,y,z (Z-up)")
    parser.add_argument("--init_path", type=str, required=True,
                        help="Path to start-state .npz (IGNORED by TRUMANS — "
                             "generates motion from position + goal)")
    parser.add_argument("--init_frame", type=int, default=0,
                        help="Frame index within init_path (IGNORED by TRUMANS)")
    parser.add_argument("--tgt_limb", type=str, required=True,
                        help="Path to target ELIMBS .npy [5,3] (Z-up world)")
    parser.add_argument("--output_npz", type=str, required=True,
                        help="Output .npz path")
    parser.add_argument("--history_path", type=str, default="",
                        help="Path to _state.npz from previous cycle "
                             "(empty = cold start)")

    # ---- TRUMANS-specific args ----
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to TRUMANS voxel grid .npy "
                             "(with grid_meta/<name>_grid.json sidecar)")
    parser.add_argument("--config", type=str,
                        default="config/config_sample_synhsi.yaml",
                        help="TRUMANS config YAML (relative to trumans_utils root)")
    parser.add_argument("--planner", type=str, default="astar",
                        choices=["linear", "astar"],
                        help="Global path planner mode")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scene scale factor (auto-detected from scene filename "
                             "if not set, e.g. room_hanyi_s1.4.npy → 1.4)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Convert position
    # ------------------------------------------------------------------
    init_pos = _parse_vec3(args.init_pos)
    start = (float(init_pos[0]), float(init_pos[1]))  # drop Z

    # ------------------------------------------------------------------
    # 2. Heading — 2D (dx, dy) for state adaptation
    # ------------------------------------------------------------------
    init_fdir_3d = _parse_vec3(args.init_fdir)
    fdir_2d = (float(init_fdir_3d[0]), float(init_fdir_3d[1]))

    # ------------------------------------------------------------------
    # 3. Convert target: root of tgt_limb → 2D goal
    # ------------------------------------------------------------------
    if not os.path.exists(args.tgt_limb):
        raise FileNotFoundError(f"Target ELIMBS not found: {args.tgt_limb}")
    tgt_limb = np.load(args.tgt_limb)  # [5, 3]
    goal = (float(tgt_limb[0, 0]), float(tgt_limb[0, 1]))

    # ------------------------------------------------------------------
    # 4. Resolve config path (may be relative to trumans_utils root)
    # ------------------------------------------------------------------
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_project_root, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"TRUMANS config not found: {config_path}")

    # ------------------------------------------------------------------
    # 5. Resolve scene path (may be relative)
    # ------------------------------------------------------------------
    scene_path = args.scene
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(_project_root, scene_path)

    # ------------------------------------------------------------------
    # 6. Resolve output path to absolute (before we chdir)
    # ------------------------------------------------------------------
    output_path_abs = os.path.abspath(args.output_npz)

    # ------------------------------------------------------------------
    # 7. Switch to trumans_utils root (config + checkpoint paths are relative to it)
    # ------------------------------------------------------------------
    _orig_cwd = os.getcwd()
    os.chdir(_project_root)
    print(f"[trumans_bridge] CWD → {_project_root}")

    # ------------------------------------------------------------------
    # 8. Resolve history — None if empty or missing
    # ------------------------------------------------------------------
    history = (args.history_path
               if (args.history_path and os.path.exists(args.history_path))
               else None)
    if history:
        print(f"[trumans_bridge] Resuming from: {history}")
    else:
        print(f"[trumans_bridge] Cold start")

    # ------------------------------------------------------------------
    # 9. Call TRUMANS inference
    # ------------------------------------------------------------------
    print(f"[trumans_bridge] "
          f"start={start}, goal={goal}, "
          f"scene={os.path.basename(scene_path)}, "
          f"planner={args.planner}")
    print(f"[trumans_bridge] output → {output_path_abs}")

    try:
        from nav_planning.trumans_infer import run_inference

        run_inference(
            start=start,
            goal=goal,
            scene_path=scene_path,
            output_path=output_path_abs,
            config_path=config_path,
            scale=args.scale,
            planner=args.planner,
            history_path=history,
            init_fdir=fdir_2d,
        )
    except Exception:
        print(f"[trumans_bridge] FATAL: inference failed", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 9. Verify output
    # ------------------------------------------------------------------
    if not os.path.exists(output_path_abs):
        print(f"[trumans_bridge] FATAL: output not found: {output_path_abs}",
              file=sys.stderr)
        sys.exit(1)

    data = np.load(output_path_abs, allow_pickle=True)
    n_frames = data["poses"].shape[0]
    fps = float(data.get("mocap_framerate", 30.0))

    print(f"[trumans_bridge] OK: {n_frames} frames @ {fps} Hz")


if __name__ == "__main__":
    main()
