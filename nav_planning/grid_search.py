"""
Batch grid search over scale factors for TRUMANS baseline.

For each scale: XML → voxel .npy → TRUMANS → SMPL .npz

Usage:
    python grid_search.py \
        --xml assets/scene_asset/room_hanyi/room.xml \
        --start 0 0 --goal 2 0 \
        --scales 1.0,1.1,1.2,1.3,1.4,1.5,1.6 \
        --output_dir output/
"""

import argparse
import json
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_scales(scales_str):
    return [float(s.strip()) for s in scales_str.split(",")]


def main():
    os.chdir(_PROJ)  # all relative paths are from project root

    parser = argparse.ArgumentParser(
        description="Grid search over scale factors for TRUMANS"
    )
    parser.add_argument("--xml", required=True, help="Path to MuJoCo scene XML")
    parser.add_argument("--start", nargs=2, type=float, required=True, help="Start (x z)")
    parser.add_argument("--goal", nargs=2, type=float, required=True, help="Goal (x z)")
    parser.add_argument(
        "--scales",
        type=str,
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6",
        help="Comma-separated scale factors",
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="Scene margin in metres"
    )
    parser.add_argument(
        "--planner",
        choices=["linear", "astar"],
        default="linear",
        help="Global path planner: linear (straight line) or astar (A* on voxel grid)",
    )
    parser.add_argument(
        "--clearance", type=float, default=0.25,
        help="Robot clearance in metres for A* C-space dilation",
    )
    parser.add_argument(
        "--height_min", type=float, default=0.6,
        help="Min height in metres for 2D occ projection (default: 0.6, waist)",
    )
    parser.add_argument(
        "--height_max", type=float, default=0.8,
        help="Max height in metres for 2D occ projection (default: 0.8)",
    )
    parser.add_argument(
        "--output_dir", default="output", help="Output root directory"
    )

    args = parser.parse_args()
    scales = parse_scales(args.scales)

    print(f"Grid search: {len(scales)} scales {scales}")
    print(f"XML: {args.xml}")
    print(f"Start: {args.start} → Goal: {args.goal}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Deduce scene name from XML path
    scene_name = os.path.basename(os.path.dirname(args.xml))

    # Import once
    from mujoco_scene_to_occ import build_scene_occupancy
    from trumans_infer import run_inference

    results = {}
    n_scene_dir = "Data_blocks_motion_all/Scene"

    for scale in scales:
        scale_dir = os.path.join(args.output_dir, f"scale_{scale}")
        os.makedirs(scale_dir, exist_ok=True)

        npy_name = f"{scene_name}_s{scale}.npy"
        npy_path = os.path.join(n_scene_dir, npy_name)
        npz_path = os.path.join(scale_dir, "smpl_motion.npz")

        # Step 1: XML → occupancy .npy
        if not os.path.exists(npy_path):
            print(f"[scale={scale}] Building occupancy grid...")
            _occu, _meta = build_scene_occupancy(
                args.xml, scale=scale, margin=args.margin
            )
            import numpy as np

            np.save(npy_path, _occu)

            # Save grid metadata
            meta_dir = "grid_meta"
            os.makedirs(meta_dir, exist_ok=True)
            meta_path = os.path.join(meta_dir, f"{scene_name}_s{scale}_grid.json")
            with open(meta_path, "w") as f:
                json.dump(_meta, f)
            print(f"  occupancy → {npy_path}")
            print(f"  metadata   → {meta_path}")
        else:
            print(f"[scale={scale}] Occupancy grid exists, skipping build")

        # Step 2: TRUMANS inference (start/goal scaled internally by run_inference)
        if not os.path.exists(npz_path):
            print(f"[scale={scale}] Running TRUMANS inference...")
            run_inference(
                start=args.start,
                goal=args.goal,
                scene_path=npy_path,
                output_path=npz_path,
                scale=scale,
                planner=args.planner,
                clearance=args.clearance,
                height_range=(args.height_min, args.height_max),
            )
        else:
            print(f"[scale={scale}] Inference result exists, skipping")

        results[str(scale)] = {
            "npy": npy_path,
            "npz": npz_path,
        }
        print()

    # Save summary
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "xml": args.xml,
                "start": list(args.start),
                "goal": list(args.goal),
                "scales": [float(s) for s in scales],
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Summary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
