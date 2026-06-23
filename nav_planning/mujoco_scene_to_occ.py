"""
Convert MuJoCo XML scene → TRUMANS-format voxel occupancy grid (.npy + _grid.json).

Usage:
    python mujoco_scene_to_occ.py \
        --xml assets/scene_asset/room_hanyi/room.xml \
        --scale 1.4 --margin 2.0 \
        --out_scene Data_blocks_motion_all/Scene/room_hanyi_s1.4.npy
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_vec(s, n=3):
    """Parse space-separated floats from XML attribute, return np.array of length n."""
    return np.array([float(x) for x in s.split()])[:n]


def _quat_to_rotmat(quat_str):
    """MuJoCo quaternion: w x y z → 3×3 rotation matrix."""
    w, x, y, z = _parse_vec(quat_str, 4)
    return R.from_quat([x, y, z, w]).as_matrix()


def mj_zup_to_trumans_yup(point):
    """Convert a single 3D point: MuJoCo (x, y, z) → TRUMANS (-x, z, y).

    MuJoCo uses right-handed z-up:  (x→right, y→forward, z→up).
    TRUMANS uses right-handed y-up: (x→left, y→up, z→forward).

    Simple (x,z,y) swap has det=-1 (reflection). Negating x yields det=+1,
    preserving handedness: R_x(-90°) @ [x, y, z]^T = [x, z, -y]^T."""
    return np.array([-point[0], point[2], point[1]])


def mj_size_to_trumans(size, geom_type):
    """Convert geom half-extents from MuJoCo z-up to TRUMANS y-up.

    MuJoCo: (x, y, z) where z=up
    TRUMANS: (x, y, z) where y=up

    For box:      swap y↔z
    For cylinder: radius stays, height goes from z→y
    For plane:    x-extent stays, y-extent goes to z, thickness goes to y
    """
    if geom_type == "plane":
        x_half, y_half = size[0], size[1]
        thickness = size[2] if len(size) > 2 else 0.075
        return [x_half, thickness, y_half]
    elif geom_type in ("box", "cylinder"):
        return [size[0], size[2], size[1]]
    return size


# ---------------------------------------------------------------------------
# geom collection
# ---------------------------------------------------------------------------


def collect_typed_boxes(xml_path, scale=1.0, scale_height=False):
    """Parse MuJoCo XML → list of (center_3, half_extents_3, geom_type) in TRUMANS y-up.

    Returns typed tuples so plane geoms can be identified downstream.
    Handles <body> transforms by accumulating parent pos + quat.

    Horizontal dimensions (X, Z) are always multiplied by *scale*.
    Vertical dimension (Y, height) is multiplied by *scale* only if
    *scale_height* is True (default False — keeps objects at train-set heights
    so TRUMANS can still recognise them as interactable).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sy = scale if scale_height else 1.0
    scale_vec = np.array([scale, sy, scale])

    boxes = []  # list of (center, half, type)

    def walk_body(body_elem, parent_pos, parent_rot):
        body_pos = _parse_vec(body_elem.get("pos", "0 0 0"))
        body_quat_str = body_elem.get("quat")
        if body_quat_str:
            body_rot = _quat_to_rotmat(body_quat_str)
        else:
            body_rot = np.eye(3)

        world_pos = parent_pos + parent_rot @ body_pos
        world_rot = parent_rot @ body_rot

        for child in body_elem:
            tag = child.tag.lower()
            if tag == "geom":
                gtype = child.get("type", "sphere")
                if gtype not in ("box", "plane", "cylinder"):
                    continue

                geom_pos_local = _parse_vec(child.get("pos", "0 0 0"))
                geom_pos_world = world_pos + world_rot @ geom_pos_local

                if gtype == "plane":
                    size_raw = _parse_vec(child.get("size", "1 1 0.1"))
                    half = mj_size_to_trumans(size_raw, "plane")
                    center_tr = mj_zup_to_trumans_yup(geom_pos_world)
                    boxes.append((center_tr * scale_vec,
                                  np.array(half) * scale_vec, "plane"))

                elif gtype == "box":
                    size_raw = _parse_vec(child.get("size", "0 0 0"))
                    half_tr = mj_size_to_trumans(size_raw, "box")
                    center_tr = mj_zup_to_trumans_yup(geom_pos_world)
                    boxes.append((center_tr * scale_vec,
                                  np.abs(np.array(half_tr)) * scale_vec, "box"))

                elif gtype == "cylinder":
                    parts = _parse_vec(child.get("size", "0 0"))
                    r, h = parts[0], parts[1]
                    half_tr = [r, h, r]
                    center_tr = mj_zup_to_trumans_yup(geom_pos_world)
                    boxes.append((center_tr * scale_vec,
                                  np.abs(np.array(half_tr)) * scale_vec, "cylinder"))

            elif tag == "body":
                walk_body(child, world_pos, world_rot)

    wb = root.find("worldbody")
    if wb is None:
        raise ValueError(f"No <worldbody> found in {xml_path}")

    for child in wb:
        tag = child.tag.lower()
        if tag == "body":
            walk_body(child, np.zeros(3), np.eye(3))
        elif tag == "geom":
            gtype = child.get("type", "sphere")
            if gtype not in ("box", "plane", "cylinder"):
                continue
            geom_pos = _parse_vec(child.get("pos", "0 0 0"))

            if gtype == "plane":
                size_raw = _parse_vec(child.get("size", "1 1 0.1"))
                half = mj_size_to_trumans(size_raw, "plane")
                center_tr = mj_zup_to_trumans_yup(geom_pos)
                boxes.append((center_tr * scale_vec,
                              np.array(half) * scale_vec, "plane"))

            elif gtype == "box":
                size_raw = _parse_vec(child.get("size", "0 0 0"))
                half_tr = mj_size_to_trumans(size_raw, "box")
                center_tr = mj_zup_to_trumans_yup(geom_pos)
                boxes.append((center_tr * scale_vec,
                              np.abs(np.array(half_tr)) * scale_vec, "box"))

            elif gtype == "cylinder":
                parts = _parse_vec(child.get("size", "0 0"))
                r, h = parts[0], parts[1]
                half_tr = [r, h, r]
                center_tr = mj_zup_to_trumans_yup(geom_pos)
                boxes.append((center_tr * scale_vec,
                              np.abs(np.array(half_tr)) * scale_vec, "cylinder"))

    return boxes


# ---------------------------------------------------------------------------
# voxelisation
# ---------------------------------------------------------------------------


def build_scene_occupancy(xml_path, scale=1.0, scale_height=False, margin=2.0, resolution=0.02):
    """Build TRUMANS-format occupancy grid from MuJoCo XML.

    Returns
    -------
    occu  : np.ndarray bool  [nx, ny, nz]  TRUMANS y-up
    bbox  : dict  {x_min, y_min, z_min, x_max, y_max, z_max}
    shape : tuple (nx, ny, nz)
    """
    typed = collect_typed_boxes(xml_path, scale=scale, scale_height=scale_height)

    if not typed:
        raise ValueError(f"No solid geoms found in {xml_path}")

    sy = scale if scale_height else 1.0
    scale_vec = np.array([scale, sy, scale])

    # Separate planes from solid geometry so plane extent can be clipped
    plane_boxes = [(c, h) for c, h, t in typed if t == "plane"]
    solid_boxes = [(c, h) for c, h, t in typed if t != "plane"]

    if solid_boxes:
        solid_min = np.array([c - h for c, h in solid_boxes]).min(axis=0)
        solid_max = np.array([c + h for c, h in solid_boxes]).max(axis=0)
    else:
        # No non-plane geometry — use planes as-is
        solid_min = np.array([c - h for c, h in plane_boxes]).min(axis=0) if plane_boxes else np.zeros(3)
        solid_max = np.array([c + h for c, h in plane_boxes]).max(axis=0) if plane_boxes else np.zeros(3)

    # Clip plane x/z extent to solid geometry bounds (planes are infinite in rendering
    # size — we only care about them where they physically interact with the solid scene)
    clipped_planes = []
    for center, half in plane_boxes:
        c = center.copy()
        h = half.copy()
        x_lo = max(c[0] - h[0], solid_min[0])
        x_hi = min(c[0] + h[0], solid_max[0])
        if x_hi > x_lo:
            c[0] = (x_lo + x_hi) / 2.0
            h[0] = (x_hi - x_lo) / 2.0
        z_lo = max(c[2] - h[2], solid_min[2])
        z_hi = min(c[2] + h[2], solid_max[2])
        if z_hi > z_lo:
            c[2] = (z_lo + z_hi) / 2.0
            h[2] = (z_hi - z_lo) / 2.0
        clipped_planes.append((c, h))

    all_boxes = solid_boxes + clipped_planes

    all_min = np.array([c - h for c, h in all_boxes]).min(axis=0)
    all_max = np.array([c + h for c, h in all_boxes]).max(axis=0)

    effective_margin = margin * scale_vec
    llb = all_min - effective_margin  # lower-left-back
    rub = all_max + effective_margin  # right-upper-back

    grid_shape = tuple(np.ceil((rub - llb) / resolution).astype(int))
    occu = np.zeros(grid_shape, dtype=bool)

    for center, half in all_boxes:
        lo = np.floor((center - half - llb) / resolution).astype(int)
        hi = np.ceil((center + half - llb) / resolution).astype(int)
        lo = np.clip(lo, 0, np.array(grid_shape) - 1)
        hi = np.clip(hi, 0, np.array(grid_shape) - 1)
        occu[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = True

    bbox = {
        "x_min": float(llb[0]),
        "y_min": float(llb[1]),
        "z_min": float(llb[2]),
        "x_max": float(rub[0]),
        "y_max": float(rub[1]),
        "z_max": float(rub[2]),
    }

    # scene_grid format matching TrumansDataset.scene_grid_np:
    # [x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz]
    grid_meta = {
        "scene_grid": [
            float(bbox["x_min"]),
            float(bbox["y_min"]),
            float(bbox["z_min"]),
            float(bbox["x_max"]),
            float(bbox["y_max"]),
            float(bbox["z_max"]),
            int(grid_shape[0]),
            int(grid_shape[1]),
            int(grid_shape[2]),
        ],
        "resolution": float(resolution),
        "scale": float(scale),
        "scale_height": bool(scale_height),
        "margin": float(margin),
    }

    print(
        f"[mujoco_scene_to_occ] "
        f"scale={scale} scale_height={scale_height} margin={margin} res={resolution}m "
        f"grid={list(grid_shape)} "
        f"occupied={int(occu.sum())}/{occu.size} voxels "
        f"bounds=({bbox['x_min']:.1f},{bbox['y_min']:.1f},{bbox['z_min']:.1f}) → "
        f"({bbox['x_max']:.1f},{bbox['y_max']:.1f},{bbox['z_max']:.1f})"
    )

    return occu, grid_meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MuJoCo XML → TRUMANS-format voxel occupancy grid"
    )
    parser.add_argument(
        "--xml", required=True, help="Path to MuJoCo scene XML"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor for SMPL↔G1 (default 1.0)"
    )
    parser.add_argument(
        "--scale_height",
        action="store_true",
        default=False,
        help="Also scale vertical (Y) dimension (default: False — keep original heights for interaction)",
    )
    parser.add_argument(
        "--margin", type=float, default=2.0, help="Extra free space beyond geometry in metres (default 2.0)"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.02,
        help="Voxel size in metres (default 0.02, TRUMANS standard)",
    )
    parser.add_argument(
        "--out_scene", required=True, help="Output path for .npy file"
    )

    args = parser.parse_args()

    occu, grid_meta = build_scene_occupancy(
        args.xml,
        scale=args.scale,
        scale_height=args.scale_height,
        margin=args.margin,
        resolution=args.resolution,
    )

    # Save .npy
    np.save(args.out_scene, occu)
    print(f"Saved occupancy grid to {args.out_scene}")

    # Save _grid.json sidecar in grid_meta/ (not Scene/ — avoids dataset scan)
    scene_basename = os.path.basename(args.out_scene).replace(".npy", "")
    meta_dir = "grid_meta"
    os.makedirs(meta_dir, exist_ok=True)
    json_path = os.path.join(meta_dir, f"{scene_basename}_grid.json")
    with open(json_path, "w") as f:
        json.dump(grid_meta, f, indent=2)
    print(f"Saved grid metadata to {json_path}")
