"""
Convert MuJoCo XML scene → simplified GLB mesh for visualization.

Applies the same scale factor as mujoco_scene_to_occ.py so the GLB visual
matches the .npy occupancy grid.

Usage:
    python mujoco_scene_to_glb.py \
        --xml assets/scene_asset/room_hanyi/room.xml \
        --scale 1.4 \
        --out assets/scene_asset/room_hanyi/room_s1.4.glb
"""

import argparse
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _parse_vec(s, n=3):
    return np.array([float(x) for x in s.split()])[:n]


def _quat_to_rotmat(quat_str):
    """MuJoCo quaternion (w x y z) → 3×3 rotation matrix."""
    w, x, y, z = _parse_vec(quat_str, 4)
    return R.from_quat([x, y, z, w]).as_matrix()


# ---------------------------------------------------------------------------
# box geometry builder
# ---------------------------------------------------------------------------

# Indices for a box: 6 faces × 2 triangles, CCW winding (visible from outside)
BOX_FACES = np.array([
    [0, 1, 2, 0, 2, 3],   # +z
    [4, 6, 5, 4, 7, 6],   # -z
    [0, 4, 5, 0, 5, 1],   # +y
    [2, 6, 7, 2, 7, 3],   # -y
    [0, 3, 7, 0, 7, 4],   # +x
    [1, 5, 6, 1, 6, 2],   # -x
], dtype=np.uint16)


def _box_vertices(center, half):
    """Generate 8 vertices for a box."""
    c = np.asarray(center, dtype=np.float32)
    h = np.asarray(half, dtype=np.float32)
    verts = np.array([
        c + [-h[0], -h[1],  h[2]],
        c + [ h[0], -h[1],  h[2]],
        c + [ h[0], -h[1], -h[2]],
        c + [-h[0], -h[1], -h[2]],
        c + [-h[0],  h[1],  h[2]],
        c + [ h[0],  h[1],  h[2]],
        c + [ h[0],  h[1], -h[2]],
        c + [-h[0],  h[1], -h[2]],
    ], dtype=np.float32)
    return verts


# ---------------------------------------------------------------------------
# GLB builder
# ---------------------------------------------------------------------------


def _pad(data, align=4):
    while len(data) % align != 0:
        data += b' '
    return data


def _build_glb(meshes_meta):
    """Build a GLB binary from mesh metadata.

    meshes_meta: list of dicts with 'center', 'half' (glTF y-up), 'color' (3 floats 0-1)
    """
    buffers = b''
    accessors = []
    buffer_views = []
    meshes = []
    nodes = []
    current_offset = 0

    for i, meta in enumerate(meshes_meta):
        center = meta['center']
        half = meta['half']
        color = meta['color']

        # Build vertex data: position + color per vertex
        verts = _box_vertices(center, half)  # (8, 3)
        # Repeat color for each vertex
        vert_colors = np.tile(np.array(color, dtype=np.float32), (8, 1))  # (8, 3)

        # Interleave position and color
        vertex_data = np.zeros(8, dtype=[
            ('pos', np.float32, 3),
            ('color', np.float32, 3),
        ])
        vertex_data['pos'] = verts
        vertex_data['color'] = vert_colors

        vertex_bytes = vertex_data.tobytes()
        vertex_bytes = _pad(vertex_bytes)

        index_bytes = BOX_FACES.astype(np.uint16).tobytes()
        index_bytes = _pad(index_bytes)

        # Buffer views
        bv_vertex = {
            'buffer': 0,
            'byteOffset': current_offset,
            'byteLength': len(vertex_bytes),
            'target': 34962,  # ARRAY_BUFFER
            'byteStride': 24,  # 3*4 + 3*4
        }
        bv_index = {
            'buffer': 0,
            'byteOffset': current_offset + len(vertex_bytes),
            'byteLength': len(index_bytes),
            'target': 34963,  # ELEMENT_ARRAY_BUFFER
        }

        bv_vertex_idx = len(buffer_views)
        bv_index_idx = len(buffer_views) + 1
        buffer_views.extend([bv_vertex, bv_index])

        current_offset += len(vertex_bytes) + len(index_bytes)
        buffers += vertex_bytes + index_bytes

        # Accessors
        acc_pos = {
            'bufferView': bv_vertex_idx,
            'componentType': 5126,  # FLOAT
            'count': 8,
            'type': 'VEC3',
            'min': (center - half).tolist(),
            'max': (center + half).tolist(),
        }
        acc_color = {
            'bufferView': bv_vertex_idx,
            'byteOffset': 12,  # start of color in the interleaved struct
            'componentType': 5126,
            'count': 8,
            'type': 'VEC3',
        }
        acc_idx = {
            'bufferView': bv_index_idx,
            'componentType': 5123,  # UNSIGNED_SHORT
            'count': 36,
            'type': 'SCALAR',
        }
        acc_pos_idx = len(accessors)
        accessors.extend([acc_pos, acc_color, acc_idx])

        mesh = {
            'primitives': [{
                'attributes': {
                    'POSITION': acc_pos_idx,
                    'COLOR_0': acc_pos_idx + 1,
                },
                'indices': acc_pos_idx + 2,
                'material': 0,
                'mode': 4,  # TRIANGLES
            }]
        }
        meshes.append(mesh)
        nodes.append({'mesh': i})

    # One material with vertex colors via KHR_materials_unlit
    materials = [{
        'pbrMetallicRoughness': {
            'baseColorFactor': [0.8, 0.8, 0.8, 1.0],
            'roughnessFactor': 0.5,
            'metallicFactor': 0.0,
        },
        'extensions': {
            'KHR_materials_unlit': {},
        },
        'doubleSided': True,
    }]

    gltf = {
        'asset': {'version': '2.0'},
        'scene': 0,
        'scenes': [{'nodes': list(range(len(nodes)))}],
        'nodes': nodes,
        'meshes': meshes,
        'accessors': accessors,
        'bufferViews': buffer_views,
        'buffers': [{'byteLength': current_offset}],
        'materials': materials,
        'extensionsUsed': ['KHR_materials_unlit'],
    }

    json_bytes = json.dumps(gltf).encode('utf-8')
    json_bytes = _pad(json_bytes, 4)

    # Assemble GLB
    total = 12 + 8 + len(json_bytes) + 8 + len(buffers)
    header = b'glTF' + struct.pack('<II', 2, total)

    chunk_json = struct.pack('<I', len(json_bytes)) + b'JSON' + json_bytes
    chunk_bin = struct.pack('<I', len(buffers)) + b'BIN\x00' + buffers

    return header + chunk_json + chunk_bin


# ---------------------------------------------------------------------------
# geom collection
# ---------------------------------------------------------------------------


def geom_to_meta(elem, world_pos, world_rot, materials, scale):
    """Convert MuJoCo geom → dict with center, half, color in glTF y-up."""
    gtype = elem.get("type", "sphere")
    if gtype not in ("box", "plane", "cylinder"):
        return None

    rgba_01 = np.array([0.7, 0.7, 0.7, 1.0])
    mat_ref = elem.get("material")
    if mat_ref and mat_ref in materials:
        rgba_01 = materials[mat_ref]
    elif elem.get("rgba"):
        rgba_01 = _parse_vec(elem.get("rgba"), 4)

    color = np.clip(rgba_01[:3], 0, 1).tolist()

    # Convert MuJoCo z-up → glTF y-up
    geom_pos_local = _parse_vec(elem.get("pos", "0 0 0"))
    muj_pos = world_pos + world_rot @ geom_pos_local
    center = np.array([-muj_pos[0], muj_pos[2], muj_pos[1]]) * scale

    if gtype == "box":
        half = _parse_vec(elem.get("size", "0.1 0.1 0.1"))
        half_gl = np.array([half[0], half[2], half[1]]) * scale

    elif gtype == "cylinder":
        parts = _parse_vec(elem.get("size", "0.1 0.1"), 2)
        r, h_mj = parts[0] * scale, parts[1] * scale
        half_gl = np.array([r, h_mj, r])

    elif gtype == "plane":
        size_raw = _parse_vec(elem.get("size", "1 1 0.1"))
        half_gl = np.array([size_raw[0], 0.01, size_raw[1]]) * scale

    return {'center': center, 'half': half_gl, 'color': color}


def walk_body(body_elem, parent_pos, parent_rot, materials, scale, meta_list):
    """Recurse into <body>, collecting geom metadata."""
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
            m = geom_to_meta(child, world_pos, world_rot, materials, scale)
            if m is not None:
                meta_list.append(m)
        elif tag == "body":
            walk_body(child, world_pos, world_rot, materials, scale, meta_list)


def collect_meta(xml_path, scale=1.0):
    """Parse MuJoCo XML → list of mesh metadata dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    materials = {}
    asset = root.find("asset")
    if asset is not None:
        for mat in asset.findall("material"):
            name = mat.get("name")
            if name and mat.get("rgba"):
                materials[name] = _parse_vec(mat.get("rgba"), 4)

    meta_list = []

    wb = root.find("worldbody")
    if wb is None:
        raise ValueError(f"No <worldbody> in {xml_path}")

    for child in wb:
        tag = child.tag.lower()
        if tag == "body":
            walk_body(child, np.zeros(3), np.eye(3), materials, scale, meta_list)
        elif tag == "geom":
            m = geom_to_meta(child, np.zeros(3), np.eye(3), materials, scale)
            if m is not None:
                meta_list.append(m)

    return meta_list


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MuJoCo XML → simplified GLB mesh (same scale as occupancy)"
    )
    parser.add_argument("--xml", required=True, help="Path to MuJoCo scene XML")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--out", required=True, help="Output GLB path")

    args = parser.parse_args()

    meta = collect_meta(args.xml, scale=args.scale)

    if not meta:
        raise SystemExit("No solid geoms found in XML")

    glb = _build_glb(meta)

    with open(args.out, 'wb') as f:
        f.write(glb)

    print(f"Exported {len(meta)} meshes to {args.out}")
