"""
Global path planning on TRUMANS voxel occupancy grids.

Two modes:
  linear  — straight-line interpolation (current behavior)
  astar   — A* on 2D-projected occupancy, smoothed & densified

The output in astar mode is a dense trajectory (np.array Mx3 y-up), matching
the format of convert_trajectory() from the original Flask demo. The existing
get_base_speed() + get_guidance() pipeline handles stride-based midpoint
subsampling from this dense trajectory unchanged.
"""

import heapq
import math

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def project_occupancy(occ_3d, y_min_idx, y_max_idx):
    """Project 3D occupancy to 2D XZ plane.

    For each (x, z) column, the cell is blocked if ANY voxel in the
    y-range [y_min_idx, y_max_idx] is occupied.

    Args:
        occ_3d: (nx, ny, nz) bool array, TRUMANS y-up.
        y_min_idx, y_max_idx: int, y-axis index range (inclusive).

    Returns:
        (nx, nz) bool array, True = blocked.
    """
    ny = occ_3d.shape[1]
    lo = max(0, y_min_idx)
    hi = min(ny - 1, y_max_idx)
    if lo > hi:
        return np.zeros((occ_3d.shape[0], occ_3d.shape[2]), dtype=bool)
    return np.any(occ_3d[:, lo : hi + 1, :], axis=1)


def dilate_obstacles(occ_2d, radius_voxels):
    """Dilate occupied cells by a disk of given radius (C-space expansion).

    Args:
        occ_2d: (nx, nz) bool array.
        radius_voxels: int, dilation radius in voxels.

    Returns:
        (nx, nz) bool array.
    """
    if radius_voxels <= 0:
        return occ_2d.copy()
    from scipy.ndimage import binary_dilation

    ys, xs = np.ogrid[-radius_voxels : radius_voxels + 1, -radius_voxels : radius_voxels + 1]
    kernel = (xs**2 + ys**2) <= radius_voxels**2
    return binary_dilation(occ_2d, structure=kernel)


def world_to_grid_2d(wx, wz, scene_grid):
    """Convert world coordinates (TRUMANS y-up) to 2D grid indices.

    Args:
        wx, wz: float, world X and Z (y-up convention).
        scene_grid: length-9 list [x_min, y_min, z_min, x_max, y_max, z_max, nx, ny, nz].

    Returns:
        (int, int) grid indices (ix, iz), clamped to valid range.
    """
    ox, oy, oz = scene_grid[0], scene_grid[1], scene_grid[2]
    ex, ey, ez = scene_grid[3], scene_grid[4], scene_grid[5]
    nx, ny, nz = int(scene_grid[6]), int(scene_grid[7]), int(scene_grid[8])
    vx = (ex - ox) / nx
    vz = (ez - oz) / nz
    ix = int((wx - ox) / vx)
    iz = int((wz - oz) / vz)
    ix = max(0, min(nx - 1, ix))
    iz = max(0, min(nz - 1, iz))
    return ix, iz


def grid_to_world_2d(ix, iz, scene_grid):
    """Convert 2D grid indices back to world coordinates (voxel centre).

    Args:
        ix, iz: int, grid indices.
        scene_grid: length-9 list.

    Returns:
        (float, float) world (wx, wz) in TRUMANS y-up.
    """
    ox, _, oz = scene_grid[0], scene_grid[1], scene_grid[2]
    ex, _, ez = scene_grid[3], scene_grid[4], scene_grid[5]
    nx, _, nz = int(scene_grid[6]), int(scene_grid[7]), int(scene_grid[8])
    vx = (ex - ox) / nx
    vz = (ez - oz) / nz
    wx = ox + (ix + 0.5) * vx
    wz = oz + (iz + 0.5) * vz
    return wx, wz


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

# 8-connected: 4 cardinal + 4 diagonal
_NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]


def astar(grid, start, goal):
    """A* search on a 2D boolean grid, 8-connected.

    Args:
        grid: (nx, nz) bool array, True = blocked.
        start: (int, int) start grid index.
        goal: (int, int) goal grid index.

    Returns:
        List of (ix, iz) indices from start to goal, or empty list if no path.
    """
    nx, nz = grid.shape
    sx, sz = start
    gx, gz = goal

    if grid[sx, sz] or grid[gx, gz]:
        return []

    open_set = []
    heapq.heappush(open_set, (0.0, sx, sz))
    came_from = {}
    g_score = {(sx, sz): 0.0}
    closed = set()

    while open_set:
        _, cx, cz = heapq.heappop(open_set)
        if (cx, cz) in closed:
            continue
        closed.add((cx, cz))

        if (cx, cz) == (gx, gz):
            path = [(cx, cz)]
            while (cx, cz) in came_from:
                cx, cz = came_from[(cx, cz)]
                path.append((cx, cz))
            path.reverse()
            return path

        for dx, dz, cost in _NEIGHBORS:
            nx_, nz_ = cx + dx, cz + dz
            if nx_ < 0 or nx_ >= nx or nz_ < 0 or nz_ >= nz:
                continue
            if grid[nx_, nz_]:
                continue
            if (nx_, nz_) in closed:
                continue
            tentative = g_score[(cx, cz)] + cost
            if tentative < g_score.get((nx_, nz_), float("inf")):
                came_from[(nx_, nz_)] = (cx, cz)
                g_score[(nx_, nz_)] = tentative
                h = math.sqrt((nx_ - gx) ** 2 + (nz_ - gz) ** 2)
                heapq.heappush(open_set, (tentative + h, nx_, nz_))

    return []


# ---------------------------------------------------------------------------
# Path smoothing & densification
# ---------------------------------------------------------------------------


def _bresenham_line(x0, z0, x1, z1):
    """Yield integer grid cells along the line from (x0,z0) to (x1,z1)."""
    dx = abs(x1 - x0)
    dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz

    cx, cz = x0, z0
    while True:
        yield cx, cz
        if (cx, cz) == (x1, z1):
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            cx += sx
        if e2 < dx:
            err += dx
            cz += sz


def _has_line_of_sight(p1, p2, occ_2d):
    """Check whether the line segment from p1 to p2 is obstacle-free.

    Args:
        p1, p2: (int, int) grid indices.
        occ_2d: (nx, nz) bool array — True = blocked (undilated).

    Returns:
        True if no occupied cell is traversed, False otherwise.
    """
    for ix, iz in _bresenham_line(p1[0], p1[1], p2[0], p2[1]):
        if occ_2d[ix, iz]:
            return False
    return True


def smooth_path(path_indices, occ_2d):
    """Prune a grid path by iteratively skipping to the furthest visible waypoint.

    Args:
        path_indices: list of (ix, iz) from A*.
        occ_2d: (nx, nz) undilated obstacle grid for line-of-sight checks.

    Returns:
        Pruned list of (ix, iz).
    """
    if len(path_indices) <= 2:
        return list(path_indices)

    result = [path_indices[0]]
    i = 0
    while i < len(path_indices) - 1:
        j = len(path_indices) - 1
        while j > i + 1:
            if _has_line_of_sight(path_indices[i], path_indices[j], occ_2d):
                break
            j -= 1
        result.append(path_indices[j])
        i = j
    return result


def densify_path(path_world, spacing=0.05):
    """Interpolate a sparse world-coord path back to a dense trajectory.

    Given the pruned waypoints in world coordinates (N, 3) y-up,
    linearly interpolate along the path at fixed spatial intervals.
    This produces a trajectory with density comparable to the user-drawn
    curves from the original Flask demo, suitable for get_base_speed().

    Args:
        path_world: (N, 3) np.array of waypoints in TRUMANS y-up.
        spacing: float, desired point spacing in metres (default 0.05).

    Returns:
        (M, 3) np.array of densely-sampled trajectory points.
    """
    if len(path_world) <= 1:
        return path_world.copy()

    segments = np.asarray(path_world, dtype=np.float32)

    # Compute cumulative arc length
    seg_dists = np.linalg.norm(segments[1:] - segments[:-1], axis=1)
    total_dist = seg_dists.sum()

    if total_dist < spacing:
        return path_world.copy()

    n_samples = max(int(math.ceil(total_dist / spacing)), 2)
    dense = np.zeros((n_samples, 3), dtype=np.float32)
    dense[0] = segments[0]

    cumsum = np.concatenate([[0.0], np.cumsum(seg_dists)])
    for i in range(1, n_samples):
        t = (i / (n_samples - 1)) * total_dist
        seg_idx = np.searchsorted(cumsum, t, side="right") - 1
        seg_idx = max(0, min(len(seg_dists) - 1, seg_idx))
        local_t = (t - cumsum[seg_idx]) / seg_dists[seg_idx] if seg_dists[seg_idx] > 0 else 0.0
        local_t = max(0.0, min(1.0, local_t))
        dense[i] = segments[seg_idx] + local_t * (segments[seg_idx + 1] - segments[seg_idx])

    return dense


# ---------------------------------------------------------------------------
# Linear interpolation (existing behaviour relocated from trumans_infer.py)
# ---------------------------------------------------------------------------


def build_trajectory_waypoints(start_2d, goal_2d, num_waypoints=20, ground_y=0.1):
    """Linearly interpolate between start and goal on the XZ plane.

    start / goal are in MuJoCo convention: (x, y) = (right, forward).
    Converted to TRUMANS y-up via (-x, z, y) which preserves handedness (det=+1).

    Returns (N, 3) array of waypoints in y-up TRUMANS coords.
    """
    start = np.array([-start_2d[0], ground_y, start_2d[1]], dtype=np.float32)
    goal = np.array([-goal_2d[0], ground_y, goal_2d[1]], dtype=np.float32)
    return np.linspace(start, goal, num_waypoints)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def plan_path(
    start_2d,
    goal_2d,
    occu_3d=None,
    grid_meta=None,
    mode="linear",
    num_waypoints=20,
    ground_y=0.1,
    clearance=0.25,
    height_range=(0.6, 0.8),
):
    """Plan a trajectory from start to goal.

    Args:
        start_2d: (float, float) start in MuJoCo convention (x=right, y=forward).
        goal_2d:  (float, float) goal in MuJoCo convention.
        occu_3d: (nx, ny, nz) bool, occupancy grid (TRUMANS y-up). Required for astar.
        grid_meta: dict with 'scene_grid' and 'resolution'. Required for astar.
        mode: 'linear' or 'astar'.
        num_waypoints: int, number of waypoints for linear mode.
        ground_y: float, fixed Y height for all waypoints (default 0.1).
        clearance: float, robot clearance in metres (astar only).
        height_range: (float, float) min/max height in metres for 2D projection
                      (astar only — default 0.6 to 0.8, waist height).

    Returns:
        (N, 3) np.array of waypoints in TRUMANS y-up coords.
        For linear mode: N = num_waypoints.
        For astar mode: N varies (dense trajectory, typically 50-500 points).
    """
    if mode == "linear":
        return build_trajectory_waypoints(start_2d, goal_2d, num_waypoints, ground_y)

    if mode == "astar":
        if occu_3d is None or grid_meta is None:
            print("[path_planner] WARNING: A* mode requires occu_3d and grid_meta. "
                  "Falling back to linear.")
            return build_trajectory_waypoints(start_2d, goal_2d, num_waypoints, ground_y)

        scene_grid = grid_meta["scene_grid"]
        resolution = grid_meta.get("resolution", 0.02)

        # Convert start/goal from MuJoCo to TRUMANS y-up
        start_yup = np.array([-start_2d[0], ground_y, start_2d[1]], dtype=np.float32)
        goal_yup = np.array([-goal_2d[0], ground_y, goal_2d[1]], dtype=np.float32)

        # Compute y-index range from height_range
        ny = int(scene_grid[7])
        vy = (scene_grid[4] - scene_grid[1]) / ny
        y_min_idx = int(round((height_range[0] - scene_grid[1]) / vy))
        y_max_idx = int(round((height_range[1] - scene_grid[1]) / vy))
        y_min_idx = max(0, min(ny - 1, y_min_idx))
        y_max_idx = max(0, min(ny - 1, y_max_idx))

        print(f"[path_planner] 2D projection: y ∈ [{height_range[0]}, {height_range[1]}]m "
              f"→ indices [{y_min_idx}, {y_max_idx}]")

        # Project 3D → 2D
        occ_2d = project_occupancy(occu_3d, y_min_idx, y_max_idx)
        print(f"[path_planner] 2D grid: {occ_2d.shape}, occupied={occ_2d.sum()}/{occ_2d.size}")

        # Dilate for C-space
        radius_voxels = int(math.ceil(clearance / resolution))
        dilated = dilate_obstacles(occ_2d, radius_voxels)
        added = int(dilated.sum()) - int(occ_2d.sum())
        print(f"[path_planner] Dilated by {radius_voxels} voxels ({clearance}m), "
              f"added {added} blocked cells")

        # World → grid
        sx, sz = world_to_grid_2d(start_yup[0], start_yup[2], scene_grid)
        gx, gz = world_to_grid_2d(goal_yup[0], goal_yup[2], scene_grid)
        print(f"[path_planner] Start grid ({sx},{sz}), goal grid ({gx},{gz})")

        if dilated[sx, sz]:
            print("[path_planner] WARNING: start point inside dilated obstacle. "
                  "Falling back to linear.")
            return build_trajectory_waypoints(start_2d, goal_2d, num_waypoints, ground_y)
        if dilated[gx, gz]:
            print("[path_planner] WARNING: goal point inside dilated obstacle. "
                  "Falling back to linear.")
            return build_trajectory_waypoints(start_2d, goal_2d, num_waypoints, ground_y)

        # A*
        grid_path = astar(dilated, (sx, sz), (gx, gz))
        if not grid_path:
            print("[path_planner] WARNING: A* found no path. Falling back to linear.")
            return build_trajectory_waypoints(start_2d, goal_2d, num_waypoints, ground_y)

        print(f"[path_planner] A* raw path: {len(grid_path)} grid cells")

        # Smooth (prune)
        pruned = smooth_path(grid_path, dilated)  # check against dilated grid to respect clearance at corners
        print(f"[path_planner] After pruning: {len(pruned)} waypoints")

        # Convert grid → world
        waypoints_world = []
        for ix, iz in pruned:
            wx, wz = grid_to_world_2d(ix, iz, scene_grid)
            waypoints_world.append([wx, ground_y, wz])
        waypoints_world = np.array(waypoints_world, dtype=np.float32)

        # Densify to match user-drawn trajectory density
        dense = densify_path(waypoints_world, spacing=0.05)
        print(f"[path_planner] Densified: {len(dense)} points ({dense.shape[0]} pts / "
              f"{np.linalg.norm(dense[-1] - dense[0], axis=-1):.1f}m path)")

        return dense

    raise ValueError(f"Unknown planner mode: {mode}")
