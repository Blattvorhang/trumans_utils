"""
GenerationState — serializable snapshot of the TRUMANS autoregressive state.

Stored at the end of each generation run, consumed by the next replan cycle
to ensure seamless motion continuity across independent script invocations.

Coordinate convention:
    All arrays are in TRUMANS y-up WORLD space (x=left, y=up, z=forward).
    External interfaces (start/goal, heading) use z-up (x=right, y=forward, z=up).
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class GenerationState:
    """Autoregressive state captured between diffusion episodes.

    Attributes:
        mat:           (1, 4, 4) float32 — y-up world transform.
                       SE(2) embedded in 4x4: yaw-only rotation + (x, 0, z) translation.
        fixed_points:  (1, fixed_frame, 72) float32 — last fixed_frame frames
                       of 24-joint 3D positions, in y-up world space.
    """
    mat: np.ndarray
    fixed_points: np.ndarray


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------

def save_state(state: GenerationState, path: str) -> None:
    """Save GenerationState to a .npz file."""
    np.savez_compressed(
        path,
        mat=state.mat.astype(np.float32),
        fixed_points=state.fixed_points.astype(np.float32),
    )


def load_state(path: str) -> GenerationState:
    """Load GenerationState from a .npz file."""
    data = np.load(path)
    return GenerationState(
        mat=data["mat"].astype(np.float32),
        fixed_points=data["fixed_points"].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# state adaptation — the core replanning operation
# ---------------------------------------------------------------------------

def adapt_state(
    state: GenerationState,
    new_start_zup: tuple[float, float],
    heading_zup: tuple[float, float],
) -> GenerationState:
    """Rigidly transform the history buffer to align with G1's actual pose.

    Applies the SAME rigid transform T = [R|t] (translation + yaw rotation)
    to the entire history buffer, so that in the egocentric frame of the new
    mat, the root is at origin and the body pose is perfectly continuous.

    Rationale:
        The TRUMANS model operates in an EGOCENTRIC coordinate frame — it
        expects the root (pelvis) to be at the origin, with history frames
        showing body pose relative to that root.  If G1's real position and
        heading change between replan cycles (which they always do), we must
        pull the ENTIRE history buffer by the same rigid transform so the
        model's egocentric view remains continuous.

        A translation-only shift would leave a heading mismatch between the
        history buffer's body orientation and the new mat's yaw, causing
        the model to receive out-of-distribution input → jitter in the
        generated motion → controller oscillation.

    Args:
        state:          GenerationState from the previous run.
        new_start_zup:  G1's actual (x, y) ground position, z-up convention.
        heading_zup:    G1's actual heading vector (dx, dy) in z-up ground plane.
    """
    # ---- 1. convert G1 position from z-up to y-up ----
    new_pos_yup = np.array(
        [-new_start_zup[0], 0.0, new_start_zup[1]], dtype=np.float32
    )

    # ---- 2. convert heading vector to y-up yaw ----
    dx, dy = heading_zup
    yaw_yup = math.atan2(-dx, dy)

    # ---- 3. build new_mat from G1 position + heading ----
    rot_yup = R.from_rotvec(np.array([0, yaw_yup, 0])).as_matrix()
    new_mat = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
    new_mat[0, :3, :3] = rot_yup
    new_mat[0, 0, 3] = new_pos_yup[0]  # x
    new_mat[0, 2, 3] = new_pos_yup[2]  # z
    # Y translation stays 0 — height is in the joint positions

    # ---- 4. rigid transform: old world → new world ----
    old_mat = state.mat
    T = new_mat[0] @ np.linalg.inv(old_mat[0])  # (4, 4)

    fp = state.fixed_points[0]  # (F, 72)
    fp_reshaped = fp.reshape(-1, 3)

    ones = np.ones((fp_reshaped.shape[0], 1), dtype=np.float32)
    fp_transformed = (T @ np.concatenate([fp_reshaped, ones], axis=1).T).T
    fp_new = fp_transformed[:, :3].reshape(1, fp.shape[0], 72)

    return GenerationState(mat=new_mat, fixed_points=fp_new)
