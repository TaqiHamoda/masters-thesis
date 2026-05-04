import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple

from ..dataset import Pose, DeltaHit
from .utils import get_distances, get_distances_to_plane


def _optimize_extrinsics(
        delta_pose: Tuple[float, float, float],
        delta_hits: List[DeltaHit],
        sss_poses: List[Pose],
        slant_range: float,
        num_samples: int
):
    delta_position = np.array(delta_pose[:3])

    losses = []
    for d_hit in delta_hits:
        offset = np.array((d_hit.hit.offset_x, d_hit.hit.offset_y, d_hit.hit.offset_z))

        pose = sss_poses[d_hit.hit.ping_idx + d_hit.delta_ping]
        pose = pose.translate(offset).translate(delta_position)

        point = np.array((d_hit.hit.p_x, d_hit.hit.p_y, d_hit.hit.p_z))

        ping_loss = get_distances_to_plane(pose, point)

        bin_idx = d_hit.hit.bin_idx + d_hit.delta_bin
        bin_loss = bin_idx % num_samples                                          # Wrap around
        bin_loss = bin_loss if bin_idx > num_samples else num_samples - bin_loss  # Account for channel
        bin_loss = bin_loss * slant_range / num_samples                           # Convert to meters
        bin_loss -= get_distances(pose, point)                                    # Calculate difference

        losses.append(np.power(ping_loss[0], 2) + np.power(bin_loss[0], 2))

    return np.array(losses)


def optimize_extrinsics(
    delta_hits: List[DeltaHit],
    sss_poses: List[Pose],
    slant_range: float,
    num_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    res = least_squares(
        lambda x: _optimize_extrinsics(x, delta_hits, sss_poses, slant_range, num_samples),
        (0, 0, 0),
        method='trf',
        jac='3-point',
        verbose=2
    )

    position_delta = res.x[:3]
    rotation_delta = res.x[3:]

    return position_delta, rotation_delta
