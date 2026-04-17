import pycolmap
import numpy as np
from typing import Tuple, List, Dict

from .dataset import Dataset, Pose
from .photogrammetry import Photogrammetry


def interpolate(pose_timestamps: List[int], refined_timestamps: List[int], refined_poses: Dict[int, pycolmap.Rigid3d]) -> Dict[int, Pose]:
    # Source: https://deepwiki.com/colmap/pycolmap/4.4-geometric-transformations#rigid3d
    # Source: https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Rigid3d

    inter_poses = {}

    n = len(refined_timestamps)
    for ts in pose_timestamps:
        idx = (np.abs(refined_timestamps - ts)).argmin()

        r_ts = refined_timestamps[idx]

        delta_t = abs(ts - r_ts)
        if ts > r_ts and idx < n - 1:
            pose_1 = refined_poses[refined_timestamps[idx]]
            pose_2 = refined_poses[refined_timestamps[idx + 1]]
            delta_t /= refined_timestamps[idx + 1] - refined_timestamps[idx]
        elif ts <= r_ts and idx > 0:
            pose_1 = refined_poses[refined_timestamps[idx - 1]]
            pose_2 = refined_poses[refined_timestamps[idx]]
            delta_t /= refined_timestamps[idx] - refined_timestamps[idx - 1]
        else:
            continue

        # Invert from cam_from_world to world_from_cam (same as dataset poses)
        interpolated_pose = pycolmap.Rigid3d.interpolate(pose_1, pose_2, delta_t).inverse()

        inter_poses[ts] = Pose(
            timestamp=ts,
            qx=interpolated_pose.params[0],
            qy=interpolated_pose.params[1],
            qz=interpolated_pose.params[2],
            qw=interpolated_pose.params[3],
            x=interpolated_pose.params[4],
            y=interpolated_pose.params[5],
            z=interpolated_pose.params[6],
        )

    return inter_poses


def interpolate_poses(dataset: Dataset) -> Tuple[Dict[int, Pose], Dict[int, Pose]]:
    """
    Returns the interpolated camera and sonar poses based on COLMAP's sparse refinement.
    """
    reconstruction = Photogrammetry.get_reconstruction(dataset)

    refined_poses = {}
    for image in reconstruction.images.values():
        timestamp = int(image.name.replace(".jpg", ''))
        refined_poses[timestamp] = image.cam_from_world()

    refined_timestamps = np.array(sorted(refined_poses.keys()))

    camera_poses = interpolate(dataset.images.keys(), refined_timestamps, refined_poses)
    sonar_poses = interpolate(dataset.sonar.keys(), refined_timestamps, refined_poses)

    # Correct the extrinsics for sonar pose (colmap is wrt camera frame)
    for ts in list(sonar_poses.keys()):
        pose = sonar_poses[ts]
        pose = pose.translate(-1 * dataset.camera_trans)
        pose = pose.translate(dataset.sonar_trans)

        sonar_poses[ts] = pose

    return camera_poses, sonar_poses
