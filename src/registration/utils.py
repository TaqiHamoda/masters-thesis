import pycolmap
import numpy as np
from typing import Tuple, List, Dict

from ..dataset import Dataset, Pose


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


def interpolate_poses(dataset: Dataset, reconstruction: pycolmap.Reconstruction) -> Tuple[Dict[int, Pose], Dict[int, Pose]]:
    """
    Returns the interpolated camera and sonar poses based on COLMAP's sparse refinement.
    """
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


def get_image_geometry(reconstruction: pycolmap.Reconstruction, img_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        reconstruction: sparse reconstruction object.
        img_name: The name of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two nx3 numpy arrays that contain the 2D points in the image frame along with their corresponding 3D points in NED frame.
    """
    image = reconstruction.image(img_id)
    if image.num_points3D < 0:
        return np.array([]), np.array([])

    points_2d, points_3d = [], []
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue

        point3d = reconstruction.points3D[point2d.point3D_id]

        u, v = point2d.xy
        x, y, z = point3d.xyz

        points_2d.append((int(u), int(v)))
        points_3d.append((x, y, z))

    return np.array(points_2d), np.array(points_3d)


def get_distances(pose: Pose, points: np.ndarray) -> np.ndarray:
    """
    Parameters:
        pose: AUV pose object
        points: np.ndarray that is nx3

    Returns:
        np.ndarray: An nx1 numpy array where each element is the distance in meters.
    """

    return np.linalg.norm(points - pose.get_position(), axis=1)


def get_intersections(pose: Pose, points: np.ndarray, n_local: np.ndarray = np.array([1.0, 0.0, 0.0]), thickness: float = 0.01):
    """
    Parameters:
        pose: AUV pose object
        points: np.ndarray that is nx3
        n_local: Local normal vector to define a plane. Default assumes YZ plane fan (standard for Side-Scan Sonar).
        thickness: Plane thickness in meters.

    Returns:
        np.ndarray: A boolean array of which points intersect the plane.
        If a triangle is within half the thickness of the plane, it is marked as True.
    """
    body_R_ned = pose.get_rotation_matrix()

    plane_pos = pose.get_position()
    plane_normal = body_R_ned @ n_local  # Rotate the local normal vector to the global frame

    # shortest distance to the plane
    distance = np.dot(plane_normal, (points - plane_pos).T).T
    return np.abs(distance) <= thickness / 2.0


def get_corresponding_channels(pose: Pose, points: np.ndarray) -> np.ndarray:
    """
    Parameters:
        pose: AUV pose object
        points: np.ndarray that is nx3

    Returns:
        np.ndarray: An nx1 numpy array where each element is 0 for port or 1 for starboard.
    """
    ned_R_body = pose.get_rotation_matrix().T

    v_ned = points - pose.get_position()
    v_body = (ned_R_body @ v_ned.T).T

    # Check Y component distance in local frame. If positive, then starboard is closer
    # If negative, then port is closer.
    return np.astype(v_body[:, 1] > 0, int)


def get_incidence_angles(pose: Pose, points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Parameters:
        pose: AUV pose object
        points: np.ndarray that is nx3

    Returns:
        np.ndarray: An nx1 numpy array where each element is the incidence angle in radians.
    """

    v_ned = points - pose.get_position()

    opposite = v_ned[:, 2]  # Z difference
    adjacent = v_ned[:, 1]  # Y difference

    return np.arctan(opposite / (adjacent + eps))