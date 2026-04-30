import pycolmap
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Dict

from ..dataset import Dataset, Pose


def mean_position_orientation_error(refined_poses: Dict[int, Pose], poses: Dict[int, Pose]) -> Tuple[np.ndarray, np.ndarray]:
    pos_d, rot_d = [], []
    for ts, refined_pose in refined_poses.items():
        pose = poses[ts]

        pos_d.append(pose.get_position() - refined_pose.get_position())

        r_c = R.from_quat(refined_pose.get_quaternion())
        r_i = R.from_quat(pose.get_quaternion())
        r_diff = r_i * r_c.inv()
        rot_d.append(r_diff.as_quat())

    avg_pos = np.mean(pos_d, axis=0)
    avg_rot = R.from_quat(rot_d).mean().as_quat()

    return avg_pos, avg_rot


def interpolate(poses: Dict[int, Pose], refined_poses: Dict[int, Pose], colmap_R_world: np.ndarray) -> Dict[int, Pose]:
    # Source: https://deepwiki.com/colmap/pycolmap/4.4-geometric-transformations#rigid3d
    # Source: https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.Rigid3d

    n = len(refined_poses)
    pose_timestamps = np.array(sorted(poses.keys()))
    refined_timestamps = np.array(sorted(refined_poses.keys()))

    inter_poses = {}
    for ts in pose_timestamps:
        idx = (np.abs(refined_timestamps - ts)).argmin()

        r_ts = refined_timestamps[idx]

        delta_t = abs(ts - r_ts)
        if ts >= r_ts and idx < n - 1:
            pose_1 = refined_poses[refined_timestamps[idx]].to_pycolmap()
            pose_2 = refined_poses[refined_timestamps[idx + 1]].to_pycolmap()
            delta_t /= refined_timestamps[idx + 1] - refined_timestamps[idx]
        elif ts <= r_ts and idx > 0:
            pose_1 = refined_poses[refined_timestamps[idx - 1]].to_pycolmap()
            pose_2 = refined_poses[refined_timestamps[idx]].to_pycolmap()
            delta_t /= refined_timestamps[idx] - refined_timestamps[idx - 1]
        else:
            continue

        interpolated_pose = Pose.from_pycolmap(ts, pycolmap.Rigid3d.interpolate(pose_1, pose_2, delta_t))
        inter_poses[ts] = interpolated_pose.rotate(colmap_R_world)

    return inter_poses


def interpolate_poses(dataset: Dataset, reconstruction: pycolmap.Reconstruction) -> Tuple[Dict[int, Pose], Dict[int, Pose]]:
    """
    Returns the interpolated camera and sonar poses based on COLMAP's sparse refinement.
    """
    refined_poses = {}
    for image in reconstruction.images.values():
        # Invert from cam_from_world to world_from_cam (same as dataset poses)
        timestamp = int(image.name.replace(".jpg", ''))
        refined_poses[timestamp] = Pose.from_pycolmap(timestamp, image.cam_from_world().inverse())

    # Find the average rotation difference to correct for COLMAP's arbitrary global frame.
    c_poses = {ts: img.pose for ts, img in dataset.images.items()}
    s_poses = {ts: sss.navigation.pose for ts, sss in dataset.sonar.items()}

    _, colmap_R_world = mean_position_orientation_error(refined_poses, c_poses)

    camera_poses = interpolate(c_poses, refined_poses, colmap_R_world)
    sonar_poses = interpolate(s_poses, refined_poses, colmap_R_world)

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

        points_2d.append((int(round(u)), int(round(v))))
        points_3d.append((x, y, z))

    return np.array(points_2d), np.array(points_3d)


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


def get_distances(pose: Pose, points: np.ndarray) -> np.ndarray:
    """
    Parameters:
        pose: AUV pose object
        points: np.ndarray that is nx3

    Returns:
        np.ndarray: An nx1 numpy array where each element is the distance in meters.
    """
    return np.linalg.norm(points - pose.get_position(), axis=1)


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

    # Rotate to Body frame to get correct cross-track distance
    ned_R_body = pose.get_rotation_matrix().T
    v_body = (ned_R_body @ v_ned.T).T

    opposite = np.abs(v_ned[:, 2])   # Depth/Z difference
    adjacent = np.abs(v_body[:, 1])  # Cross-track/Y difference in body frame

    return np.arctan(adjacent / (opposite + eps))