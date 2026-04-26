import numpy as np
from typing import List, Dict

from .utils import get_distances, get_intersections, get_corresponding_channels, get_incidence_angles
from ..dataset import Dataset, Pose, ImageHit


def process_optical_sidescan_matches(
    dataset: Dataset,
    pose: Pose,
    sss_poses: Dict[int, Pose],
    optical: np.ndarray,
    points: np.ndarray,
    sonar_offset: np.ndarray
) -> List[ImageHit]:
    matches = []
    for s_ts in sss_poses.keys():
        sss = dataset.sonar[s_ts]
        sss_pose = sss_poses[s_ts]

        is_valid = get_intersections(sss_pose, points)

        dists = get_distances(sss_pose, points[is_valid])
        is_valid[is_valid] = dists < sss.slant_range

        if not np.any(is_valid):
            continue

        p_inters = points[is_valid]
        o_inters = optical[is_valid]

        distances = get_distances(sss_pose, p_inters)
        channels = get_corresponding_channels(sss_pose, p_inters)
        incidence_angles = get_incidence_angles(sss_pose, p_inters)

        bins = sss.num_samples * distances / sss.slant_range
        bins = sss.num_samples + np.power(-1, 1 - channels) * bins
        bins = np.round(bins).astype(int)

        for i in range(len(distances)):
            offset = np.power(-1, 1 - channels[i]) * np.array(sonar_offset)
            matches.append(ImageHit(
                pose=Pose(
                    timestamp=pose.timestamp,  # Use the optical image timestamp
                    x=sss_pose.x, y=sss_pose.y, z=sss_pose.z,
                    qw=sss_pose.qw, qx=sss_pose.qx, qy=sss_pose.qy, qz=sss_pose.qz
                ).translate(offset),
                u=o_inters[i, 0],
                v=o_inters[i, 1],
                p_x=p_inters[i, 0],
                p_y=p_inters[i, 1],
                p_z=p_inters[i, 2],
                ping_idx=sss.ping_idx,
                bin_idx=bins[i],
                distance=distances[i],
                incidence_angle=incidence_angles[i]
            ))

    return matches
