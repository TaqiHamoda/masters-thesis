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
) -> List[ImageHit]:
    matches = []
    for s_ts in sss_poses.keys():
        sss = dataset.sonar[s_ts]
        sss_pose = sss_poses[s_ts]

        is_valid = get_intersections(sss_pose, points)

        dists = get_distances(sss_pose, points[is_valid])
        is_valid[is_valid] = dists <= sss.slant_range

        if not np.any(is_valid):
            continue

        p_inters = points[is_valid]
        o_inters = optical[is_valid]

        distances = get_distances(sss_pose, p_inters)
        channels = get_corresponding_channels(sss_pose, p_inters)
        incidence_angles = get_incidence_angles(sss_pose, p_inters)
        bins = np.round(sss.num_samples * distances / sss.slant_range).astype(int)

        for i in range(len(distances)):
            matches.append(ImageHit(
                pose=pose,
                u=o_inters[i, 0],
                v=o_inters[i, 1],
                p_x=p_inters[i, 0],
                p_y=p_inters[i, 1],
                p_z=p_inters[i, 2],
                ping_idx=sss.ping_idx,
                channel_idx=channels[i],
                bin_idx=bins[i],
                distance=distances[i],
                incidence_angle=incidence_angles[i]
            ))

    return matches
