import numpy as np
from typing import List

from .utils import interpolate_poses, get_image_geometry, get_distances, get_intersections, get_corresponding_channels, get_incidence_angles
from ..dataset import Dataset, ImageHit


def process_optical_sidescan_matches(dataset: Dataset, img_name: str) -> List[ImageHit]:
    _, sss_poses = interpolate_poses(dataset)

    optical, points = get_image_geometry(dataset, img_name)
    ts = int(img_name.replace(".jpg", ''))

    matches = []
    for s_ts in sss_poses.keys():
        sss = dataset.sonar[s_ts]
        pose = sss_poses[s_ts]

        inters = get_intersections(pose, points)
        if not np.any(inters):
            continue

        p_inters = points[inters]
        o_inters = optical[inters]

        distances = get_distances(pose, p_inters)
        channels = get_corresponding_channels(pose, p_inters)
        incidence_angles = get_incidence_angles(pose, p_inters)
        bins = np.round(sss.num_samples * distances / sss.slant_range).astype(int)

        for i in range(len(distances)):
            matches.append(ImageHit(
                timestamp=ts,
                u_idx=o_inters[i, 0],
                v_idx=o_inters[i, 1],
                ping_idx=sss.ping_idx,
                bin_idx=bins[i],
                channel_idx=channels[i],
                distance=distances[i],
                incidence_angle=incidence_angles[i]
            ))

    return matches
