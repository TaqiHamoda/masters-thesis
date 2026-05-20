import numpy as np
import open3d as o3d
from sklearn import linear_model

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .utils import (
    spatial_median,
    get_image_geometry,
    get_distances,
    get_intersections,
    get_in_body_frame,
    get_corresponding_channels,
    get_incidence_angles,
)
from ..dataset import Dataset, Pose, AcousticHit, ImageHit, VertexHit
from ..photogrammetry import Photogrammetry


class Registration:
    def __init__(self,
        dataset: Dataset,
        sonar_offset: np.ndarray,
        thickness: float,
        n_local: Tuple[float, float, float],
        num_threads: int,
        w_size: int = 10
    ):
        self.dataset = dataset
        self.sonar_offset = sonar_offset
        self.thickness = thickness
        self.n_local = np.array(n_local)
        self.num_threads = num_threads
        self.w_size = w_size

        self.sss_poses = {
            ts: sss.navigation.pose
            for ts, sss in dataset.sonar.items()
        }

        self.reconstruction = Photogrammetry.get_reconstruction(dataset)
        self.img_ids = {
            int(img.name.replace(".jpg", '')): img.image_id
            for img in self.reconstruction.images.values()
        }

        self.first_returns = np.load(self.dataset.first_return)["data"]

        self.mesh = None
        self.scene = None
        self.vertices = None

    def load_mesh(self):
        # The RaycastingScene requires Open3D's Tensor-based mesh, so we convert the legacy mesh
        self.mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

        self.vertices = self.mesh.vertex.positions.numpy()

    def optimize_extrinsics(self):
        if not self.dataset.extrinsics_file.exists():
            data = self.calculate_offsets()
        else:
            data = np.load(self.dataset.extrinsics_file)["data"]

        indices = data[:, 0]
        for ts in tqdm(list(self.sss_poses.keys()), desc="Optimizing Extrinsics"):
            sss = self.dataset.sonar[ts]

            window = (indices > sss.ping_idx - self.w_size) & (indices < sss.ping_idx + self.w_size)
            if not np.any(window):
                continue

            points = data[:, 4][window].reshape(-1, 1)
            z_delta = spatial_median(points)[0]

            extrinsics = np.array((0, 0, z_delta))
            self.sss_poses[ts] = self.sss_poses[ts].translate(extrinsics)

    def get_visible_points(self,
        ts: int,
        points: np.ndarray,
        raytrace: bool = False,
        eps: float = 1e-1
    ) -> np.ndarray:
        """
        Raytraces from triangle centroids towards the transducer to check for collisions.
        Returns the indices of which vertices have a clear, unobstructed line-of-sight.
        """
        pose = self.sss_poses[ts]
        sss = self.dataset.sonar[ts]

        inters = get_intersections(pose, points, thickness=self.thickness, n_local=self.n_local)
        dists = get_distances(pose, points[inters])

        inters[inters] = dists < sss.slant_range
        if not np.any(inters):
            return np.array([])
        elif not raytrace:
            return np.flatnonzero(inters)

        origins = points[inters]
        vs_ned = pose.get_position() - origins
        distances = np.linalg.norm(vs_ned, axis=1, keepdims=True)

        directions = vs_ned / (distances + eps)
        origins_offset = origins + (directions * eps)

        rays = np.hstack((origins_offset, directions)).astype(np.float32)
        rays_t = o3d.core.Tensor(rays)

        ans = self.scene.cast_rays(rays_t)
        hit_distances = ans['t_hit'].numpy()

        # A collision occurs if the ray hits the mesh BEFORE it reaches the transducer.
        # We subtract epsilon from the target distance because we pushed the origin forward.
        inters[inters] = hit_distances >= distances.reshape(-1) - eps
        return np.flatnonzero(inters)
    
    def get_geometry(self, ts: int, points: np.ndarray):
        pose = self.sss_poses[ts]
        sss = self.dataset.sonar[ts]

        channels = get_corresponding_channels(pose, points)
        local_offsets = np.power(-1, 1 - channels.reshape(-1, 1)) * self.sonar_offset
        offsets = (pose.get_rotation_matrix() @ local_offsets.T).T  # Transform offsets to world frame

        distances = get_distances(pose, points - offsets)
        incidence_angles = get_incidence_angles(pose, points - offsets)

        bins = sss.num_samples * distances / sss.slant_range
        bins = sss.num_samples + np.power(-1, 1 - channels) * bins
        bins = np.round(bins).astype(int)

        return channels, local_offsets, offsets, distances, incidence_angles, bins

    def get_hits(self, ts: int, points: np.ndarray) -> List[AcousticHit]:
        pose = self.sss_poses[ts]
        sss = self.dataset.sonar[ts]

        (channels,
        local_offsets,
        offsets,
        distances,
        incidence_angles,
        bins) = self.get_geometry(ts, points)

        return [
            AcousticHit(
                pose=Pose(
                    timestamp=pose.timestamp,
                    x=pose.x + offsets[i, 0], y=pose.y + offsets[i, 1], z=pose.z + offsets[i, 2],
                    qw=pose.qw, qx=pose.qx, qy=pose.qy, qz=pose.qz
                ),
                ping_idx=sss.ping_idx,
                bin_idx=bins[i],
                distance=distances[i],
                incidence_angle=incidence_angles[i],
                offset_x=local_offsets[i, 0],
                offset_y=local_offsets[i, 1],
                offset_z=local_offsets[i, 2],
                p_x=points[i, 0],
                p_y=points[i, 1],
                p_z=points[i, 2],
            )
            for i in range(len(distances))
        ]

    def get_matches(self, ts: int) -> List[ImageHit]:
        points_2d, points_3d = get_image_geometry(self.reconstruction, self.img_ids[ts])

        matches = []
        for s_ts in self.sss_poses.keys():
            is_valid = self.get_visible_points(s_ts, points_3d)
            if len(is_valid) == 0:
                continue

            p_inters = points_3d[is_valid]
            o_inters = points_2d[is_valid]

            hits = self.get_hits(s_ts, p_inters)
            for i in range(len(hits)):
                matches.append(ImageHit(
                    hit=hits[i],
                    u=o_inters[i, 0],
                    v=o_inters[i, 1]
                ))

        return matches

    def get_vertices(self, ts: int) -> List[VertexHit]:
        is_valid = self.get_visible_points(ts, self.vertices, raytrace=True)
        if len(is_valid) == 0:
            return []

        vertices = self.vertices[is_valid]
        hits = self.get_hits(ts, vertices)

        return [
            VertexHit(
                hit=hits[i],
                vertex_idx=is_valid[i]
            )
            for i in range(len(hits))
        ]
    
    def _calculate_offsets(self, ts: int) -> List[Tuple[float]]:
        """
        Returns a 2x5 array where the rows are port and stbd and columns are:
        ping index, first return distance, observed distance, Z-axis distance
        in local frame, and difference in Z-axis between first return and observed.
        """
        pose = self.sss_poses[ts]
        sss = self.dataset.sonar[ts]

        is_valid = self.get_visible_points(ts, self.vertices)
        if len(is_valid) == 0:
            return []

        v_points = self.vertices[is_valid]

        (channels,
        local_offsets,
        offsets,
        distances,
        incidence_angles,
        bins) = self.get_geometry(ts, v_points)

        max_dist = np.max(distances)
        port_idx = np.argmin(distances + max_dist * (channels != 0))
        stbd_idx = np.argmin(distances + max_dist * (channels != 1))

        port_vertex = get_in_body_frame(
            pose.translate(local_offsets[port_idx]),
            v_points[port_idx]
        )
        stbd_vertex = get_in_body_frame(
            pose.translate(local_offsets[stbd_idx]),
            v_points[stbd_idx]
        )

        gt_dists = self.first_returns[sss.ping_idx] * sss.slant_range / sss.num_samples

        port_delta = np.sqrt(np.abs(np.power(gt_dists[0], 2) - np.power(distances[port_idx], 2) + np.power(port_vertex[2], 2)))
        port_delta = port_vertex[2] - port_delta

        stbd_delta = np.sqrt(np.abs(np.power(gt_dists[1], 2) - np.power(distances[stbd_idx], 2) + np.power(stbd_vertex[2], 2)))
        stbd_delta = stbd_vertex[2] - stbd_delta

        return [
            (sss.ping_idx, gt_dists[0], distances[port_idx], port_vertex[2], port_delta),
            (sss.ping_idx, gt_dists[1], distances[stbd_idx], stbd_vertex[2], stbd_delta)
        ]

    def calculate_offsets(self) -> np.ndarray:
        self.load_mesh()
        sonars_ts = list(self.sss_poses.keys())

        data = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(
                executor.map(
                    lambda ts: data.extend(self._calculate_offsets(ts)),
                    sonars_ts
                ),
                total=len(sonars_ts),
                desc="Calculating offsets per ping"
            ))

        data = np.array(data)
        np.savez_compressed(self.dataset.extrinsics_file, data=data)

        return data

    def _save_matches(self, ts: int) -> None:
        matches_file = self.dataset.image_matches_dir / f"{ts}.csv"
        if matches_file.exists():
            return

        matches = self.get_matches(ts)
        if len(matches) == 0:
            return

        ImageHit.to_csv(matches_file, matches)

    def _save_vertices(self, ts: int) -> None:
        vertices_file = self.dataset.vertex_matches_dir / f"{ts}.csv"
        if vertices_file.exists():
            return

        hits = self.get_vertices(ts)
        if len(hits) == 0:
            return

        VertexHit.to_csv(vertices_file, hits)

    def save_matches(self) -> None:
        images_ts = list(self.img_ids.keys())
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(
                executor.map(self._save_matches, images_ts),
                total=len(images_ts),
                desc="Processing optical matches"
            ))

    def save_vertices(self) -> None:
        self.load_mesh()

        sonars_ts = list(self.sss_poses.keys())
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(
                executor.map(self._save_vertices, sonars_ts),
                total=len(sonars_ts),
                desc="Processing vertex hits"
            ))
