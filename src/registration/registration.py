import numpy as np
import open3d as o3d

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .utils import (
    interpolate_poses,
    get_image_geometry,
    get_distances,
    get_intersections,
    get_corresponding_channels,
    get_incidence_angles,
)
from ..dataset import Dataset, Pose, SideScanSonar, AcousticHit, ImageHit, VertexHit
from ..photogrammetry import Photogrammetry


class Registration:
    def __init__(self,
        dataset: Dataset,
        sonar_offset: np.ndarray,
        thickness: float,
        n_local: Tuple[float, float, float],
        num_threads: int
    ):
        self.dataset = dataset
        self.sonar_offset = sonar_offset
        self.thickness = thickness
        self.n_local = np.array(n_local)
        self.num_threads = num_threads

        self.reconstruction = Photogrammetry.get_reconstruction(dataset)
        self.cam_poses, self.sss_poses = interpolate_poses(dataset, self.reconstruction)

        self.img_ids = {
            int(img.name.replace(".jpg", '')): img.image_id
            for img in self.reconstruction.images.values()
        }

        # The RaycastingScene requires Open3D's Tensor-based mesh, so we convert the legacy mesh
        self.mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

        self.vertices = self.mesh.vertex.positions.numpy()

    def get_visible_points(self,
        sss: SideScanSonar,
        points: np.ndarray,
        raytrace: bool = False,
        eps: float = 1e-1
    ) -> np.ndarray:
        """
        Raytraces from triangle centroids towards the transducer to check for collisions.
        Returns the indices of which vertices have a clear, unobstructed line-of-sight.
        """
        pose = sss.navigation.pose

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

        if o3d.core.cuda.is_available():
            rays_t = rays_t.cuda(0)

        ans = self.scene.cast_rays(rays_t)
        hit_distances = ans['t_hit'].numpy()

        # A collision occurs if the ray hits the mesh BEFORE it reaches the transducer.
        # We subtract epsilon from the target distance because we pushed the origin forward.
        inters[inters] = hit_distances >= distances.reshape(-1) - eps
        return np.flatnonzero(inters)

    def get_hits(self, sss: SideScanSonar, points: np.ndarray) -> List[AcousticHit]:
        pose = self.sss_poses[sss.navigation.pose.timestamp]

        channels = get_corresponding_channels(pose, points)
        offsets = np.power(-1, 1 - channels.reshape(-1, 1)) * self.sonar_offset
        offsets = (pose.get_rotation_matrix() @ offsets.T).T  # Transform offsets to world frame

        distances = get_distances(pose, points - offsets)
        incidence_angles = get_incidence_angles(pose, points - offsets)

        bins = sss.num_samples * distances / sss.slant_range
        bins = sss.num_samples + np.power(-1, 1 - channels) * bins
        bins = np.round(bins).astype(int)

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
                incidence_angle=incidence_angles[i]
            )
            for i in range(len(distances))
        ]

    def get_matches(self, pose: Pose) -> List[ImageHit]:
        points_2d, points_3d = get_image_geometry(self.reconstruction, self.img_ids[pose.timestamp])

        matches = []
        for s_ts in self.sss_poses.keys():
            sss = self.dataset.sonar[s_ts]

            is_valid = self.get_visible_points(sss, points_3d)
            if len(is_valid) == 0:
                continue

            p_inters = points_3d[is_valid]
            o_inters = points_2d[is_valid]

            hits = self.get_hits(sss, p_inters)
            for i in range(len(hits)):
                hits[i].pose.timestamp = pose.timestamp  # Use the optical image timestamp
                matches.append(ImageHit(
                    hit=hits[i],
                    u=o_inters[i, 0],
                    v=o_inters[i, 1],
                    p_x=p_inters[i, 0],
                    p_y=p_inters[i, 1],
                    p_z=p_inters[i, 2],
                ))

        return matches

    def get_vertices(self, sss: SideScanSonar) -> List[VertexHit]:
        is_valid = self.get_visible_points(sss, self.vertices, raytrace=True)

        if len(is_valid) == 0:
            return []

        vertices = self.vertices[is_valid]
        hits = self.get_hits(sss, vertices)

        return [
            VertexHit(
                hit=hits[i],
                vertex_idx=is_valid[i],
                p_x=vertices[i, 0],
                p_y=vertices[i, 1],
                p_z=vertices[i, 2],
            )
            for i in range(len(hits))
        ]

    def _save_matches(self, img_name: str) -> None:
        ts = int(img_name.replace(".jpg", ''))

        matches_file = self.dataset.matches_dir / f"{ts}.csv"
        if matches_file.exists():
            return

        pose = self.cam_poses[ts]
        matches = self.get_matches(pose)
        if len(matches) == 0:
            return

        Dataset.write_data(matches_file, matches)

    def _save_vertices(self, s_ts: int) -> None:
        vertices_file = self.dataset.vertices_dir / f"{s_ts}.csv"
        if vertices_file.exists():
            return

        sss = self.dataset.sonar[s_ts]
        hits = self.get_vertices(sss)
        if len(hits) == 0:
            return

        Dataset.write_data(vertices_file, hits)

    def save_matches(self) -> None:
        images = list(self.reconstruction.images.values())

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            list(tqdm(
                executor.map(
                    lambda img: self._save_matches(img.name),
                    images
                ),
                total=len(images),
                desc="Processing optical matches"
            ))

    def save_vertices(self) -> None:
        sonars = list(self.sss_poses.keys())

        for ts in tqdm(sonars, desc="Processing vertex hits"):
            self._save_vertices(ts)
