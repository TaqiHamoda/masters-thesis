import open3d as o3d
import numpy as np
from typing import List

from ..dataset import Dataset, Pose, SideScanSonar, VertexHit
from ..photogrammetry import Photogrammetry
from .utils import get_distances, get_intersections, get_corresponding_channels, get_incidence_angles


class ReflectivityMesh:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        photogram = Photogrammetry(self.dataset)

        # The RaycastingScene requires Open3D's Tensor-based mesh, so we convert the legacy mesh
        self.mesh = o3d.io.read_triangle_mesh(str(photogram.mesh_ply))
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)

        self.vertices = np.asarray(self.mesh.vertices)

    def get_visible_vertices(self,
        sss: SideScanSonar,
        eps: float = 1e-4
    ) -> np.ndarray:
        """
        Raytraces from triangle centroids towards the transducer to check for collisions.
        Returns the indices of which vertices have a clear, unobstructed line-of-sight.
        """
        pose = sss.navigation.pose

        inters = get_intersections(pose, self.vertices)
        dists = get_distances(pose, self.vertices[inters])

        inters[inters] = dists < sss.slant_range
        if not np.any(inters):
            return np.array([])

        origins = self.vertices[inters]
        vs_ned = pose.get_position() - origins
        distances = np.linalg.norm(vs_ned, axis=1)

        directions = vs_ned / (distances + eps)
        origins_offset = origins + (directions * eps)

        rays = np.hstack((origins_offset, directions)).astype(np.float32)
        rays_t = o3d.core.Tensor(rays)

        ans = self.scene.cast_rays(rays_t)
        hit_distances = ans['t_hit'].numpy()

        # A collision occurs if the ray hits the mesh BEFORE it reaches the transducer.
        # We subtract epsilon from the target distance because we pushed the origin forward.
        inters[inters] = hit_distances < distances - eps
        return np.flatnonzero(inters)

    def process_sidescan_hits(
        self,
        timestamp: int,
        pose: Pose,
    ) -> List[VertexHit]:
        """
        Parameters:
            timestamp: sonar ping timestamp
            pose: sonar ping pose
        """
        sss = self.dataset.sonar[timestamp]

        is_valid = self.get_visible_vertices(sss)
        if not np.any(is_valid):
            return []

        vertices = self.vertices[is_valid]
        distances = get_distances(pose, vertices)
        channels = get_corresponding_channels(pose, vertices)
        incidence_angles = get_incidence_angles(pose, vertices)
        bins = np.round(sss.num_samples * distances / sss.slant_range).astype(int)

        hits = []
        for i, v_id in enumerate(is_valid):
            hits.append(VertexHit(
                pose=pose,
                ping_idx=sss.ping_idx,
                channel_idx=channels[i],
                bin_idx=bins[i],
                vertex_idx=v_id,
                v_x=vertices[i, 0],
                v_y=vertices[i, 1],
                v_z=vertices[i, 2],
                distance=distances[i],
                incidence_angle=incidence_angles[i]
            ))

        return hits
