import open3d as o3d
import numpy as np
from typing import List

from ..dataset import Dataset, Plane
from ..photogrammetry import Photogrammetry


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

    def filter_occluded_vertices(self,
        plane: Plane,
        epsilon: float = 1e-4
    ) -> np.ndarray:
        """
        Raytraces from triangle centroids towards the transducer to check for collisions.
        Returns the indices of triangles that have a clear line-of-sight.
        """
        vertex_indices = plane.get_intersections(self.vertices)

        if len(vertex_indices) == 0:
            return np.array([])

        # 3. Calculate vectors, distances, and normalized directions to the transducer
        origins = self.centroids[vertex_indices]
        transducer_pos = np.array([plane.x, plane.y, plane.z])

        vectors_to_transducer = transducer_pos - origins
        distances = np.linalg.norm(vectors_to_transducer, axis=1)

        # Handle potential division by zero if a centroid is exactly at the transducer
        with np.errstate(divide='ignore', invalid='ignore'):
            directions = vectors_to_transducer / distances[:, np.newaxis]
            directions = np.nan_to_num(directions)

        # 4. Offset the ray origins slightly to avoid self-intersection with the starting triangle
        origins_offset = origins + (directions * epsilon)

        # 5. Prepare the rays for Open3D (Format: [ox, oy, oz, dx, dy, dz])
        rays = np.hstack((origins_offset, directions)).astype(np.float32)
        rays_t = o3d.core.Tensor(rays)

        # 6. Cast the rays
        ans = self.scene.cast_rays(rays_t)
        
        # 't_hit' is the distance along the ray until it hits the mesh. 
        # If it doesn't hit anything, it returns infinity (inf).
        hit_distances = ans['t_hit'].numpy()

        # 7. Check for collisions
        # A collision occurs if the ray hits the mesh BEFORE it reaches the transducer.
        # We subtract epsilon from the target distance because we pushed the origin forward.
        target_distances = distances - epsilon
        
        # True if the ray hits a mesh face before reaching the transducer
        is_occluded = hit_distances < target_distances

        # Return only the indices of the triangles that are NOT occluded
        visible_triangles = vertex_indices[~is_occluded]

        return visible_triangles