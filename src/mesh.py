import open3d as o3d
import numpy as np
from typing import List

from .dataset import Dataset, Pose
from .photogrammetry import Photogrammetry


class Mesh:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        photogram = Photogrammetry(self.dataset)
        self.mesh = o3d.io.read_triangle_mesh(str(photogram.mesh_ply))

        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
        self.centroids = (
            self.vertices[self.triangle_indices[0]] +\
            self.vertices[self.triangle_indices[1]] +\
            self.vertices[self.triangle_indices[2]]
        ) / 3.0

    def get_triangles_that_intersect_plane(self,
        plane_pos: np.ndarray,
        plane_normal: np.ndarray,
        thickness: float = 0.01
    ) -> np.ndarray:
        """
        Returns the indices of which triangles intersect the plane.

        If a triangle is within half the thickness of the plane, it is included.
        """
        # shortest distance to the plane
        distance = np.dot(plane_normal, self.centroids - plane_pos)
        return np.flatnonzero(np.abs(distance) <= thickness / 2.0)