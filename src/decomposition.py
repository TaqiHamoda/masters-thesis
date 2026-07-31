import cv2
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata

import pymeshlab
import imageio.v3 as iio
from plyfile import PlyData, PlyElement

from typing import Tuple
from tqdm import tqdm

from .dataset import Dataset, VertexHit


def rasterize_texture(
    uv_matrix: np.ndarray,          # (N, 2) normalized UVs per vertex
    faces: np.ndarray,              # (F, 3) triangle indices
    vertex_data: np.ndarray,        # (N,) float32 values
    tex_size: int
) -> np.ndarray:
    """
    Interpolates continuous vertex float values onto a 2D UV grid buffer (Rasterization).
    """

    # Convert wedge UVs to per-vertex UVs
    uvs = np.zeros((vertex_data.shape[0], 2), dtype=np.float32)
    flat_faces = faces.flatten()
    uvs[flat_faces] = uv_matrix

    # Create 2D pixel grid [0, 1]
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, tex_size),
        np.linspace(0, 1, tex_size)
    )

    # Image Y-axis origin is top-left, so we flip V coordinate (1.0 - V)
    grid_y_flipped = 1.0 - grid_y

    # Single channel scalar field (e.g., Reflectivity)
    texture = griddata(
        uvs,
        vertex_data,
        (grid_x, grid_y_flipped),
        method='linear',
        fill_value=0.0
    )
    return texture.astype(np.float32)


def encode_mesh(mesh_path, faces, vertices, normals, colors, uvs, quality):
    # Define structured array for vertices including 'quality'
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),           # XYZ coordinates
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),        # Normals
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # Colors
        ('quality', 'f4')                                # Reflectivity
    ]

    # Extract and populate vertex data
    vertex_data = np.empty(vertices.shape[0], dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['nx'] = normals[:, 0]
    vertex_data['ny'] = normals[:, 1]
    vertex_data['nz'] = normals[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    vertex_data['quality'] = quality

    ply_vertex = PlyElement.describe(vertex_data, 'vertex')
    ply_elements = [ply_vertex]

    # Open3D's triangle_uvs shape is (N_faces * 3, 2).
    # We reshape it to (N_faces, 6) so each face holds [u0, v0, u1, v1, u2, v2].
    formatted_uvs = uvs.reshape(-1, 6)

    # Extract and format faces
    face_dtype = [
        ('vertex_indices', 'i4', (3,)),
        ('texcoord', 'f4', (6,))
    ]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces
    face_data['texcoord'] = formatted_uvs

    ply_face = PlyElement.describe(face_data, 'face')
    ply_elements.append(ply_face)

    PlyData(ply_elements, text=False).write(mesh_path)


class Decomposition:
    def __init__(self, dataset: Dataset, lower: float, upper: float):
        self.dataset = dataset
        self.lower = lower
        self.upper = upper

        self.waterfall = np.load(dataset.sonar_file)["data"]
        self.vertex_hits = sorted(self.dataset.vertex_matches_dir.glob("*.csv"))

    def get_incidence_angle_map(self) -> Tuple[np.ndarray, np.ndarray]:
        angles = np.zeros_like(self.waterfall, dtype=np.float32)
        counts = np.zeros_like(self.waterfall)

        for v_hit in tqdm(self.vertex_hits):
            for vertex in VertexHit.from_csv(v_hit):
                if vertex.hit.ping_idx >= angles.shape[0] or vertex.hit.bin_idx >= angles.shape[1]:
                    continue

                angles[vertex.hit.ping_idx, vertex.hit.bin_idx] += vertex.hit.incidence_angle
                counts[vertex.hit.ping_idx, vertex.hit.bin_idx] += 1

        is_valid = counts > 0
        angles[is_valid] /= counts[is_valid]

        return angles, is_valid

    def get_decomposition(self, eps=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the incidence angle map and reflectivity map based on the vertex hits."""
        angles, is_valid = self.get_incidence_angle_map()
        reflectivity = np.zeros_like(self.waterfall, dtype=np.float32)

        # Factor out incidence angle to get reflectivity
        cos_angle = np.clip(np.cos(angles[is_valid]), 0, 1) + eps
        reflectivity[is_valid] = self.waterfall[is_valid] / cos_angle

        # Normalize by mean reflectivity per column to factor out range dependence
        col_sums = np.sum(reflectivity, axis=0)
        col_counts = np.sum(is_valid, axis=0)
        prop_loss = np.divide(col_sums, col_counts, out=np.ones_like(col_sums), where=col_counts > 0)

        reflectivity /= prop_loss

        angles[~is_valid] = np.nan
        reflectivity[~is_valid] = np.nan

        return angles, prop_loss, reflectivity

    def process_decomposition(self) -> None:
        angles, prop_loss, reflectivity = self.get_decomposition()

        np.savez_compressed(self.dataset.sonar_angles, data=angles)
        np.savez_compressed(self.dataset.sonar_loss, data=prop_loss)
        np.savez_compressed(self.dataset.sonar_reflectivity, data=reflectivity)

    def print_stats(self) -> None:
        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]
        data = reflectivity[~np.isnan(reflectivity)]
        stats = {
            "Min": np.min(data),
            "1st %": np.percentile(data, 1),
            "25th %": np.percentile(data, 25),
            "Median": np.median(data),
            "75th %": np.percentile(data, 75),
            "99th %": np.percentile(data, 99),
            "Max": np.max(data),
            "Mean": np.mean(data),
            "Std Dev": np.std(data)
        }

        print("="*35)
        print(f"{'Reflectivity Metric':<18} | {'Value':>12}")
        print("-" * 35)

        for metric, value in stats.items():
            # Use :.4f for precision, or :.2e if you expect very tiny/huge numbers
            print(f"{metric:<18} | {value:>12.4f}")

        print("="*35)

    def save_reflectivity_image(self) -> None:
        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]

        is_valid = ~np.isnan(reflectivity)
        reflectivity[is_valid] = np.clip(
            reflectivity[is_valid],
            np.percentile(reflectivity[is_valid], self.lower),
            np.percentile(reflectivity[is_valid], self.upper)
        )

        # Normalize reflectivity values
        reflectivity -= np.min(reflectivity[is_valid])
        reflectivity /= np.max(reflectivity[is_valid])
        reflectivity[~is_valid] = 0

        reflectivity = (255 * reflectivity).astype(np.uint8)
        reflectivity = cv2.applyColorMap(reflectivity, cv2.COLORMAP_PARULA)
        reflectivity[~is_valid] = (0, 0, 0)  # Set Invalid to black

        # Flip to match PNG outputted from XTF orientation
        reflectivity = cv2.flip(reflectivity, 0)
        cv2.imwrite(
            str(self.dataset.reflectivity_png),
            reflectivity
        )

        # Overlay onto Sonar image to compare results
        sonar = cv2.imread(str(self.dataset.sonar_png))
        cv2.imwrite(
            str(self.dataset.overlay_png),
            cv2.addWeighted(sonar, 0.5, reflectivity, 0.5, 0)
        )

    def save_reflectivity_mesh(self,
        slant_sigma: float,
        angle_sigma: float,
        angle_center: float,
        face_num: int,
    ):
        # Use Guassian Decay for the weighting function
        w_func = lambda x, sigma: np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))

        # Docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
        mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        colors = np.clip(255 * np.asarray(mesh.vertex_colors), 0, 255).astype(np.uint8)
        faces = np.asarray(mesh.triangles)
        uvs = np.asarray(mesh.triangle_uvs)

        n = vertices.shape[0]
        v_weights = np.zeros((n,), dtype=np.float32)
        v_reflectivity = np.zeros((n,), dtype=np.float32)

        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]

        # Clip reflectivity values to remove outliers
        is_valid = ~np.isnan(reflectivity)
        reflectivity[~is_valid] = 0
        reflectivity[is_valid] = np.clip(
            reflectivity[is_valid],
            np.percentile(reflectivity[is_valid], self.lower),
            np.percentile(reflectivity[is_valid], self.upper)
        )

        for v_hit in tqdm(list(self.dataset.vertex_matches_dir.glob("*.csv"))):
            for vertex in VertexHit.from_csv(v_hit):
                if vertex.hit.ping_idx >= reflectivity.shape[0] or vertex.hit.bin_idx >= reflectivity.shape[1]:
                    continue

                slant_weight = w_func(vertex.hit.distance, slant_sigma)
                angle_weight = w_func(vertex.hit.incidence_angle - angle_center, angle_sigma)
                weight = slant_weight * angle_weight

                v_weights[vertex.vertex_idx] += weight
                v_reflectivity[vertex.vertex_idx] += weight * reflectivity[vertex.hit.ping_idx, vertex.hit.bin_idx]

        valid_mask = v_weights > 0
        v_reflectivity[valid_mask] /= v_weights[valid_mask]
        np.savez(self.dataset.reflectivity_vertices, data=v_reflectivity)

        encode_mesh(
            str(self.dataset.mesh_ply),
            faces, vertices, normals, colors, uvs, v_reflectivity
        )

        tex_size = cv2.imread(str(self.dataset.colors_texture)).shape[0]
        iio.imwrite(str(self.dataset.reflectivity_texture),
            rasterize_texture(
                uvs, faces, v_reflectivity, tex_size=tex_size
            )
        )

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.dataset.mesh_ply))

        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing_decimation_quadric_edge_collapse_with_texture
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse_with_texture',
            targetfacenum=face_num,
            preservenormal=True,
            preservetopology=True
        )

        ms.save_current_mesh(
            str(self.dataset.output_ply),
            save_textures=True,
            save_vertex_normal=True,
        )
