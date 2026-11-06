import cv2
import numpy as np
from scipy.interpolate import griddata

import pymeshlab
import open3d as o3d
import imageio.v3 as iio
from plyfile import PlyData, PlyElement

from typing import Tuple
from tqdm import tqdm

from .dataset import Dataset, VertexHit
from .bundle_adjustment import BundleAdjustment, collect_garbage


def rasterize_texture(
    uvs: np.ndarray,                # (N, 2) normalized UVs per vertex
    faces: np.ndarray,              # (F, 3) triangle indices
    vertex_data: np.ndarray,        # (N,) float32 values
    tex_size: int
) -> np.ndarray:
    """
    Interpolates continuous vertex float values onto a 2D UV grid buffer (Rasterization).
    """

    # Expand vertex values to match the wedge UVs length (F*3)
    flat_faces = faces.flatten()
    wedge_values = vertex_data[flat_faces]

    # Create 2D pixel grid [0, 1]
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, tex_size),
        np.linspace(0, 1, tex_size)
    )

    # Image Y-axis origin is top-left, so we flip V coordinate (1.0 - V)
    grid_y_flipped = 1.0 - grid_y

    # Single channel scalar field (e.g., Reflectivity)
    texture = griddata(
        uvs,                  # Shape: (F*3, 2)
        wedge_values,               # Shape: (F*3,)
        (grid_x, grid_y_flipped),
        method='linear',
        fill_value=0.0
    )
    return texture.astype(np.float32)


def encode_mesh(mesh_path, faces, vertices, normals, colors, quality, uvs=None):
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

    # Extract and format faces
    if uvs is not None:
        face_dtype = [
            ('vertex_indices', 'i4', (3,)),
            ('texcoord', 'f4', (6,))
        ]
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data['vertex_indices'] = faces

        # Open3D's triangle_uvs shape is (N_faces * 3, 2).
        # We reshape it to (N_faces, 6) so each face holds [u0, v0, u1, v1, u2, v2].
        face_data['texcoord'] = uvs.reshape(-1, 6)
    else:
        face_dtype = [('vertex_indices', 'i4', (3,))]
        face_data = np.empty(len(faces), dtype=face_dtype)
        face_data['vertex_indices'] = faces

    ply_face = PlyElement.describe(face_data, 'face')
    ply_elements.append(ply_face)

    PlyData(ply_elements, text=False).write(mesh_path)


def decode_mesh(mesh_path):
    ply_data = PlyData.read(str(mesh_path))
    vertex_data = ply_data['vertex'].data
    face_data = ply_data['face'].data

    # Extract data
    vertices = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
    colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T.astype(np.uint8)
    faces = np.vstack(face_data['vertex_indices'])
    quality = np.vstack(vertex_data['quality']) if 'quality' in vertex_data.dtype.names else None

    uvs = None
    if 'u' in vertex_data.dtype.names and 'v' in vertex_data.dtype.names:
        vertex_uvs = np.vstack([vertex_data['u'], vertex_data['v']]).T
        # Map per-vertex UVs to Open3D's per-triangle-corner layout
        uvs = vertex_uvs[faces.reshape(-1)]
    elif 's' in vertex_data.dtype.names and 't' in vertex_data.dtype.names:
        vertex_uvs = np.vstack([vertex_data['s'], vertex_data['t']]).T
        uvs = vertex_uvs[faces.reshape(-1)]
    elif 'texcoord' in face_data.dtype.names:
        # Reshape array of shape (num_faces, 6) -> (num_faces * 3, 2)
        uvs = np.vstack(face_data['texcoord']).reshape(-1, 2)
    elif 'texture_coords' in face_data.dtype.names:
        uvs = np.vstack(face_data['texture_coords']).reshape(-1, 2)
    
    return vertices, normals, colors, faces, quality, uvs


class Decomposition:
    def __init__(self, dataset: Dataset, lower: float, upper: float, ba_cfg: dict = None):
        self.dataset = dataset
        self.lower = lower
        self.upper = upper
        self.ba_cfg = ba_cfg if ba_cfg is not None else {}

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
        if self.ba_cfg.get("enabled", False):
            self.process_decomposition_ba()
        else:
            angles, prop_loss, reflectivity = self.get_decomposition()

            np.savez_compressed(self.dataset.sonar_angles, data=angles)
            np.savez_compressed(self.dataset.sonar_loss, data=prop_loss)
            np.savez_compressed(self.dataset.sonar_reflectivity, data=reflectivity)

    def process_decomposition_ba(self) -> None:
        """Replaces the closed-form decomposition with a bundle adjustment solve."""

        print("Performing Bundle Adjustment for Reflectivity Map...")

        cfg = self.ba_cfg
        ba = BundleAdjustment(
            self.dataset,
            ang_samples=cfg.get("ang_samples", 360),
            max_angle=cfg.get("max_angle", np.pi / 2),
            absorption_coeff=cfg.get("absorption_coeff", 0.1),
            max_iterations=cfg.get("max_iterations", 100),
            loss_delta=cfg.get("loss_delta", 1000.0),
            gauge_weight=cfg.get("gauge_weight", 1.0),
            num_threads=cfg.get("num_threads", 6),
            verbose=cfg.get("verbose", True),
        )

        reflectivity, beam_port, beam_stbd, absorption_coeff, summary = ba.solve()

        np.savez_compressed(
            self.dataset.sonar_ba,
            reflectivity_vertices=reflectivity,
            beam_port=beam_port,
            beam_stbd=beam_stbd,
            absorption_coeff=absorption_coeff,
        )

        # Snapshot only the per-observation values needed to build the maps,
        # then release the (potentially mesh-sized) solver outputs
        obs_values = reflectivity[ba.obs_vertex]
        del reflectivity, beam_port, beam_stbd

        # Incidence angles from the vertex hits (invalid cells stay NaN)
        angles, is_valid = self.get_incidence_angle_map()

        rows, cols = self.waterfall.shape

        # Average the per-vertex reflectivity over each (ping, bin) cell
        sums = np.zeros((rows, cols), dtype=np.float64)
        counts = np.zeros((rows, cols), dtype=np.int32)
        np.add.at(sums, (ba.obs_ping_idx, ba.obs_bin_idx), obs_values)
        del obs_values
        np.add.at(counts, (ba.obs_ping_idx, ba.obs_bin_idx), 1)

        # Per-column propagation loss from the refined absorption coefficient
        col_sums = np.bincount(ba.obs_bin_idx, weights=ba.obs_range, minlength=cols)
        col_counts = np.bincount(ba.obs_bin_idx, minlength=cols)
        prop_loss = np.ones(cols, dtype=np.float64)
        observed_cols = col_counts > 0
        col_slant = col_sums[observed_cols] / col_counts[observed_cols]
        prop_loss[observed_cols] = 10.0 ** (-2.0 * absorption_coeff * col_slant / 10.0)
        del col_sums, col_counts, col_slant, observed_cols

        # The bundle adjustment object (and its remaining observation arrays)
        # are no longer needed once the column statistics are computed
        del ba
        collect_garbage()

        reflectivity_map = np.full((rows, cols), np.nan, dtype=np.float32)
        hit_cells = counts > 0
        reflectivity_map[hit_cells] = (sums[hit_cells] / counts[hit_cells]).astype(np.float32)
        del sums, counts, hit_cells

        # Cells without angle observations stay NaN in the angles map
        angles[~is_valid] = np.nan

        np.savez_compressed(self.dataset.sonar_angles, data=angles)
        np.savez_compressed(self.dataset.sonar_loss, data=prop_loss)
        np.savez_compressed(self.dataset.sonar_reflectivity, data=reflectivity_map)
        del angles, is_valid, reflectivity_map, prop_loss
        collect_garbage()

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
    ):
        # Use Guassian Decay for the weighting function
        w_func = lambda x, sigma: np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))

        # Docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
        mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        colors = np.clip(255 * np.asarray(mesh.vertex_colors), 0, 255).astype(np.uint8)
        faces = np.asarray(mesh.triangles)

        n = vertices.shape[0]
        v_weights = np.zeros((n,), dtype=np.float32)
        v_reflectivity = np.zeros((n,), dtype=np.float32)

        if self.ba_cfg.get("enabled", False) and self.dataset.sonar_ba.exists():
            # The bundle adjustment solved reflectivity per vertex directly,
            # so no Gaussian-weighted splatting is needed.
            ba_reflectivity = np.load(self.dataset.sonar_ba)["reflectivity_vertices"]
            keep = min(n, ba_reflectivity.size)
            v_reflectivity[:keep] = ba_reflectivity[:keep].astype(np.float32)

            # Clip reflectivity values to remove outliers
            is_observed = v_reflectivity > 0
            v_reflectivity[is_observed] = np.clip(
                v_reflectivity[is_observed],
                np.percentile(v_reflectivity[is_observed], self.lower),
                np.percentile(v_reflectivity[is_observed], self.upper)
            )

            np.savez(self.dataset.reflectivity_vertices, data=v_reflectivity)

            encode_mesh(
                str(self.dataset.mesh_ply),
                faces, vertices, normals, colors, v_reflectivity
            )
            return

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
            faces, vertices, normals, colors, v_reflectivity
        )

    def export_textures(self, tex_size: int, face_num: int):
        # Load the newly generated mesh which has faces
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.dataset.mesh_ply))

        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing_decimation_quadric_edge_collapse_with_texture
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse',
            targetfacenum=face_num,
            preservenormal=True,
            preservetopology=True
        )

        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html#compute_texcoord_parametrization_triangle_trivial_per_wedge
        ms.apply_filter(
            'compute_texcoord_parametrization_triangle_trivial_per_wedge',
            textdim=tex_size,
            border=1
        )

        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html#transfer_attributes_to_texture_per_vertex
        for attribute, name in [
            (1, self.dataset.normals_texture.name),
            (0, self.dataset.colors_texture.name),
        ]:
            ms.apply_filter(
                'transfer_attributes_to_texture_per_vertex',
                sourcemesh=0,
                targetmesh=0,
                attributeenum=attribute,
                textname=name,
                textw=tex_size,
                texth=tex_size
            )

            # https://pymeshlab.readthedocs.io/en/latest/io_format_list.html#save-mesh-parameters
            ms.save_current_mesh(
                str(self.dataset.output_ply),
                save_textures=True,
                save_vertex_normal=True,
            )

        vertices, normals, colors, faces, quality, uvs = decode_mesh(self.dataset.output_ply)

        # Source: https://imageio.readthedocs.io/en/v2.10.5/reference/_backends/imageio.plugins.freeimage.html#module-imageio.plugins.freeimage
        # Source: https://cfis.github.io/free-image-ruby/classes/FreeImage/AbstractSource/Encoder.html
        iio.imwrite(str(self.dataset.reflectivity_texture),
            rasterize_texture(
                uvs, faces, quality, tex_size=tex_size
            ).astype(np.float32),
            flags=1  # Use the FreeImage EXR_FLOAT flag for encoding
        )
