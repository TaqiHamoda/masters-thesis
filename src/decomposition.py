import cv2
import numpy as np
import open3d as o3d

from typing import Tuple
from tqdm import tqdm

from .dataset import Dataset, VertexHit


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

    def get_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the incidence angle map and reflectivity map based on the vertex hits."""
        angles, is_valid = self.get_incidence_angle_map()
        reflectivity = np.zeros_like(self.waterfall, dtype=np.float32)

        # Factor out incidence angle to get reflectivity
        reflectivity[is_valid] = self.waterfall[is_valid] / np.cos(angles[is_valid])

        # Normalize by mean reflectivity per column to factor out range dependence
        col_sums = np.sum(reflectivity, axis=0)
        col_counts = np.sum(is_valid, axis=0)
        prop_loss = np.divide(col_sums, col_counts, out=np.ones_like(col_sums), where=col_counts > 0)

        reflectivity /= prop_loss

        return angles, prop_loss, reflectivity

    def process_decomposition(self) -> None:
        angles, prop_loss, reflectivity = self.get_decomposition()

        np.savez_compressed(self.dataset.sonar_angles, data=angles)
        np.savez_compressed(self.dataset.sonar_loss, data=prop_loss)
        np.savez_compressed(self.dataset.sonar_reflectivity, data=reflectivity)

    def print_stats(self) -> None:
        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]
        data = reflectivity[reflectivity > 0]
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

        is_valid = reflectivity > 0
        reflectivity[is_valid] = np.clip(
            reflectivity[is_valid],
            np.percentile(reflectivity[is_valid], self.lower),
            np.percentile(reflectivity[is_valid], self.upper)
        )

        # Use logarithmic scale to amplify variations in reflectivity
        reflectivity = np.log1p(reflectivity)

        # Normalize reflectivity values
        reflectivity -= np.min(reflectivity[is_valid])
        reflectivity /= np.max(reflectivity)
        reflectivity[~is_valid] = 0

        reflectivity = (255 * reflectivity).astype(np.uint8)
        reflectivity = cv2.applyColorMap(reflectivity, cv2.COLORMAP_RAINBOW)
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
        angle_center: float
    ):
        # Use Guassian Decay for the weighting function
        w_func = lambda x, sigma: np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))

        # Docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
        mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        vertices = np.asarray(mesh.vertices)

        n = vertices.shape[0]
        v_weights = np.zeros((n,), dtype=np.float32)
        v_reflectivity = np.zeros((n,), dtype=np.float32)

        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]

        # Clip reflectivity values to remove outliers
        is_valid = reflectivity > 0
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

        # TODO: Use Trimesh instead to save vertex attributes. Open3D doesn't support PLY that well
        # TODO: Save the propogation loss values as well in the ply
        v_reflectivity = np.log1p(v_reflectivity)
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        t_mesh.vertex["reflectivity"] = o3d.core.Tensor(
            v_reflectivity.astype(np.float32), 
            device=o3d.core.Device("CPU:0")
        )
        o3d.t.io.write_triangle_mesh(str(self.dataset.reflectivity_mesh), t_mesh)
