import cv2
import numpy as np

import csv
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from .dataset import Dataset, VertexHit


class Decomposition:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.waterfall = np.load(dataset.sonar_file)["data"]
        self.vertex_hits = sorted(self.dataset.vertices_dir.glob("*.csv"))

    def get_vertex_hits(self, filepath: Path) -> List[VertexHit]:
        if not filepath.exists():
            return []

        hits = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                hits.append(VertexHit.from_dict(row))

        return hits

    def get_incidence_angle_map(self) -> Tuple[np.ndarray, np.ndarray]:
        angles = np.zeros_like(self.waterfall, dtype=np.float32)
        counts = np.zeros_like(self.waterfall)

        for v_hit in tqdm(self.vertex_hits):
            for vertex in self.get_vertex_hits(v_hit):
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
        col_means = np.divide(col_sums, col_counts, out=np.zeros_like(col_sums), where=col_counts > 0)

        _, cols = np.where(is_valid)
        reflectivity[is_valid] /= col_means[cols]

        return angles, reflectivity

    def process_decomposition(self) -> None:
        angles, reflectivity = self.get_decomposition()

        np.savez_compressed(self.dataset.sonar_angles, data=angles)
        np.savez_compressed(self.dataset.sonar_reflectivity, data=reflectivity)

    def save_reflectivity_image(self, lower: float, upper: float) -> None:
        reflectivity = np.load(self.dataset.sonar_reflectivity)["data"]

        reflectivity = np.clip(
            reflectivity,
            np.percentile(reflectivity, lower),
            np.percentile(reflectivity, upper)
        )

        reflectivity -= np.min(reflectivity)
        reflectivity /= np.max(reflectivity)
        reflectivity = (reflectivity * 255).astype(np.uint8)

        # Flip to match PNG outputted from XTF orientation
        reflectivity = cv2.flip(reflectivity, 0)

        cv2.imwrite(
            str(self.dataset.reflectivity_png),
            cv2.applyColorMap(reflectivity, cv2.COLORMAP_HSV)
        )
