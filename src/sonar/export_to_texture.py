import cv2
import numpy as np
import open3d as o3d
from skimage.draw import polygon

from tqdm import tqdm

from ..dataset import Dataset, VertexHit


def encode_float_to_bgra(float_array):
    """
    Encodes an array of float32 into an NxM map of BGRA uint8s.
    """
    # View the float32 as a uint32 to get raw bits
    bits = np.array(float_array, dtype=np.float32).view(np.uint32)

    # Extract 4 bytes using bit shifting
    r = (bits >> 24) & 0xFF
    g = (bits >> 16) & 0xFF
    b = (bits >> 8) & 0xFF
    a = bits & 0xFF

    # Stack into an BGRA array
    return np.stack([b, g, r, a], axis=-1).astype(np.uint8)


def gaussian_decay(x: float, sigma: float):
    return np.exp(-np.power(x, 2) / (2 * np.power(sigma, 2)))


def export_to_texture(
    dataset: Dataset,
    slant_sigma: float,
    angle_sigma: float,
    angle_center: float
):
    # Docs: https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
    mesh = o3d.io.read_triangle_mesh(str(dataset.mesh_ply))

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    texture = (0 * cv2.imread(str(dataset.reflectivity_texture), cv2.IMREAD_GRAYSCALE)).astype(np.float32)
    h, w = texture.shape

    # Convert coordinates from normalized space to pixel space
    uvs = np.asarray(mesh.triangle_uvs) % 1.0
    uvs[:, 0] = uvs[:, 0] * (w - 1)
    uvs[:, 1] = (1 - uvs[:, 1]) * (h - 1)

    n = vertices.shape[0]
    v_weights = np.zeros((n, 1), dtype=np.float32)
    v_reflectivity = np.zeros_like((n, 1), dtype=np.float32)

    reflectivity = np.load(dataset.sonar_reflectivity)["data"]
    for v_hit in tqdm(dataset.vertices_dir.glob("*.csv")):
        for vertex in VertexHit.from_csv(v_hit).values():
            if vertex.hit.ping_idx >= reflectivity.shape[0] or vertex.hit.bin_idx >= reflectivity.shape[1]:
                continue

            slant_weight = gaussian_decay(vertex.hit.distance, slant_sigma)
            angle_weight = gaussian_decay(vertex.hit.incidence_angle - angle_center, angle_sigma)
            weight = slant_weight * angle_weight

            v_weights[vertex.vertex_idx] += weight
            v_reflectivity[vertex.vertex_idx] += weight * reflectivity[vertex.hit.ping_idx, vertex.hit.bin_idx]

    v_reflectivity[v_weights > 0] /= v_weights[v_weights > 0]
    np.savez(dataset.reflectivity_vertices, data=v_reflectivity)

    for t_idx in range(triangles.shape[0]):
        t_ref = np.mean(v_reflectivity[triangles[t_idx]])
        t_uvs = uvs[3 * t_idx : 3 * (t_idx + 1)]

        rr, cc = polygon(t_uvs[:, 1], t_uvs[:, 0], texture.shape)
        texture[rr, cc] = t_ref

    cv2.imwrite(
        str(dataset.reflectivity_texture),
        encode_float_to_bgra(texture)
    )
