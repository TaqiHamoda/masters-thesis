import cv2
import numpy as np
import open3d as o3d

import viser

import csv
import time
from typing import List

from ..dataset import Dataset, VertexHit


class VertexVisualizer:
    def __init__(self, dataset: Dataset, patch_size: int = 1000):
        """
        Initializes the Viser-based visualization tool.
        """
        self.camera_pos = np.array((50.31622641, 5.14074192, -109.29152408))
        self.camera_wxyz = np.array((0.999994465, -4.99997233e-07, -1.66354234e-09, 0.00332708463))

        self.dataset = dataset
        self.patch_size = patch_size

        self.current_scan_idx = 0
        self.current_vertex_idx = 0

        self.scans = sorted(dataset.vertices_dir.glob("*.csv"))
        self.vertices: List[VertexHit] = []

        # Start Viser server
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible", dark_mode=True)

        # Load SSS waterfall (convert to RGB so we can draw a red dot on it)
        self.sss_image = cv2.imread(str(dataset.sonar_png), cv2.IMREAD_COLOR_RGB)
        self.sss_image = cv2.flip(self.sss_image, 0)

        # Pre-declare layout handles
        self.sss_patch_marked = None
        self.target_3d = None
        self.auv_pose = None

        # Load Colmap Point Cloud
        self._load_point_cloud()

        # Build GUI Controls
        self._build_gui()

        # Load initial data
        self.set_scan()
        self.update_view()

    def _load_point_cloud(self):
        """Loads and displays the sparse point cloud in the 3D scene."""
        mesh = o3d.io.read_triangle_mesh(str(self.dataset.mesh_ply))
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        points = mesh.vertex.positions.numpy()
        colors = mesh.vertex.colors.numpy()

        # Center the point cloud roughly around the origin for easier viewing
        self.center_offset = points.mean(axis=0)
        points -= self.center_offset

        self.server.scene.add_point_cloud(
            name="/colmap/pcd",
            points=points,
            colors=colors,
            point_size=0.1,
        )

    def _build_gui(self):
        """Creates the side-panel buttons and info displays."""
        self.gui_info = self.server.gui.add_markdown("Loading data...")

        # View Controls
        folder_view = self.server.gui.add_folder("View Controls")
        with folder_view:
            btn_reset_view = self.server.gui.add_button("Reset View")

        # Scan Controls
        folder_scan = self.server.gui.add_folder("Side-Scan Navigation")
        with folder_scan:
            btn_prev_scan_100 = self.server.gui.add_button("-100 Scans")
            btn_prev_scan = self.server.gui.add_button("Prev Scan")
            btn_next_scan = self.server.gui.add_button("Next Scan")
            btn_next_scan_100 = self.server.gui.add_button("+100 Scans")

        # Vertex Controls
        folder_vertex = self.server.gui.add_folder("Vertex Navigation")
        with folder_vertex:
            btn_prev_vertex_100 = self.server.gui.add_button("-100 Vertices")
            btn_prev_vertex = self.server.gui.add_button("Prev Vertex")
            btn_next_vertex = self.server.gui.add_button("Next Vertex")
            btn_next_vertex_100 = self.server.gui.add_button("+100 Vertices")

        # Layout & Display Controls
        folder_layout = self.server.gui.add_folder("Display Settings")
        with folder_layout:
            self.gui_sss_pos = self.server.gui.add_vector3(
                "SSS Pos (x,y,z)", initial_value=(100, 0, -20), step=0.1
            )
            self.gui_sss_scale = self.server.gui.add_slider(
                "SSS Scale", min=0.1, max=100.0, step=0.1, initial_value=75.0
            )

            self.gui_marker_size = self.server.gui.add_slider(
                "3D Marker Size", min=0.01, max=2.0, step=0.01, initial_value=1.0
            )

            self.gui_auv_scale = self.server.gui.add_slider(
                "AUV Axes Scale", min=0.1, max=10.0, step=0.1, initial_value=7.0
            )

        # --- Callbacks: View ---
        @btn_reset_view.on_click
        def _(_) -> None:
            for client in self.server.get_clients().values():
                client.camera.wxyz = self.camera_wxyz
                client.camera.position = self.camera_pos

        # --- Callbacks: Navigation ---
        @btn_prev_scan_100.on_click
        def _(_) -> None:
            if self.current_scan_idx > 0:
                self.current_scan_idx = max(0, self.current_scan_idx - 100)
                self.set_scan()
                self.update_view()

        @btn_prev_scan.on_click
        def _(_) -> None:
            if self.current_scan_idx > 0:
                self.current_scan_idx -= 1
                self.set_scan()
                self.update_view()

        @btn_next_scan.on_click
        def _(_) -> None:
            if self.current_scan_idx < len(self.scans) - 1:
                self.current_scan_idx += 1
                self.set_scan()
                self.update_view()

        @btn_next_scan_100.on_click
        def _(_) -> None:
            if self.current_scan_idx < len(self.scans) - 1:
                self.current_scan_idx = min(len(self.scans) - 1, self.current_scan_idx + 100)
                self.set_scan()
                self.update_view()

        @btn_prev_vertex_100.on_click
        def _(_) -> None:
            self.current_vertex_idx = max(0, self.current_vertex_idx - 100)
            self.update_view()

        @btn_prev_vertex.on_click
        def _(_) -> None:
            self.current_vertex_idx = max(0, self.current_vertex_idx - 1)
            self.update_view()

        @btn_next_vertex.on_click
        def _(_) -> None:
            self.current_vertex_idx = min(len(self.vertices) - 1, self.current_vertex_idx + 1)
            self.update_view()

        @btn_next_vertex_100.on_click
        def _(_) -> None:
            self.current_vertex_idx = min(len(self.vertices) - 1, self.current_vertex_idx + 100)
            self.update_view()

        # --- Callbacks: Display Settings ---
        # When display settings change, we only redraw the scene using existing arrays
        @self.gui_sss_pos.on_update
        def _(_) -> None: self._render_scene_objects()
        @self.gui_sss_scale.on_update
        def _(_) -> None: self._render_scene_objects()

        @self.gui_marker_size.on_update
        def _(_) -> None: self._render_scene_objects()

        @self.gui_auv_scale.on_update
        def _(_) -> None: self._render_scene_objects()

    def get_timestamp(self) -> int:
        """Returns the timestamp of the current image."""
        return int(self.scans[self.current_scan_idx].name.replace(".csv", ''))

    def set_scan(self) -> None:
        """Loads the matches and optical image for the current index."""
        self.current_vertex_idx = 0
        self.vertices.clear()

        with open(self.scans[self.current_scan_idx], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.vertices.append(VertexHit.from_dict(row))

    def draw_target(self, img_array: np.ndarray, u: int, v: int) -> np.ndarray:
        """Draws a highly visible red target on a copy of the image array."""
        canvas = img_array.copy()
        
        # Draw white halo then red center so it pops against any background
        cv2.circle(canvas, (u, v), radius=14, color=(255, 255, 255), thickness=-1)
        cv2.circle(canvas, (u, v), radius=10, color=(255, 0, 0), thickness=-1)
        return canvas

    def update_view(self):
        """Processes image crops and redraws OpenCV targets, then updates Viser."""
        if not self.vertices:
            self.gui_info.content = f"### Scan {self.current_scan_idx}\nNo matches found."
            return

        vertex = self.vertices[self.current_vertex_idx]
        self.auv_pose = vertex.hit.pose

        # --- 1. Prepare SSS Patch ---
        half_patch = self.patch_size // 2
        ping_start = max(0, vertex.hit.ping_idx - half_patch)
        ping_end = min(self.sss_image.shape[0], vertex.hit.ping_idx + half_patch)
        bin_start = max(0, vertex.hit.bin_idx - half_patch)
        bin_end = min(self.sss_image.shape[1], vertex.hit.bin_idx + half_patch)

        sss_patch = self.sss_image[ping_start:ping_end, bin_start:bin_end]

        dot_x = vertex.hit.bin_idx - bin_start
        dot_y = vertex.hit.ping_idx - ping_start
        self.sss_patch_marked = self.draw_target(sss_patch, int(dot_x), int(dot_y))

        # --- 3. Prepare 3D Point ---
        self.target_3d = np.array([[vertex.p_x, vertex.p_y, vertex.p_z]]) - self.center_offset

        # Update text info
        markdown_text = (
            f"### Status\n"
            f"**Timestamp:** {vertex.hit.pose.timestamp}\n\n"
            f"**Scan:** {self.current_scan_idx + 1} / {len(self.scans)}\n\n"
            f"**Vertex:** {self.current_vertex_idx + 1} / {len(self.vertices)}\n\n"
            f"---\n"
            f"**Sonar Ping:** {vertex.hit.ping_idx}\n\n"
            f"**Sonar Bin:** {vertex.hit.bin_idx}\n\n"
            f"**Distance:** {vertex.hit.distance:.2f}m\n\n"
            f"**Incidence Angle:** {vertex.hit.incidence_angle:.2f} rad\n\n"
            f"**AUV Pose (NED):** ({vertex.hit.pose.x:.2f}, {vertex.hit.pose.y:.2f}, {vertex.hit.pose.z:.2f}) m\n\n"
            f"**3D Point (NED):** ({vertex.p_x:.2f}, {vertex.p_y:.2f}, {vertex.p_z:.2f}) m\n\n"
        )
        self.gui_info.content = markdown_text

        # Finally, push everything to the 3D scene
        self._render_scene_objects()

    def _render_scene_objects(self):
        """Pushes the current arrays and positions to the Viser 3D scene."""
        if self.sss_patch_marked is None:
            return

        # Place SSS Patch
        self.server.scene.add_image(
            name="/views/sss_patch",
            image=self.sss_patch_marked,
            render_width=self.gui_sss_scale.value,
            render_height=self.gui_sss_scale.value,
            position=self.gui_sss_pos.value,
        )

        # Draw 3D Point Target
        if self.target_3d is not None:
            self.server.scene.add_point_cloud(
                name="/colmap/match_point",
                points=self.target_3d,
                colors=np.array([[255, 0, 0]]), # Red
                point_size=self.gui_marker_size.value,
            )

        # Draw AUV Pose
        if self.auv_pose is not None:
            # Shift AUV position by the same offset applied to the point cloud
            auv_pos = self.auv_pose.get_position() - self.center_offset
            
            # Viser expects quaternion in w, x, y, z format
            auv_wxyz = (
                self.auv_pose.qw,
                self.auv_pose.qx,
                self.auv_pose.qy,
                self.auv_pose.qz
            )

            # Draw coordinate axes (Red=X, Green=Y, Blue=Z)
            self.server.scene.add_frame(
                name="/colmap/auv_pose",
                wxyz=auv_wxyz,
                position=auv_pos,
                axes_length=self.gui_auv_scale.value,
                axes_radius=self.gui_auv_scale.value * 0.05,
            )

    def run(self):
        """Keeps the server running."""
        while True:
            time.sleep(1.0)