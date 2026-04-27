import csv
import time
from typing import List

import cv2
import numpy as np
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

from ..dataset import Dataset, ImageHit
from ..photogrammetry import Photogrammetry


class MatchVisualizer:
    def __init__(self, dataset: Dataset, patch_size: int = 1000, downsample_factor: int = 1):
        """
        Initializes the Viser-based visualization tool.
        """
        self.dataset = dataset
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor

        self.current_img_idx = 0
        self.current_match_idx = 0

        self.images = sorted(dataset.matches_dir.glob("*.csv"))
        self.matches: List[ImageHit] = []

        # Start Viser server
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # Load SSS waterfall (convert to RGB so we can draw a red dot on it)
        self.sss_image = cv2.imread(str(dataset.sonar_png), cv2.IMREAD_COLOR_RGB)
        self.sss_image = cv2.flip(self.sss_image, 0)

        # Pre-declare layout handles
        self.cam_img_marked = None
        self.sss_patch_marked = None
        self.target_3d = None
        self.auv_pose = None

        self.camera = None
        self.camera_poses = None
        self.camera_frame = None

        # Load Colmap Point Cloud
        self._load_point_cloud()

        # Build GUI Controls
        self._build_gui()

        # Load initial data
        self.set_image()
        self.update_view()

    def _load_point_cloud(self):
        """Loads and displays the sparse point cloud in the 3D scene."""
        colmap_path = Photogrammetry(self.dataset).sparse_path / "0"
        points3d = read_points3d_binary(colmap_path / "points3D.bin")
        cameras = read_cameras_binary(colmap_path / "cameras.bin")
        images = read_images_binary(colmap_path / "images.bin")

        self.camera = cameras.popitem()[1]
        self.camera_poses = {
            int(img.name.replace(".jpg", '')): (img.qvec, img.tvec)
            for img in images.values()
        }

        points = np.array([points3d[p_id].xyz for p_id in points3d])
        colors = np.array([points3d[p_id].rgb for p_id in points3d])

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

        # Image Controls
        folder_img = self.server.gui.add_folder("Image Navigation")
        with folder_img:
            btn_prev_img_100 = self.server.gui.add_button("-100 Images")
            btn_prev_img = self.server.gui.add_button("Prev Image")
            btn_next_img = self.server.gui.add_button("Next Image")
            btn_next_img_100 = self.server.gui.add_button("+100 Images")

        # Match Controls
        folder_match = self.server.gui.add_folder("Match Navigation")
        with folder_match:
            btn_prev_match_100 = self.server.gui.add_button("-100 Matches")
            btn_prev_match = self.server.gui.add_button("Prev Match")
            btn_next_match = self.server.gui.add_button("Next Match")
            btn_next_match_100 = self.server.gui.add_button("+100 Matches")

        # Layout & Display Controls
        folder_layout = self.server.gui.add_folder("Display Settings")
        with folder_layout:
            self.gui_img_pos = self.server.gui.add_vector3(
                "Image Pos (x,y,z)", initial_value=(100, -30, -20), step=0.1
            )
            self.gui_img_scale = self.server.gui.add_slider(
                "Image Scale", min=0.1, max=100.0, step=0.1, initial_value=50.0
            )

            self.gui_sss_pos = self.server.gui.add_vector3(
                "SSS Pos (x,y,z)", initial_value=(100, 30, -20), step=0.1
            )
            self.gui_sss_scale = self.server.gui.add_slider(
                "SSS Scale", min=0.1, max=100.0, step=0.1, initial_value=50.0
            )

            self.gui_marker_size = self.server.gui.add_slider(
                "3D Marker Size", min=0.01, max=2.0, step=0.01, initial_value=1.0
            )

            self.gui_auv_scale = self.server.gui.add_slider(
                "AUV Axes Scale", min=0.1, max=10.0, step=0.1, initial_value=7.0
            )

        # --- Callbacks: Navigation ---
        @btn_prev_img_100.on_click
        def _(_) -> None:
            if self.current_img_idx > 0:
                self.current_img_idx = max(0, self.current_img_idx - 100)
                self.set_image()
                self.update_view()

        @btn_prev_img.on_click
        def _(_) -> None:
            if self.current_img_idx > 0:
                self.current_img_idx -= 1
                self.set_image()
                self.update_view()

        @btn_next_img.on_click
        def _(_) -> None:
            if self.current_img_idx < len(self.images) - 1:
                self.current_img_idx += 1
                self.set_image()
                self.update_view()

        @btn_next_img_100.on_click
        def _(_) -> None:
            if self.current_img_idx < len(self.images) - 1:
                self.current_img_idx = min(len(self.images) - 1, self.current_img_idx + 100)
                self.set_image()
                self.update_view()

        @btn_prev_match_100.on_click
        def _(_) -> None:
            self.current_match_idx = max(0, self.current_match_idx - 100)
            self.update_view()

        @btn_prev_match.on_click
        def _(_) -> None:
            self.current_match_idx = max(0, self.current_match_idx - 1)
            self.update_view()

        @btn_next_match.on_click
        def _(_) -> None:
            self.current_match_idx = min(len(self.matches) - 1, self.current_match_idx + 1)
            self.update_view()

        @btn_next_match_100.on_click
        def _(_) -> None:
            self.current_match_idx = min(len(self.matches) - 1, self.current_match_idx + 100)
            self.update_view()

        # --- Callbacks: Display Settings ---
        # When display settings change, we only redraw the scene using existing arrays
        @self.gui_img_pos.on_update
        def _(_) -> None: self._render_scene_objects()
        @self.gui_img_scale.on_update
        def _(_) -> None: self._render_scene_objects()
        
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
        return int(self.images[self.current_img_idx].name.replace(".csv", ''))

    def set_image(self) -> None:
        """Loads the matches and optical image for the current index."""
        self.current_match_idx = 0
        self.matches.clear()

        with open(self.images[self.current_img_idx], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.matches.append(ImageHit.from_dict(row))

        # Load optical image and ensure it's RGB
        img_name = self.images[self.current_img_idx].name.replace(".csv", ".jpg")
        img_path = self.dataset.image_dir / img_name
        self.image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        if self.camera_frame is not None:
            self.camera_frame.remove()
            self.camera_frame = None

    def draw_target(self, img_array: np.ndarray, u: int, v: int) -> np.ndarray:
        """Draws a highly visible red target on a copy of the image array."""
        canvas = img_array.copy()
        
        # Draw white halo then red center so it pops against any background
        cv2.circle(canvas, (u, v), radius=14, color=(255, 255, 255), thickness=-1)
        cv2.circle(canvas, (u, v), radius=10, color=(255, 0, 0), thickness=-1)
        return canvas

    def update_view(self):
        """Processes image crops and redraws OpenCV targets, then updates Viser."""
        if not self.matches:
            self.gui_info.content = f"### Image {self.current_img_idx}\nNo matches found."
            return

        match = self.matches[self.current_match_idx]
        self.auv_pose = match.hit.pose

        # --- 1. Prepare Camera Image ---
        scaled_u = int(match.u / self.downsample_factor)
        scaled_v = int(match.v / self.downsample_factor)
        self.cam_img_marked = self.draw_target(self.image, scaled_u, scaled_v)

        # --- 2. Prepare SSS Patch ---
        half_patch = self.patch_size // 2
        ping_start = max(0, match.hit.ping_idx - half_patch)
        ping_end = min(self.sss_image.shape[0], match.hit.ping_idx + half_patch)
        bin_start = max(0, match.hit.bin_idx - half_patch)
        bin_end = min(self.sss_image.shape[1], match.hit.bin_idx + half_patch)

        sss_patch = self.sss_image[ping_start:ping_end, bin_start:bin_end]

        dot_x = match.hit.bin_idx - bin_start
        dot_y = match.hit.ping_idx - ping_start
        self.sss_patch_marked = self.draw_target(sss_patch, int(dot_x), int(dot_y))

        # --- 3. Prepare 3D Point ---
        self.target_3d = np.array([[match.p_x, match.p_y, match.p_z]]) - self.center_offset

        image_pose = self.dataset.images[self.get_timestamp()].pose

        # Update text info
        markdown_text = (
            f"### Status\n"
            f"**Timestamp:** {match.hit.pose.timestamp}\n\n"
            f"**Image:** {self.current_img_idx + 1} / {len(self.images)}\n\n"
            f"**Match:** {self.current_match_idx + 1} / {len(self.matches)}\n\n"
            f"---\n"
            f"**Optical Pixel (u, v):** ({match.u}, {match.v})\n\n"
            f"**Sonar Ping:** {match.hit.ping_idx}\n\n"
            f"**Sonar Bin:** {match.hit.bin_idx}\n\n"
            f"**Distance:** {match.hit.distance:.2f}m\n\n"
            f"**Incidence Angle:** {match.hit.incidence_angle:.2f} rad\n\n"
            f"**Camera Pose (NED):** ({image_pose.x:.2f}, {image_pose.y:.2f}, {image_pose.z:.2f}) m\n\n"
            f"**AUV Pose (NED):** ({match.hit.pose.x:.2f}, {match.hit.pose.y:.2f}, {match.hit.pose.z:.2f}) m\n\n"
            f"**3D Point (NED):** ({match.p_x:.2f}, {match.p_y:.2f}, {match.p_z:.2f}) m\n\n"
        )
        self.gui_info.content = markdown_text

        # Finally, push everything to the 3D scene
        self._render_scene_objects()

    def _render_scene_objects(self):
        """Pushes the current arrays and positions to the Viser 3D scene."""
        if self.cam_img_marked is None or self.sss_patch_marked is None:
            return

        # Place Camera Image
        self.server.scene.add_image(
            name="/views/camera_image",
            image=self.cam_img_marked,
            render_width=self.gui_img_scale.value,
            render_height=self.gui_img_scale.value * self.cam_img_marked.shape[0] / self.cam_img_marked.shape[1],
            position=self.gui_img_pos.value,
        )

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

        # Draw Camera Pose
        if self.camera_poses is not None and self.camera_frame is None:
            ts = self.get_timestamp()
            qvec, tvec = self.camera_poses[ts]

            T_world_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(qvec), tvec
            ).inverse()
            self.camera_frame = self.server.scene.add_frame(
                f"/colmap/frame_{ts}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation() - self.center_offset,
                axes_length=0.1,
                axes_radius=0.005,
            )

            H, W = self.camera.height, self.camera.width
            fy = self.camera.params[1]
            frustum = self.server.scene.add_camera_frustum(
                f"/colmap/frame_{ts}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=1.0,
                image=self.cam_img_marked,
            )

            @frustum.on_click
            def _(_, frame=self.camera_frame) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

    def run(self):
        """Keeps the server running."""
        while True:
            time.sleep(1.0)