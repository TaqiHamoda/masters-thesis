import cv2
import numpy as np

import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
)

import time
from typing import List

from .dataset import Dataset, ImageHit
from .photogrammetry import Photogrammetry
from .registration import Registration

CHUNK_SIZE = 10_000_000  # Number of vertices to load at a time by the GPU


class MatchVisualizer:
    def __init__(
        self,
        dataset: Dataset,
        registration: Registration,
        patch_size: int = 1000
    ):
        """
        Initializes the Viser-based visualization tool.
        """
        self.camera_pos = np.array((0, 0, -100))
        self.camera_wxyz = np.array((1.0, 0.0, 0.0, 0.0))

        self.dataset = dataset
        self.registration = registration
        self.patch_size = patch_size

        self.current_img_idx = 0
        self.current_match_idx = 0

        self.images = sorted(self.registration.img_ids.keys())
        self.matches: List[ImageHit] = []

        # Start Viser server
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
            titlebar_content=None, 
            control_layout="collapsible", 
            control_width="large", 
            dark_mode=True
        )

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
        self.camera_frustum = None
        
        # New handles for GUI sidebar images
        self.gui_cam_image = None
        self.gui_sss_image = None

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
        cameras = read_cameras_binary(colmap_path / "cameras.bin")
        images = read_images_binary(colmap_path / "images.bin")

        self.camera = cameras.popitem()[1]
        self.camera_poses = {
            int(img.name.replace(".jpg", '')): (img.qvec, img.tvec)
            for img in images.values()
        }

        self.registration.load_mesh()
        points = self.registration.vertices
        colors = self.registration.mesh.vertex.colors.numpy()

        reflectivity = np.load(self.dataset.reflectivity_vertices)["data"]
        reflectivity = np.clip(reflectivity, 0, np.percentile(reflectivity, 99))
        reflectivity -= np.min(reflectivity)
        reflectivity /= np.max(reflectivity)
        reflectivity = (255 * reflectivity).astype(np.uint8)
        reflectivity = cv2.applyColorMap(reflectivity, cv2.COLORMAP_BONE).reshape((len(reflectivity), 3))

        # Center the point cloud roughly around the origin for easier viewing
        self.center_offset = points.mean(axis=0)
        points -= self.center_offset

        for i in range(0, len(points), CHUNK_SIZE):
            chunk_idx = i // CHUNK_SIZE
            
            # Slice the arrays
            point_chunk = points[i : i + CHUNK_SIZE]
            color_chunk = colors[i : i + CHUNK_SIZE]
            reflectivity_chunk = reflectivity[i : i + CHUNK_SIZE]

            # Push each chunk as a separate object in the scene tree
            self.server.scene.add_point_cloud(
                name=f"/colmap/pcd/chunk_{chunk_idx}",
                points=point_chunk,
                colors=color_chunk,
                point_size=0.1,
            )

            self.server.scene.add_point_cloud(
                name=f"/colmap/reflectivity/chunk_{chunk_idx}",
                points=point_chunk,
                colors=reflectivity_chunk,
                point_size=0.1,
            )

    def _build_gui(self):
        """Creates the side-panel buttons and info displays."""
        self.gui_info = self.server.gui.add_markdown("Loading data...")

        # Add image widgets to sidebar
        folder_images = self.server.gui.add_folder("Match Views")
        with folder_images:
            # Initialize with small placeholder arrays, they update dynamically in _render_scene_objects
            dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
            self.gui_cam_image = self.server.gui.add_image(dummy_img, format="jpeg")
            self.gui_sss_image = self.server.gui.add_image(dummy_img, format="jpeg")

        # View Controls
        folder_view = self.server.gui.add_folder("View Controls")
        with folder_view:
            btn_reset_view = self.server.gui.add_button("Reset View")

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
            # Cleaned out the redundant image positioning sliders
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
        @self.gui_marker_size.on_update
        def _(_) -> None: self._render_scene_objects()

        @self.gui_auv_scale.on_update
        def _(_) -> None: self._render_scene_objects()

    def get_timestamp(self) -> int:
        """Returns the timestamp of the current image."""
        return self.images[self.current_img_idx]

    def set_image(self) -> None:
        """Loads the matches and optical image for the current index."""
        self.current_match_idx = 0
        self.matches.clear()

        ts = self.get_timestamp()

        matches_path = self.dataset.image_matches_dir / f"{ts}.csv"
        if matches_path.exists():
            self.matches.extend(ImageHit.from_csv(matches_path))
        else:
            self.matches.extend(self.registration.get_matches(ts))
            if len(self.matches) > 0:
                ImageHit.to_csv(matches_path, self.matches)

        # Load optical image and ensure it's RGB
        img_path = self.dataset.image_dir / f"{ts}.jpg"
        self.image = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)

        if self.camera_frame is not None:
            self.camera_frame.remove()
            self.camera_frame = None
            self.camera_frustum = None  # When you remove camera_frame, the frustum is automatically removed by Viser

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
        self.cam_img_marked = self.draw_target(self.image, match.u, match.v)

        # --- 2. Prepare SSS Patch (Fixed padding logic) ---
        half_patch = self.patch_size // 2
        
        c_y = match.hit.ping_idx
        c_x = match.hit.bin_idx
        
        # Calculate theoretical bounds (can be negative or exceed image size)
        y1, y2 = c_y - half_patch, c_y + half_patch
        x1, x2 = c_x - half_patch, c_x + half_patch
        
        img_h, img_w = self.sss_image.shape[:2]
        
        # Calculate valid bounds within the actual image array
        y1_valid, y2_valid = max(0, y1), min(img_h, y2)
        x1_valid, x2_valid = max(0, x1), min(img_w, x2)
        
        # Extract the valid pixels from the source image
        valid_patch = self.sss_image[y1_valid:y2_valid, x1_valid:x2_valid]
        
        # Create a perfectly square black canvas
        sss_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Calculate where to insert the valid pixels into the black canvas
        insert_y1 = max(0, -y1)
        insert_y2 = insert_y1 + (y2_valid - y1_valid)
        insert_x1 = max(0, -x1)
        insert_x2 = insert_x1 + (x2_valid - x1_valid)
        
        # Paste the valid image data into the square patch
        sss_patch[insert_y1:insert_y2, insert_x1:insert_x2] = valid_patch

        # The target is now guaranteed to be dead-center in the square patch
        self.sss_patch_marked = self.draw_target(sss_patch, int(half_patch), int(half_patch))

        # --- 3. Prepare 3D Point ---
        self.target_3d = np.array([[match.hit.p_x, match.hit.p_y, match.hit.p_z]]) - self.center_offset

        image_pose = self.dataset.images[self.get_timestamp()].pose

        # Update text info
        markdown_text = (
            f"### Status\n"
            f"**Timestamp:** {self.get_timestamp()}\n\n"
            f"**Image:** {self.current_img_idx + 1} / {len(self.images)}\n\n"
            f"**Match:** {self.current_match_idx + 1} / {len(self.matches)}\n\n"
            f"---\n"
            f"**Optical Pixel (u, v):** ({match.u}, {match.v})\n\n"
            f"**Sonar Ping:** {match.hit.ping_idx}\n\n"
            f"**Sonar Bin:** {match.hit.bin_idx}\n\n"
            f"**Distance:** {match.hit.distance:.2f}m\n\n"
            f"**Incidence Angle:** {match.hit.incidence_angle:.2f} rad\n\n"
            f"**CAM (NED):** ({image_pose.x:.2f}, {image_pose.y:.2f}, {image_pose.z:.2f}) m\n\n"
            f"**AUV (NED):** ({match.hit.pose.x:.2f}, {match.hit.pose.y:.2f}, {match.hit.pose.z:.2f}) m\n\n"
            f"**PCD (NED):** ({match.hit.p_x:.2f}, {match.hit.p_y:.2f}, {match.hit.p_z:.2f}) m\n\n"
        )
        self.gui_info.content = markdown_text

        # Finally, push everything to the 3D scene
        self._render_scene_objects()

    def _render_scene_objects(self):
        """Pushes the current arrays and positions to the Viser 3D scene."""
        if self.cam_img_marked is None or self.sss_patch_marked is None:
            return

        # Dynamically push arrays to the GUI images
        self.gui_cam_image.image = self.cam_img_marked
        self.gui_sss_image.image = self.sss_patch_marked

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
            self.camera_frustum = self.server.scene.add_camera_frustum(
                f"/colmap/frame_{ts}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=1.0,
                image=self.cam_img_marked,
            )

            @self.camera_frustum.on_click
            def _(_, frame=self.camera_frame) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        self.camera_frustum.image = self.cam_img_marked

    def run(self):
        """Keeps the server running."""
        while True:
            time.sleep(1.0)