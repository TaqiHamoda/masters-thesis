import pycolmap
import numpy as np
from pathlib import Path
from typing import Tuple

from .dataset import Dataset

class Photogrammetry:
    # Example: https://deepwiki.com/colmap/pycolmap/5.2-multi-view-stereo-(mvs)#complete-mvs-pipeline-example

    def __init__(self, dataset: Dataset, output_dir: str = "output/"):
        self.dataset = dataset
        if len(self.dataset.images) == 0:
            raise ValueError("Dataset contains no images. Please load data before initializing Photogrammetry.")

        self.images_dir = self.dataset.image_dir
        if not self.images_dir.exists():
            raise ValueError("Image directory does not exist.")

        # Camera params
        img = self.dataset.images[0]
        self.camera_params = (img.fx, img.fy, img.cx, img.cy)

        # Prepare paths for COLMAP outputs
        self.output_dir = Path(output_dir)

        self.database_path = self.output_dir / "database.db"
        self.sparse_path = self.output_dir / "sparse"
        self.sparse_refined_path = self.output_dir / "sparse_pruned"
        self.mvs_path = self.output_dir / "mvs"
        self.dense_ply = self.mvs_path / "dense.ply"
        self.mesh_ply = self.output_dir / "mesh.ply"

    def extract_and_match_features(self,
        contrast_threshold: float = 0.002,
        pos_std: Tuple[float, float, float] = (2.0, 2.0, 0.1),
        max_distance: float = 5.0
    ):
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = ",".join(map(str, self.camera_params))

        extraction_options = pycolmap.SiftExtractionOptions()
        extraction_options.peak_threshold = contrast_threshold    # Default is 0.006666666666666667

        pycolmap.extract_features(
            database_path=self.database_path,
            image_path=self.images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            sift_options=extraction_options
        )

        with pycolmap.Database(str(self.database_path)) as colmap_db:
            position_covariance = np.diag(np.power(pos_std, 2))

            for img in Dataset.images:
                if colmap_db.exists_image(img.filename):
                    image = colmap_db.read_image(img.filename)
                    # Reshape position to a 3x1 column vector as expected by PosePrior
                    position = np.array([img.pose.x, img.pose.y, img.pose.z]).reshape(3, 1)

                    # Coordinate system: Cartesian (X,Y,Z coords, not Lat/Lon)
                    colmap_db.write_pose_prior(image.image_id,
                        pycolmap.PosePrior(
                            position,
                            position_covariance,
                            pycolmap.PosePriorCoordinateSystem(pycolmap.PosePriorCoordinateSystem.CARTESIAN)
                        )
                    )
                else:
                    print(f"Warning: Image {img.filename} not found in DB.")

        # Configure Spatial Matching
        spatial_opts = pycolmap.SpatialMatchingOptions()
        spatial_opts.max_distance = max_distance
        spatial_opts.ignore_z = False 

        pycolmap.match_spatial(self.database_path, matching_options=spatial_opts)

    def bundle_adjustment(self):
        if not self.database_path.exists():
            raise ValueError("Database not found. Please run feature extraction and matching before bundle adjustment.")

        self.sparse_path.mkdir(exist_ok=True)

        # Incrementally registers images, Triangulates 3D points,
        # and Performs local and global bundle adjustment
        maps = pycolmap.incremental_mapping(self.database_path, self.images_dir, self.sparse_path)
        maps[0].write(self.sparse_path)
        maps[0].write_text(self.sparse_path)

    def prune_poses(self,
        max_reproj_error: float = 1.0,
        min_num_points: int = 20
    ):
        if not self.sparse_path.exists():
            raise ValueError("Sparse reconstruction not found. Please run bundle adjustment before pruning.")

        self.sparse_refined_path.mkdir(exist_ok=True)

        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(self.sparse_path)

        for point3D_id, point3D in reconstruction.points3D.items():
            if point3D.error > max_reproj_error:
                reconstruction.delete_point3D(point3D_id)

        to_remove = []
        for img_id in reconstruction.reg_image_ids():
            img = reconstruction.images[img_id]
            if img.num_points3D < min_num_points:
                to_remove.append(img_id)

        for img_id in to_remove:
            reconstruction.deregister_image(img_id)

        reconstruction.write(self.sparse_refined_path)
        reconstruction.write_text(self.sparse_refined_path)

    def stereo_matching(self):
        if not self.sparse_path.exists():
            raise ValueError("Sparse reconstruction not found. Please run bundle adjustment before pruning.")

        self.mvs_path.mkdir(exist_ok=True)

        pycolmap.undistort_images(self.mvs_path, self.sparse_path, self.images_dir)
        pycolmap.patch_match_stereo(self.mvs_path)

    def dense_reconstruction(self,
        max_image_size: int = 2000,
        check_num_images: int = 10,
        cache_size: int = 32
    ):
        if not self.mvs_path.exists():
            raise ValueError("MVS outputs not found. Please run stereo matching before dense reconstruction.")

        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.max_image_size = max_image_size      # Maximum image size in either dimension
        fusion_options.check_num_images = check_num_images  # Number of overlapping images to transitively check for fusing points (default: 50)
        fusion_options.use_cache = True                     # Enables disk caching to save RAM
        fusion_options.cache_size = cache_size              # Cache size in gigabytes for fusion (default: 32.0)

        pycolmap.stereo_fusion(self.dense_ply, self.mvs_path, options=fusion_options)

    def create_mesh(self):
        if not self.dense_ply.exists():
            raise ValueError("Dense reconstruction outputs not found. Please run dense reconstruction before meshing.")

        pycolmap.poisson_meshing(self.dense_ply, self.mesh_ply)

    @staticmethod
    def has_cuda() -> bool:
        return pycolmap.has_cuda

