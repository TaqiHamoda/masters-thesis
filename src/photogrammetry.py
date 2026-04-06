import pycolmap
import numpy as np
from pathlib import Path
from typing import Tuple

from .dataset import Dataset

class Photogrammetry:
    # Example: https://deepwiki.com/colmap/pycolmap/5.2-multi-view-stereo-(mvs)#complete-mvs-pipeline-example

    def __init__(self, dataset: Dataset, output_path: str = "output/"):
        self.dataset = dataset
        if len(self.dataset.images) == 0:
            raise ValueError("Dataset contains no images. Please load data before initializing Photogrammetry.")

        self.images_dir = self.dataset.image_dir
        if not self.images_dir.exists():
            raise ValueError("Image directory does not exist.")

        # Camera params
        img = list(self.dataset.images.values())[0]
        self.camera_params = (img.fx, img.fy, img.cx, img.cy)

        # Prepare paths for COLMAP outputs
        self.output_dir = Path(output_path)

        self.database_path = self.output_dir / "database.db"
        self.sparse_path = self.output_dir / "sparse"
        self.sparse_refined_path = self.output_dir / "sparse_pruned"
        self.mvs_path = self.output_dir / "mvs"
        self.dense_ply = self.mvs_path / "dense.ply"
        self.mesh_ply = self.output_dir / "mesh.ply"

    def extract_and_match_features(self,
        contrast_threshold: float = 0.002,
        max_num_features: int = 8192,
        pos_trans: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        pos_std: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        max_distance: float = 5.0
    ):
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = ",".join(map(str, self.camera_params))

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.sift.peak_threshold = contrast_threshold    # Default is 0.006666666666666667
        extraction_options.sift.max_num_features = max_num_features    # Default is 8192

        pycolmap.extract_features(
            database_path=self.database_path,
            image_path=self.images_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options 
        )

        # Source: https://github.com/colmap/colmap/issues/2976#issuecomment-3930305589
        with pycolmap.Database.open(self.database_path) as colmap_db:
            position_covariance = np.diag(np.power(pos_std, 2))

            for image in colmap_db.read_all_images():
                pose = self.dataset.images[image.name].pose
                position = (
                    np.array((pose.x, pose.y, pose.z)) + np.array(pos_trans)
                ).reshape(3, 1)

                # Coordinate system: Cartesian (X,Y,Z coords, not Lat/Lon)
                colmap_db.write_pose_prior(
                    pycolmap.PosePrior(
                        corr_data_id=image.data_id,  # Link the prior to the specific image's data identifier
                        position=position,
                        position_covariance=position_covariance,
                        coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN
                    )
                )

        # Configure Spatial Matching
        spatial_opts = pycolmap.SpatialPairingOptions()
        spatial_opts.max_distance = max_distance

        pycolmap.match_spatial(
            self.database_path,
            pairing_options=spatial_opts
        )

    def sparse_reconstruction(self,
        ba_global_ratio: float = 1.1,
        ba_global_frames_freq: int = 500,
        ba_global_points_freq: int = 250000,
        ba_global_max_num_iterations: int = 50,
        ba_global_max_refinements: int = 5,
    ):
        """Incrementally registers images, Triangulates 3D points, and Performs local and global bundle adjustment"""
        if not self.database_path.exists():
            raise ValueError("Database not found. Please run feature extraction and matching before sparse reconstruction.")

        self.sparse_path.mkdir(exist_ok=True)

        options = pycolmap.IncrementalPipelineOptions()
        options.ba_global_frames_ratio = ba_global_ratio
        options.ba_global_points_ratio = ba_global_ratio
        options.ba_global_frames_freq = ba_global_frames_freq
        options.ba_global_points_freq = ba_global_points_freq
        options.ba_global_max_num_iterations = ba_global_max_num_iterations
        options.ba_global_max_refinements = ba_global_max_refinements

        options.ba_use_gpu = self.has_cuda()
        options.use_prior_position = True

        maps = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=self.images_dir,
            output_path=self.sparse_path,
            options=options
        )

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
