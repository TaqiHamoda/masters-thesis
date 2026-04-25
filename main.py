import yaml
from time import perf_counter, sleep
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.dataset import Dataset
from src.photogrammetry import Photogrammetry
from src.sonar import export_to_xtf, export_to_png
from src.registration import interpolate_poses, get_image_geometry, process_optical_sidescan_matches
from src.visualization import visualize_sparse_pointcloud, MatchVisualizer


def photogrammetry_pipeline(photogrammetry: Photogrammetry, cfg: dict):
    if not photogrammetry.has_cuda():
        print("CUDA not available. Consider installing COLMAP with CUDA support.")
        sleep(5)

    # Feature Extraction
    if not photogrammetry.database_path.exists():
        print("Extracting and matching features...")
        start_time = perf_counter()
        photogrammetry.extract_and_match_features(**cfg['features'])
        print(f"Features completed in {perf_counter() - start_time:.2f}s")

    # Sparse Reconstruction
    if not photogrammetry.sparse_path.exists():
        print("Performing sparse reconstruction...")
        start_time = perf_counter()
        photogrammetry.sparse_reconstruction(**cfg['sparse'])
        print(f"Sparse completed in {perf_counter() - start_time:.2f}s")

    # Stereo Matching
    if not photogrammetry.stereo_path.exists():
        print(f"Performing stereo matching...")
        start_time = perf_counter()
        photogrammetry.stereo_matching()
        print(f"Stereo matching completed in {perf_counter() - start_time:.2f}s")

    # Dense Reconstruction
    if not photogrammetry.fused_ply.exists():
        print("Performing dense reconstruction...")
        start_time = perf_counter()
        photogrammetry.dense_reconstruction(**cfg['dense'])
        print(f"Dense completed in {perf_counter() - start_time:.2f}s")

    # Mesh Creation
    if not photogrammetry.mesh_ply.exists():
        print("Creating mesh...")
        start_time = perf_counter()
        photogrammetry.create_mesh(**cfg['mesh'])
        print(f"Mesh completed in {perf_counter() - start_time:.2f}s")


def process_optical_single_image(img, reconstruction, dataset, poses):
    ts = int(img.name.replace(".jpg", ''))

    optical, points = get_image_geometry(reconstruction, img.image_id)
    matches = process_optical_sidescan_matches(dataset, poses[0][ts], poses[1], optical, points)

    Dataset.write_data(dataset.matches_dir / f"{ts}.csv", matches)


if __name__ == "__main__":
    # Load settings from the YAML file
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Extract sections for easier access
    paths_cfg = cfg['paths']
    topics_cfg = cfg['topics']
    extrinsics_cfg = cfg['extrinsics']
    photogrammetry_cfg = cfg['photogrammetry']
    xtf_cfg = cfg['process_acoustic']
    optical_cfg = cfg['optical_registration']
    visual_cfg = cfg['visualizations']

    dataset = Dataset(
        data_path=paths_cfg['data_path'],
        output_path=paths_cfg['output_path'],
        img_topic=topics_cfg['img_topic'],
        odo_topic=topics_cfg['odo_topic'],
        info_topic=topics_cfg['info_topic'],
        sonar_topic=topics_cfg['sonar_topic'],
        nav_topic=topics_cfg['nav_topic'],
        camera_trans=extrinsics_cfg['camera'],
        sonar_trans=extrinsics_cfg['sonar'],
    )

    if dataset.exists():
        print("Loading processed dataset...")
        dataset.load_data_from_csv()
    else:
        print("Processing raw data from bags...")
        dataset.load_data_from_bags()
        dataset.export_data()
        dataset.inspect_bags()
        dataset.data_stats()

    if photogrammetry_cfg['enabled']:
        photogrammetry = Photogrammetry(dataset, output_path=paths_cfg['output_path'])
        photogrammetry_pipeline(photogrammetry, photogrammetry_cfg)

    if xtf_cfg['enabled']:
        if not dataset.sonar_xtf.exists():
            print("Processing Sonar Data into XTF file...")
            export_to_xtf(dataset, xtf_cfg['sonar_name'], xtf_cfg['sample_dtype'])

        if not dataset.sonar_png.exists():
            print("Processing Sonar Data into PNG file...")
            export_to_png(dataset, xtf_cfg['sample_dtype'])

    if optical_cfg['enabled']:
        print("Performing optical registration...")

        reconstruction = Photogrammetry.get_reconstruction(dataset)
        interpolated_poses = interpolate_poses(dataset, reconstruction)

        with ThreadPoolExecutor(max_workers=6) as executor:
            images = list(reconstruction.images.values())

            list(tqdm(
                executor.map(
                    lambda img: process_optical_single_image(img, reconstruction, dataset, interpolated_poses),
                    images
                ),
                total=len(images),
            ))

    if visual_cfg['sparse_pointcloud']:
        visualize_sparse_pointcloud(dataset)

    if visual_cfg['optical_registration']:
        MatchVisualizer(dataset).run()