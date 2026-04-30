import yaml
from time import perf_counter, sleep

from src.dataset import Dataset
from src.photogrammetry import Photogrammetry
from src.sonar import export_to_xtf, export_to_png
from src.registration import Registration
from src.decomposition import Decomposition
from src.visualization import MatchVisualizer, VertexVisualizer


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
    registration_cfg = cfg['registration']
    decomposition_cfg = cfg['decomposition']
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

    if registration_cfg['enabled']:
        print("Performing optical registration...")
        registration = Registration(
            dataset,
            sonar_offset=extrinsics_cfg['sonar_offset'],
            thickness=registration_cfg['plane_thickness'],
            n_local=registration_cfg['normal_vector'],
            num_threads=registration_cfg['num_threads'],
        )
        registration.save_matches()
        registration.save_vertices()

    if decomposition_cfg['enabled']:
        decomposition = Decomposition(dataset)
        if not dataset.sonar_angles.exists() or not dataset.sonar_reflectivity.exists():
            print("Performing component decomposition...")
            decomposition.process_decomposition()

        decomposition.print_stats()

        print("Saving reflectivity image...")
        decomposition.save_reflectivity_image(
            lower=decomposition_cfg['lower_percentile'],
            upper=decomposition_cfg['upper_percentile']
        )

    if visual_cfg['optical_matching']:
        MatchVisualizer(dataset).run()

    if visual_cfg['vertex_matching']:
        VertexVisualizer(dataset).run()