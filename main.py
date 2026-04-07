import yaml
from time import perf_counter, sleep

from src.dataset import Dataset
from src.photogrammetry import Photogrammetry


if __name__ == "__main__":
    # Load settings from the YAML file
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Extract sections for easier access
    p_cfg = cfg['paths']
    t_cfg = cfg['topics']
    algo_cfg = cfg['photogrammetry']

    dataset = Dataset(
        data_path=p_cfg['data_path'],
        output_path=p_cfg['output_path'],
        img_topic=t_cfg['img_topic'],
        odo_topic=t_cfg['odo_topic'],
        info_topic=t_cfg['info_topic'],
        sonar_topic=t_cfg['sonar_topic'],
        nav_topic=t_cfg['nav_topic']
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

    photogrammetry = Photogrammetry(dataset, output_path=p_cfg['output_path'])

    if not photogrammetry.has_cuda():
        print("CUDA not available. Consider installing COLMAP with CUDA support.")
        sleep(5)

    # Feature Extraction
    if not photogrammetry.database_path.exists():
        print("Extracting and matching features...")
        start_time = perf_counter()
        photogrammetry.extract_and_match_features(**algo_cfg['features'])
        print(f"Features completed in {perf_counter() - start_time:.2f}s")

    # Sparse Reconstruction
    if not photogrammetry.sparse_path.exists():
        print("Performing sparse reconstruction...")
        start_time = perf_counter()
        photogrammetry.sparse_reconstruction(**algo_cfg['sparse'])
        photogrammetry.prune_poses(**algo_cfg['pruning'])
        print(f"Sparse completed in {perf_counter() - start_time:.2f}s")

    # Exit if dense is not requested
    if not cfg['flags']['perform_dense_reconstruction']:
        print("Dense reconstruction disabled in config. Exiting.")
        exit()

    # Stereo Matching
    if not photogrammetry.mvs_path.exists():
        print(f"Performing stereo matching...")
        start_time = perf_counter()
        photogrammetry.stereo_matching()
        print(f"Stereo matching completed in {perf_counter() - start_time:.2f}s")

    # Dense Reconstruction
    if not photogrammetry.dense_ply.exists():
        print("Performing dense reconstruction...")
        start_time = perf_counter()
        photogrammetry.dense_reconstruction(**algo_cfg['dense'])
        print(f"Dense completed in {perf_counter() - start_time:.2f}s")

    photogrammetry.create_mesh()
