from time import perf_counter, sleep

from src.dataset import Dataset
from src.photogrammetry import Photogrammetry

img_topic = '/girona1000/flir_spinnaker_camera/image_raw/compressed'
odo_topic = '/girona1000/navigator/odometry'
info_topic = '/girona1000/flir_spinnaker_camera/camera_info'

perform_dense_reconstruction = False


if __name__ == "__main__":
    dataset = Dataset(
        data_path="data/rosbags/",
        output_path="output/",
        img_topic=img_topic,
        odo_topic=odo_topic,
        info_topic=info_topic
    )

    if dataset.csv_file.exists():
        dataset.load_data_from_csv()
    else:
        dataset.load_data_from_bags()
        dataset.export_data()
        dataset.data_stats()
        dataset.inspect_bags()

    photogrammetry = Photogrammetry(dataset, output_path="output/")

    if not photogrammetry.has_cuda():
        print("CUDA not available. Consider installing COLMAP with CUDA support for faster processing.")
        sleep(5)

    if photogrammetry.database_path.exists():
        print("Database already exists. Skipping feature extraction and matching.")
    else:
        print("Extracting and matching features")

        start_time = perf_counter()
        photogrammetry.extract_and_match_features(
            contrast_threshold=0.002,
            pos_std=(2.0, 2.0, 0.1),
            max_distance=5.0
        )
        end_time = perf_counter()

        print(f"Feature extraction and matching completed in {end_time - start_time:.2f} seconds.")

    if photogrammetry.sparse_path.exists():
        print("Sparse reconstruction already exists. Skipping...")
    else:
        print("Performing sparse reconstruction")

        start_time = perf_counter()
        photogrammetry.bundle_adjustment()
        photogrammetry.prune_poses()
        end_time = perf_counter()

        print(f"Sparse reconstruction completed in {end_time - start_time:.2f} seconds.")

    if not perform_dense_reconstruction:
        exit()

    if photogrammetry.mvs_path.exists():
        print("Stereo matching already exists. Skipping...")
    else:
        print("Performing stereo matching")

        start_time = perf_counter()
        photogrammetry.stereo_matching()
        end_time = perf_counter()

        print(f"Stereo matching completed in {end_time - start_time:.2f} seconds.")

    if photogrammetry.dense_ply.exists():
        print("Dense reconstruction already exists. Skipping...")
    else:
        print("Performing dense reconstruction")

        start_time = perf_counter()
        photogrammetry.dense_reconstruction(
            max_image_size=2000,
            check_num_images=10,
            cache_size=32
        )
        end_time = perf_counter()

        print(f"Dense reconstruction completed in {end_time - start_time:.2f} seconds.")

    photogrammetry.create_mesh()
