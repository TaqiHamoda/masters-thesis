import cv2
import numpy as np

from ..dataset import Dataset


def export_png(dataset: Dataset):
    waterfall = np.load(dataset.sonar_file)['data'].astype(np.float32)

    # Remove Outliers
    waterfall = np.clip(
        waterfall,
        np.percentile(waterfall, 1),
        np.percentile(waterfall, 99),
    )

    waterfall = np.log1p(waterfall)         # Sonar data is log-scaled. Compress dynamic range
    waterfall = np.flip(waterfall, axis=0)  # Flip vertically to match standard for sonar imagery

    # Need to find minimum and maximum value for scaling
    vmin = np.min(waterfall)
    vmax = np.max(waterfall)

    # Scaling values to fit uint8
    waterfall = 255 * (waterfall - vmin) / (vmax - vmin)

    cv2.imwrite(str(dataset.sonar_png), waterfall.astype(np.uint8))
