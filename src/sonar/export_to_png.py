import numpy as np
import cv2
from pyxtf import xtf_read, concatenate_channel, XTFHeaderType

from ..dataset import Dataset
from .utils import get_sample_dtype


def export_to_png(dataset: Dataset, sample_dtype: str = "uint8"):
    # source: https://github.com/oysstu/pyxtf/blob/master/examples/xtf_to_image.py

    (fh, p) = xtf_read(str(dataset.sonar_xtf))

    sample_datatype = get_sample_dtype(sample_dtype)
    upper_limit = np.iinfo(sample_datatype).max

    if XTFHeaderType.sonar not in p:
        raise ValueError(f"Invalid or corrupt XTF file provided: {dataset.sonar_xtf}")

    # Toggle concatenate_channel weighted argument to fit your data requirements.
    port = concatenate_channel(p[XTFHeaderType.sonar], file_header=fh, channel=0, weighted=False)
    stbd = concatenate_channel(p[XTFHeaderType.sonar], file_header=fh, channel=1, weighted=False)

    waterfall = np.hstack((port, stbd))

    # Clip to range (max cannot be used due to outliers)
    # More robust methods are possible (through histograms / statistical outlier removal)
    waterfall = np.clip(waterfall, 0, upper_limit - 1)

    # The sonar data is logarithmic (dB), add small value to avoid log10(0)
    waterfall = np.log10(waterfall + 1, dtype=np.float32)

    # Need to find minimum and maximum value for scaling
    vmin = waterfall.min()
    vmax = waterfall.max()

    # Scaling values to fit uint8
    waterfall = ((waterfall - vmin) / (vmax - vmin)) * 255
    waterfall = np.clip(waterfall, 0, 255)

    cv2.imwrite(str(dataset.sonar_png), waterfall.astype(np.uint8))
