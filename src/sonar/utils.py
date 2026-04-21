import numpy as np
import pyxtf


def get_sample_dtype(sample_dtype: str) -> np.dtype:
    if sample_dtype == "uint8":
        return np.uint8
    elif sample_dtype == "uint16":
        return np.uint16
    elif sample_dtype == "uint32":
        return np.uint32
    elif sample_dtype == "float32":
        return np.float32

    raise ValueError(f"Invalid sample dtype: {sample_dtype}")


def get_sample_format(sample_dtype: str):
    if sample_dtype == "uint8":
        return pyxtf.XTFSampleFormat.byte.value
    elif sample_dtype == "uint16":
        return pyxtf.XTFSampleFormat.word.value
    elif sample_dtype == "uint32":
        return pyxtf.XTFSampleFormat.int.value
    elif sample_dtype == "float32":
        return pyxtf.XTFSampleFormat.float.value

    raise ValueError(f"Invalid sample dtype: {sample_dtype}")