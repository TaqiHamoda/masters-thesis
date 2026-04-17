from scipy.spatial.transform import Rotation as R
import numpy as np

from typing import Self, Dict, Any


class Pose:
    headers = ["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]

    def __init__(self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def translate(self, local_delta: np.ndarray) -> Self:
        # scipy expects [x, y, z, w]
        rot = R.from_quat([self.qx, self.qy, self.qz, self.qw])

        # Transform into base frame
        global_delta = rot.apply(local_delta)

        return Pose(
            self.timestamp,
            self.x + global_delta[0],
            self.y + global_delta[1],
            self.z + global_delta[2],
            self.qw,
            self.qx,
            self.qy,
            self.qz
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return Pose(
            int(data["timestamp"]),
            float(data["x"]),
            float(data["y"]),
            float(data["z"]),
            float(data["qw"]),
            float(data["qx"]),
            float(data["qy"]),
            float(data["qz"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qw": self.qw,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz
        }