from scipy.spatial.transform import Rotation as R
import numpy as np

from typing import Self, Dict, Any

from .datatype import Datatype


class Pose(Datatype):
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

    def get_position(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns rotation matrix which transforms point from body frame to NED frame"""
        # scipy expects [x, y, z, w]
        return R.from_quat([self.qx, self.qy, self.qz, self.qw]).as_matrix()

    def translate(self, local_delta: np.ndarray) -> Self:
        body_R_ned = self.get_rotation_matrix()

        # Transform into base frame
        global_delta = body_R_ned @ local_delta

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