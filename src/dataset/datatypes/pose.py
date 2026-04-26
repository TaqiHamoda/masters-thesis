import numpy as np
from scipy.spatial.transform import Rotation as R
import pycolmap

from typing import Self, Dict, Any

from .datatype import Datatype


class Pose(Datatype):
    headers = ["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]

    def __init__(self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def get_position(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def get_quaternion(self) -> np.ndarray:
        """Returns quaternion in [x, y, z, w] format"""
        return np.array((self.qx, self.qy, self.qz, self.qw))

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns rotation matrix which transforms point from body frame to NED frame"""
        # scipy expects [x, y, z, w]
        return R.from_quat(self.get_quaternion()).as_matrix()

    def rotate(self, quat: np.ndarray) -> Self:
        """Expects a quaternion in [x, y, z, w] format. Rotates the pose by the given quaternion and returns a new Pose."""
        new_quat = (R.from_quat(quat) * R.from_quat(self.get_quaternion())).as_quat()
        return Pose(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            z=self.z,
            qx=new_quat[0],
            qy=new_quat[1],
            qz=new_quat[2],
            qw=new_quat[3],
        )

    def translate(self, local_delta: np.ndarray) -> Self:
        body_R_ned = self.get_rotation_matrix()

        # Transform into base frame
        global_delta = body_R_ned @ local_delta

        return Pose(
            timestamp=self.timestamp,
            x=self.x + global_delta[0],
            y=self.y + global_delta[1],
            z=self.z + global_delta[2],
            qx=self.qx,
            qy=self.qy,
            qz=self.qz,
            qw=self.qw,
        )

    @staticmethod
    def from_pycolmap(timestamp: int, rigid3d: pycolmap.Rigid3d) -> Self:
        return Pose(
            timestamp=timestamp,
            x=rigid3d.params[4],
            y=rigid3d.params[5],
            z=rigid3d.params[6],
            qx=rigid3d.params[0],
            qy=rigid3d.params[1],
            qz=rigid3d.params[2],
            qw=rigid3d.params[3],
        )

    def to_pycolmap(self) -> pycolmap.Rigid3d:
        return pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(self.get_quaternion().reshape(4, 1)),
            translation=self.get_position().reshape(3, 1)
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return Pose(
            timestamp=int(data["timestamp"]),
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            qx=float(data["qx"]),
            qy=float(data["qy"]),
            qz=float(data["qz"]),
            qw=float(data["qw"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }