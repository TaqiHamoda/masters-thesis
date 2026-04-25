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

    def get_quaternion(self) -> np.ndarray:
        """Returns quaternion in [x, y, z, w] format"""
        return np.array((self.qx, self.qy, self.qz, self.qw))

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns rotation matrix which transforms point from body frame to NED frame"""
        # scipy expects [x, y, z, w]
        return R.from_quat(self.get_quaternion()).as_matrix()

    def rotate(self, quat: np.ndarray) -> Self:
        """Expects a quaternion in [x, y, z, w] format. Rotates the pose by the given quaternion and returns a new Pose."""
        new_quat = R.from_quat(self.get_quaternion()) * R.from_quat(quat)
        return Pose(
            self.timestamp,
            self.x,
            self.y,
            self.z,
            qw=new_quat.as_quat()[3],
            qx=new_quat.as_quat()[0],
            qy=new_quat.as_quat()[1],
            qz=new_quat.as_quat()[2]
        )

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
    def from_pycolmap(timestamp: int, rigid3d: pycolmap.Rigid3d) -> Self:
        return Pose(
            timestamp,
            qx=rigid3d.params[0],
            qy=rigid3d.params[1],
            qz=rigid3d.params[2],
            qw=rigid3d.params[3],
            x=rigid3d.params[4],
            y=rigid3d.params[5],
            z=rigid3d.params[6],
        )

    def to_pycolmap(self) -> pycolmap.Rigid3d:
        return pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(self.get_quaternion().reshape(4, 1)),
            translation=self.get_position().reshape(3, 1)
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