from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class AcousticHit(Datatype):
    headers = Pose.headers + ["ping_idx", "bin_idx", "distance", "incidence_angle", "p_x", "p_y", "p_z"]

    def __init__(self,
        pose: Pose,
        ping_idx: int,
        bin_idx: int,
        distance: float,
        incidence_angle: float,
        p_x: float,
        p_y: float,
        p_z: float,
    ):
        self.pose = pose
        self.ping_idx = ping_idx
        self.bin_idx = bin_idx
        self.distance = distance
        self.incidence_angle = incidence_angle
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return AcousticHit(
            pose=Pose.from_dict(data),
            ping_idx=int(data["ping_idx"]),
            bin_idx=int(data["bin_idx"]),
            distance=float(data["distance"]),
            incidence_angle=float(data["incidence_angle"]),
            p_x=float(data["p_x"]),
            p_y=float(data["p_y"]),
            p_z=float(data["p_z"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "ping_idx": self.ping_idx,
            "bin_idx": self.bin_idx,
            "distance": self.distance,
            "incidence_angle": self.incidence_angle,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "p_z": self.p_z
        }