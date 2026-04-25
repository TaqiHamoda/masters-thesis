from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class ImageHit(Datatype):
    headers = Pose.headers + ["u", "v", "p_x", "p_y", "p_z", "ping_idx", "bin_idx", "distance", "incidence_angle"]

    def __init__(self,
        pose: Pose,
        u: int,
        v: int,
        p_x: float,
        p_y: float,
        p_z: float,
        ping_idx: int,
        bin_idx: int,
        distance: float,
        incidence_angle: float
    ):
        self.pose = pose
        self.u = u
        self.v = v
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        self.ping_idx = ping_idx
        self.bin_idx = bin_idx
        self.distance = distance
        self.incidence_angle = incidence_angle

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return ImageHit(
            pose=Pose.from_dict(data),
            u=int(data["u"]),
            v=int(data["v"]),
            p_x=float(data["p_x"]),
            p_y=float(data["p_y"]),
            p_z=float(data["p_z"]),
            ping_idx=int(data["ping_idx"]),
            bin_idx=int(data["bin_idx"]),
            distance=float(data["distance"]),
            incidence_angle=float(data["incidence_angle"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "u": self.u,
            "v": self.v,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "p_z": self.p_z,
            "ping_idx": self.ping_idx,
            "bin_idx": self.bin_idx,
            "distance": self.distance,
            "incidence_angle": self.incidence_angle
        }