from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class AcousticHit(Datatype):
    headers = Pose.headers + ["ping_idx", "bin_idx", "distance", "incidence_angle"]

    def __init__(self,
        pose: Pose,
        ping_idx: int,
        bin_idx: int,
        distance: float,
        incidence_angle: float
    ):
        self.pose = pose
        self.ping_idx = ping_idx
        self.bin_idx = bin_idx
        self.distance = distance
        self.incidence_angle = incidence_angle

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return AcousticHit(
            pose=Pose.from_dict(data),
            ping_idx=int(data["ping_idx"]),
            bin_idx=int(data["bin_idx"]),
            distance=float(data["distance"]),
            incidence_angle=float(data["incidence_angle"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "ping_idx": self.ping_idx,
            "bin_idx": self.bin_idx,
            "distance": self.distance,
            "incidence_angle": self.incidence_angle
        }