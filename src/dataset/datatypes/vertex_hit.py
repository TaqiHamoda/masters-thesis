from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class VertexHit(Datatype):
    headers = Pose.headers + ["vertex_idx", "channel_idx", "distance", "incidence_angle"]

    def __init__(self,
        pose: Pose,
        vertex_idx: int,
        channel_idx: int,
        distance: float,
        incidence_angle: float
    ):
        self.pose = pose
        self.vertex_idx = vertex_idx
        self.channel_idx = channel_idx
        self.distance = distance
        self.incidence_angle = incidence_angle

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return VertexHit(
            pose=Pose.from_dict(data),
            vertex_idx=int(data["vertex_idx"]),
            channel_idx=int(data["channel_idx"]),
            distance=float(data["distance"]),
            incidence_angle=float(data["incidence_angle"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "vertex_idx": self.vertex_idx,
            "channel_idx": self.channel_idx,
            "distance": self.distance,
            "incidence_angle": self.incidence_angle
        }