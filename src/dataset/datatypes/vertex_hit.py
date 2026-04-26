from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class VertexHit(Datatype):
    headers = Pose.headers + ["vertex_idx", "v_x", "v_y", "v_z", "ping_idx", "bin_idx", "distance", "incidence_angle"]

    def __init__(self,
        pose: Pose,
        vertex_idx: int,
        v_x: float,
        v_y: float,
        v_z: float,
        ping_idx: int,
        bin_idx: int,
        distance: float,
        incidence_angle: float
    ):
        self.pose = pose
        self.vertex_idx = vertex_idx
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.ping_idx = ping_idx
        self.bin_idx = bin_idx
        self.distance = distance
        self.incidence_angle = incidence_angle

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return VertexHit(
            pose=Pose.from_dict(data),
            vertex_idx=int(data["vertex_idx"]),
            v_x=float(data["v_x"]),
            v_y=float(data["v_y"]),
            v_z=float(data["v_z"]),
            ping_idx=int(data["ping_idx"]),
            bin_idx=int(data["bin_idx"]),
            distance=float(data["distance"]),
            incidence_angle=float(data["incidence_angle"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "vertex_idx": self.vertex_idx,
            "v_x": self.v_x,
            "v_y": self.v_y,
            "v_z": self.v_z,
            "ping_idx": self.ping_idx,
            "bin_idx": self.bin_idx,
            "distance": self.distance,
            "incidence_angle": self.incidence_angle
        }