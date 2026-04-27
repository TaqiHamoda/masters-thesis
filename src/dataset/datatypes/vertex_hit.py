from typing import Self, Dict, Any

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class VertexHit(Datatype):
    headers = AcousticHit.headers + ["vertex_idx", "v_x", "v_y", "v_z"]

    def __init__(self,
        hit: AcousticHit,
        vertex_idx: int,
        v_x: float,
        v_y: float,
        v_z: float
    ):
        self.hit = hit
        self.vertex_idx = vertex_idx
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return VertexHit(
            hit=AcousticHit.from_dict(data),
            vertex_idx=int(data["vertex_idx"]),
            v_x=float(data["v_x"]),
            v_y=float(data["v_y"]),
            v_z=float(data["v_z"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "vertex_idx": self.vertex_idx,
            "v_x": self.v_x,
            "v_y": self.v_y,
            "v_z": self.v_z
        }