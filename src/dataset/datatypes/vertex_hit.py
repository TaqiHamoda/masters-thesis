from typing import Self, Dict, Any

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class VertexHit(Datatype):
    headers = AcousticHit.headers + ["vertex_idx", "p_x", "p_y", "p_z"]

    def __init__(self,
        hit: AcousticHit,
        vertex_idx: int,
        p_x: float,
        p_y: float,
        p_z: float
    ):
        self.hit = hit
        self.vertex_idx = vertex_idx
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return VertexHit(
            hit=AcousticHit.from_dict(data),
            vertex_idx=int(data["vertex_idx"]),
            p_x=float(data["p_x"]),
            p_y=float(data["p_y"]),
            p_z=float(data["p_z"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "vertex_idx": self.vertex_idx,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "p_z": self.p_z
        }