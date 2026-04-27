from typing import Self, Dict, Any

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class ImageHit(Datatype):
    headers = AcousticHit.headers + ["u", "v", "p_x", "p_y", "p_z"]

    def __init__(self,
        hit: AcousticHit,
        u: int,
        v: int,
        p_x: float,
        p_y: float,
        p_z: float
    ):
        self.hit = hit
        self.u = u
        self.v = v
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return ImageHit(
            hit=AcousticHit.from_dict(data),
            u=int(data["u"]),
            v=int(data["v"]),
            p_x=float(data["p_x"]),
            p_y=float(data["p_y"]),
            p_z=float(data["p_z"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "u": self.u,
            "v": self.v,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "p_z": self.p_z
        }