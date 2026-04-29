from typing import Self, Dict, Any

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class ImageHit(Datatype):
    headers = AcousticHit.headers + ["u", "v"]

    def __init__(self,
        hit: AcousticHit,
        u: int,
        v: int
    ):
        self.hit = hit
        self.u = u
        self.v = v

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return ImageHit(
            hit=AcousticHit.from_dict(data),
            u=int(data["u"]),
            v=int(data["v"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "u": self.u,
            "v": self.v
        }
