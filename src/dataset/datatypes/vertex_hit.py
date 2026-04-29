from pathlib import Path
from typing import Self, List, Dict, Any

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class VertexHit(Datatype):
    headers = AcousticHit.headers + ["vertex_idx"]

    def __init__(self,
        hit: AcousticHit,
        vertex_idx: int
    ):
        self.hit = hit
        self.vertex_idx = vertex_idx

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return VertexHit(
            hit=AcousticHit.from_dict(data),
            vertex_idx=int(data["vertex_idx"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "vertex_idx": self.vertex_idx
        }

    @staticmethod
    def from_csv(csv_file: str | Path) -> List[Self]:
        return Datatype._from_csv(csv_file, VertexHit.from_dict)