from pathlib import Path
from typing import List, Dict, Any, Self

from .datatype import Datatype
from .acoustic_hit import AcousticHit


class DeltaHit(Datatype):
    headers = AcousticHit.headers + ["delta_ping", "delta_bin"]

    def __init__(
        self,
        hit: AcousticHit,
        delta_ping: int,
        delta_bin: int
    ):
        self.hit = hit
        self.delta_ping = delta_ping
        self.delta_bin = delta_bin

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return DeltaHit(
            hit=AcousticHit.from_dict(data),
            delta_ping=int(data["delta_ping"]),
            delta_bin=int(data["delta_bin"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.hit.to_dict() | {
            "delta_ping": self.delta_ping,
            "delta_bin": self.delta_bin
        }

    @staticmethod
    def from_csv(csv_file: str | Path) -> List[Self]:
        return Datatype._from_csv(csv_file, DeltaHit.from_dict)