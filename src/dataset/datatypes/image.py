from pathlib import Path
from typing import Self, List, Dict, Any

from .datatype import Datatype
from .pose import Pose


class Image(Datatype):
    headers = Pose.headers + ["fx", "fy", "cx", "cy"]

    def __init__(self,
        pose: Pose,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        self.pose = pose

        self.filename = f"{pose.timestamp}.jpg"

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return Image(
            pose=Pose.from_dict(data),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"])
        )


    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy
        }

    @staticmethod
    def from_csv(csv_file: str | Path) -> List[Self]:
        return Datatype._from_csv(csv_file, Image.from_dict)
