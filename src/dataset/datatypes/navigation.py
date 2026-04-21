from typing import Self, Dict, Any

from .datatype import Datatype
from .pose import Pose


class Navigation(Datatype):
    headers = Pose.headers + ["latitude", "longitude", "altitude", "roll", "pitch", "yaw", "speed"]

    def __init__(self,
        pose: Pose,
        latitude: float,
        longitude: float,
        altitude: float,
        roll: float,
        pitch: float,
        yaw: float,
        speed: float
    ):
        self.pose = pose
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.speed = speed

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return Navigation(
            pose=Pose.from_dict(data),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            altitude=float(data["altitude"]),
            roll=float(data["roll"]),
            pitch=float(data["pitch"]),
            yaw=float(data["yaw"]),
            speed=float(data["speed"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.pose.to_dict() | {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "speed": self.speed
        }