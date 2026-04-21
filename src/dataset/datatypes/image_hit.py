from typing import Self, Dict, Any

from .datatype import Datatype


class ImageHit(Datatype):
    headers = ["timestamp", "u_idx", "v_idx", "ping_idx", "bin_idx", "channel_idx", "distance", "incidence_angle"]

    def __init__(self,
        timestamp: int,
        u_idx: int,
        v_idx: int,
        ping_idx: int,
        bin_idx: int,
        channel_idx: int
    ):
        self.timestamp = timestamp
        self.u_idx = u_idx
        self.v_idx = v_idx
        self.ping_idx = ping_idx
        self.bin_idx = bin_idx
        self.channel_idx = channel_idx

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return ImageHit(
            timestamp=int(data["timestamp"]),
            u_idx=int(data["u_idx"]),
            v_idx=int(data["v_idx"]),
            ping_idx=int(data["ping_idx"]),
            bin_idx=int(data["bin_idx"]),
            channel_idx=int(data["channel_idx"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "u_idx": self.u_idx,
            "v_idx": self.v_idx,
            "ping_idx": self.ping_idx,
            "bin_idx": self.bin_idx,
            "channel_idx": self.channel_idx
        }