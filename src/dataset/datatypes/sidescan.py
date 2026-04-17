from scipy.spatial.transform import Rotation as R
import numpy as np

from typing import Self, Tuple, Dict, Any

from .navigation import Navigation


class SideScanSonar:
    headers = Navigation.headers + ["num_samples", "slant_range", "delay_range", "bin_size", "frequency", "speed_of_sound"]

    def __init__(self,
        navigation: Navigation,
        num_samples: int,
        slant_range: float,
        delay_range: float,
        frequency: int,
        speed_of_sound: float,
    ):
        self.navigation = navigation
        self.num_samples = num_samples
        self.slant_range = slant_range
        self.delay_range = delay_range
        self.frequency = frequency
        self.speed_of_sound = speed_of_sound

        self.bin_size = self.slant_range / self.num_samples

    def get_plane(self, n_local: np.ndarray = np.array([1.0, 0.0, 0.0])) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns point and normal vector to define plane.

        Default local normal vector assumes YZ plane fan (standard for Side-Scan Sonar).
        """
        pose = self.navigation.pose

        pos = np.array([pose.x, pose.y, pose.z])
        rot = R.from_quat([pose.qx, pose.qy, pose.qz, pose.qw])

        # Rotate the local normal vector to the global frame
        n_global = rot.apply(n_local)

        return pos, n_global

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return SideScanSonar(
            navigation=Navigation.from_dict(data),
            num_samples=int(data["num_samples"]),
            slant_range=float(data["slant_range"]),
            delay_range=float(data["delay_range"]),
            frequency=int(data["frequency"]),
            speed_of_sound=float(data["speed_of_sound"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.navigation.to_dict() | {
            "num_samples": self.num_samples,
            "slant_range": self.slant_range,
            "delay_range": self.delay_range,
            "bin_size": self.bin_size,
            "frequency": self.frequency,
            "speed_of_sound": self.speed_of_sound
        }