import cv2, pickle, numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

import csv
from tabulate import tabulate
from pathlib import Path
from typing import Self, List, Tuple, Dict, Any

SONARTYPE_PORT_SIDESCAN = 1
SONARTYPE_STARBOARD_SIDESCAN = 2


class Pose:
    def __init__(self,
        timestamp: int,
        x: float,
        y: float,
        z: float,
        qw: float,
        qx: float,
        qy: float,
        qz: float
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qz = qz

    def __str__(self):
        return f"Pose(timestamp={self.timestamp}, x={self.x}, y={self.y}, z={self.z}, qw={self.qw}, qx={self.qx}, qy={self.qy}, qz={self.qz})"


class Image:
    headers = ("timestamp", "x", "y", "z", "qw", "qx", "qy", "qz", "fx", "fy", "cx", "cy")

    def __init__(self,
        pose: Pose,
        fx: float,
        fy: float,
        cx: float,
        cy: float
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
            pose=Pose(
                int(data["timestamp"]),
                float(data["x"]),
                float(data["y"]),
                float(data["z"]),
                float(data["qw"]),
                float(data["qx"]),
                float(data["qy"]),
                float(data["qz"])
            ),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.pose.timestamp,
            "x": self.pose.x,
            "y": self.pose.y,
            "z": self.pose.z,
            "qw": self.pose.qw,
            "qx": self.pose.qx,
            "qy": self.pose.qy,
            "qz": self.pose.qz,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy
        }


class SideScanSonar:
    headers = ("timestamp", "num_samples", "slant_range", "bin_size", "east", "north", "altitude", "roll", "pitch", "yaw")

    def __init__(self,
        timestamp: int,
        num_samples: int,
        slant_range: float,
        east: float,
        north: float,
        altitude: float,
        roll: float,
        pitch: float,
        yaw: float
    ):
        self.timestamp = timestamp
        self.num_samples = num_samples
        self.slant_range = slant_range
        self.east = east
        self.north = north
        self.altitude = altitude
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        self.bin_size = self.slant_range / self.num_samples

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return SideScanSonar(
            int(data["timestamp"]),
            int(data["num_samples"]),
            float(data["slant_range"]),
            float(data["east"]),
            float(data["north"]),
            float(data["altitude"]),
            float(data["roll"]),
            float(data["pitch"]),
            float(data["yaw"])
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "num_samples": self.num_samples,
            "slant_range": self.slant_range,
            "bin_size": self.bin_size,
            "east": self.east,
            "north": self.north,
            "altitude": self.altitude,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw
        }


class Dataset:
    def __init__(self,
        data_path: str,
        output_path: str,
        img_topic: str,
        odo_topic: str,
        info_topic: str,
        sonar_topic: str,
        nav_topic: str
    ):
        self.data_path = Path(data_path)
        self.msg_path = self.data_path / "msgs"
        self.bag_paths = [bag for bag in self.data_path.glob("*.bag")]

        self.typestore = get_typestore(Stores.EMPTY)  # ROS1 Store
        self.register_msgs()

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.cameras_csv = self.output_path / "camera_poses.csv"
        self.sonar_csv = self.output_path / "sonar_data.csv"
        self.sonar_file = self.output_path / "sonar.pkl"

        self.image_dir = self.output_path / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.img_topic = img_topic
        self.odo_topic = odo_topic
        self.info_topic = info_topic
        self.sonar_topic = sonar_topic
        self.nav_topic = nav_topic

        self.images: Dict[str, Image] = {}
        self.sonar: Dict[str, SideScanSonar] = {}

    def exists(self) -> bool:
        return self.cameras_csv.exists() and\
            self.sonar_csv.exists() and\
            self.sonar_file.exists() and\
            self.image_dir.exists() and\
            self.image_dir.is_dir()

    def _recurse_dir(self, path: Path) -> List[Path]:
        files = []
        for item in path.iterdir():
            if item.is_file():
                files.append(item)
            elif item.is_dir():
                files.extend(self._recurse_dir(item))

        return files

    def register_msgs(self):
        suffix = '/'.join(self.msg_path.parts) + '/'

        add_types = {}
        for msg in self._recurse_dir(self.msg_path):
            msgname = '/'.join(msg.parts).replace(suffix, '').replace(".msg", '')
            msgdef = msg.read_text(encoding='utf-8')

            add_types.update(get_types_from_msg(msgdef, msgname))

        self.typestore.register(add_types)

    def export_data(self):
        with open(self.cameras_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Image.headers)
            writer.writeheader()
            writer.writerows([image.to_dict() for image in self.images.values()])

        with open(self.sonar_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SideScanSonar.headers)
            writer.writeheader()
            writer.writerows([sonar.to_dict() for sonar in self.sonar.values()])

    def load_data_from_csv(self):
        with open(self.cameras_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = Image.from_dict(row)
                self.images[image.pose.timestamp] = image

        with open(self.sonar_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sonar = SideScanSonar.from_dict(row)
                self.sonar[sonar.timestamp] = sonar

    def load_data_from_bags(self):
        camera_params = None
        odometry_data: List[Tuple[int, Pose]] = []
        images_metadata: List[int] = []
        sonar_data: List[Tuple[int, int, float]] = []
        acoustic_data: List[Tuple[int, List[float], List[float]]] = []
        nav_data: List[Tuple[int, float, float, float, float, float, float]] = []

        with AnyReader(self.bag_paths, default_typestore=self.typestore) as reader:
            # Filter connections once for performance
            connections = [
                c for c in reader.connections if c.topic in
                (self.img_topic, self.odo_topic, self.info_topic, self.sonar_topic, self.nav_topic)
            ]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # The reader now has its own deserialize method
                msg = reader.deserialize(rawdata, connection.msgtype)

                if connection.topic == self.img_topic:
                    images_metadata.append(timestamp)

                    filepath = self.image_dir / f"{timestamp}.jpg"
                    if filepath.exists():
                        continue

                    # Decompress image data
                    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imwrite(str(filepath), img)
                elif connection.topic == self.odo_topic:
                    p = msg.pose.pose
                    odometry_data.append((
                        timestamp,
                        Pose(
                            timestamp,
                            p.position.x,
                            p.position.y,
                            p.position.z,
                            p.orientation.w,
                            p.orientation.x,
                            p.orientation.y,
                            p.orientation.z
                        )
                    ))
                elif connection.topic == self.info_topic and camera_params is None:
                    # K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                    camera_params = msg.K
                elif connection.topic == self.sonar_topic:
                    speed_of_sound = msg.sonar_ping.f_speed_of_sound
                    f_range_ms = msg.sonar_samples[0].sonar_ping_channel.f_range_ms
                    f_range_delay_ms = msg.sonar_samples[0].sonar_ping_channel.f_range_delay_ms

                    slant_range = (f_range_ms + f_range_delay_ms) * speed_of_sound / 2000.0

                    port_intensities, stbd_intensities = None, None
                    for i in range(len(msg.sonar_samples)):
                        width = msg.sonar_samples[i].sonar_ping_channel.w_samples
                        if msg.sonar_samples[i].sonar_ping_channel.w_sonar_type == SONARTYPE_PORT_SIDESCAN:
                            port_intensities = msg.sonar_samples[i].data[:width][::-1]  # Reverse port side for correct left-to-right order
                        elif msg.sonar_samples[i].sonar_ping_channel.w_sonar_type == SONARTYPE_STARBOARD_SIDESCAN:
                            stbd_intensities = msg.sonar_samples[i].data[:width]

                    sonar_data.append((
                        timestamp,
                        len(port_intensities),
                        slant_range
                    ))

                    acoustic_data.append((
                        timestamp,
                        port_intensities,
                        stbd_intensities
                    ))
                elif connection.topic == self.nav_topic:
                    nav_data.append((
                        timestamp,
                        msg.position.east,
                        msg.position.north,
                        msg.altitude,
                        msg.orientation.roll,
                        msg.orientation.pitch,
                        msg.orientation.yaw
                    ))

        if not odometry_data or not images_metadata or not sonar_data or not nav_data:
            raise ValueError("Error: Missing data in topics. Check your bag topic names.")

        with open(self.sonar_file, "wb") as f:
            pickle.dump(acoustic_data, f)

        odo_timestamps = np.array([o[0] for o in odometry_data])
        for img_ts in images_metadata:
            idx = (np.abs(odo_timestamps - img_ts)).argmin()
            matched_pose = odometry_data[idx][1]
            matched_pose.timestamp = img_ts  # Update timestamp to match image

            image = Image(
                pose=matched_pose,
                fx=camera_params[0],
                fy=camera_params[4],
                cx=camera_params[2],
                cy=camera_params[5]
            )
            self.images[image.pose.timestamp] = image

        nav_timestamps = np.array([n[0] for n in nav_data])
        for sonar_ts, num_samples, slant_range in sonar_data:
            idx = (np.abs(nav_timestamps - sonar_ts)).argmin()
            matched_nav = nav_data[idx]

            sonar = SideScanSonar(
                timestamp=sonar_ts,
                num_samples=num_samples,
                slant_range=slant_range,
                east=matched_nav[1],
                north=matched_nav[2],
                altitude=matched_nav[3],
                roll=matched_nav[4],
                pitch=matched_nav[5],
                yaw=matched_nav[6]
            )
            self.sonar[sonar.timestamp] = sonar

    def data_stats(self):
        min_x, min_y, min_z, min_alt = np.inf, np.inf, np.inf, np.inf
        max_x, max_y, max_z, max_alt = -np.inf, -np.inf, -np.inf, -np.inf

        for sonar in self.sonar.values():
            if sonar.altitude > max_alt: max_alt = sonar.altitude
            elif sonar.altitude < min_alt: min_alt = sonar.altitude

        for image in self.images.values():
            if image.pose.x > max_x: max_x = image.pose.x
            elif image.pose.x < min_x: min_x = image.pose.x

            if image.pose.y > max_y: max_y = image.pose.y
            elif image.pose.y < min_y: min_y = image.pose.y

            if image.pose.z > max_z: max_z = image.pose.z
            elif image.pose.z < min_z: min_z = image.pose.z

        with open(self.output_path / "data_stats.txt", "w") as f:
            print(f"X range: {min_x:.2f} to {max_x:.2f} meters (span: {max_x - min_x:.2f})", file=f)
            print(f"Y range: {min_y:.2f} to {max_y:.2f} meters (span: {max_y - min_y:.2f})", file=f)
            print(f"Z range: {min_z:.2f} to {max_z:.2f} meters (span: {max_z - min_z:.2f})", file=f)
            print(f"Altitude range: {min_alt:.2f} to {max_alt:.2f} meters (span: {max_alt - min_alt:.2f})", file=f)
            print(f"Area Covered: {(max_x - min_x) * (max_y - min_y):.2f} square meters", file=f)

    def inspect_bags(self):
        topics_info = set()
        with AnyReader(self.bag_paths) as reader:
            for connection in reader.connections:
                topics_info.add((
                    connection.topic,
                    connection.msgtype,
                    connection.msgcount
                ))

        # Sort by topic name for readability
        topics_info = sorted(topics_info, key=lambda x: x[0])

        # Display the results
        headers = ["Topic Name", "Data Type (Message)", "Message Count"]
        with open(self.output_path / "topics_info.txt", "w") as f:
            print(tabulate(topics_info, headers=headers, tablefmt="grid"), file=f)
