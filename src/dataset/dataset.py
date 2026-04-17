import cv2, pickle, numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

import csv
from tabulate import tabulate
from pathlib import Path
from typing import List, Tuple, Dict

from .datatypes import Image, Pose, SideScanSonar, Navigation

SONARTYPE_PORT_SIDESCAN = 1
SONARTYPE_STARBOARD_SIDESCAN = 2


class Dataset:
    def __init__(self,
        data_path: str,
        output_path: str,
        img_topic: str,
        odo_topic: str,
        info_topic: str,
        sonar_topic: str,
        nav_topic: str,
        camera_trans: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        sonar_trans: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.data_path = Path(data_path)
        self.msg_path = self.data_path / "msgs"
        self.bag_paths = [bag for bag in self.data_path.glob("*.bag")]

        self.typestore = get_typestore(Stores.EMPTY)  # ROS1 Store
        self.register_msgs()

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.cameras_csv = self.output_path / "camera_poses.csv"
        self.sonar_csv = self.output_path / "sonar_poses.csv"
        self.sonar_file = self.output_path / "sonar.pkl"
        self.sonar_xtf = self.output_path / "sonar.xtf"

        self.image_dir = self.output_path / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.img_topic = img_topic
        self.odo_topic = odo_topic
        self.info_topic = info_topic
        self.sonar_topic = sonar_topic
        self.nav_topic = nav_topic

        self.images: Dict[int, Image] = {}
        self.sonar: Dict[int, SideScanSonar] = {}

        self.camera_trans = np.array(camera_trans)
        self.sonar_trans = np.array(sonar_trans)

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
                self.sonar[sonar.navigation.pose.timestamp] = sonar

    def load_data_from_bags(self):
        camera_params = None
        odometry_data: List[Tuple[int, Pose]] = []
        images_metadata: List[int] = []
        sonar_data: List[Tuple[int, int, float]] = []
        acoustic_data: Dict[int, Tuple[List[float], List[float]]] = {}
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
                    # D: [k1, k2, p1, p2, k3, k4, k5, k6]
                    fx, fy = msg.K[0], msg.K[4]
                    cx, cy = msg.K[2], msg.K[5]

                    camera_params = (fx, fy, cx, cy)
                elif connection.topic == self.sonar_topic:
                    speed_of_sound = msg.sonar_ping.f_speed_of_sound
                    f_range_ms = msg.sonar_samples[0].sonar_ping_channel.f_range_ms
                    f_range_delay_ms = msg.sonar_samples[0].sonar_ping_channel.f_range_delay_ms

                    delay_range = f_range_delay_ms * speed_of_sound / 1000.0
                    slant_range = (f_range_ms + f_range_delay_ms) * speed_of_sound / 1000.0

                    freq = 0
                    port_intensities, stbd_intensities = None, None
                    for i in range(len(msg.sonar_samples)):
                        freq = msg.sonar_samples[i].sonar_ping_channel.f_freq_hz
                        width = msg.sonar_samples[i].sonar_ping_channel.w_samples
                        if msg.sonar_samples[i].sonar_ping_channel.w_sonar_type == SONARTYPE_PORT_SIDESCAN:
                            port_intensities = msg.sonar_samples[i].data[:width][::-1]  # Reverse port side for correct left-to-right order
                        elif msg.sonar_samples[i].sonar_ping_channel.w_sonar_type == SONARTYPE_STARBOARD_SIDESCAN:
                            stbd_intensities = msg.sonar_samples[i].data[:width]

                    sonar_data.append((
                        timestamp,
                        len(port_intensities),
                        slant_range,
                        delay_range,
                        freq,
                        speed_of_sound
                    ))

                    acoustic_data[timestamp] = (port_intensities, stbd_intensities)
                elif connection.topic == self.nav_topic:
                    x_velocity, y_velocity, z_velocity = msg.body_velocity.x, msg.body_velocity.y, msg.body_velocity.z
                    speed = np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)

                    nav_data.append((
                        timestamp,
                        msg.global_position.latitude,
                        msg.global_position.longitude,
                        msg.altitude,
                        msg.orientation.roll,
                        msg.orientation.pitch,
                        msg.orientation.yaw,
                        speed
                    ))

        if not odometry_data or not images_metadata or not sonar_data or not nav_data:
            raise ValueError("Error: Missing data in topics. Check your bag topic names.")

        if not self.sonar_file.exists():
            with open(self.sonar_file, "wb") as f:
                pickle.dump(acoustic_data, f)

        odo_timestamps = np.array([o[0] for o in odometry_data])
        for img_ts in images_metadata:
            idx = (np.abs(odo_timestamps - img_ts)).argmin()
            matched_pose = odometry_data[idx][1]
            matched_pose.timestamp = img_ts  # Update timestamp to match image

            image = Image(
                pose=matched_pose.translate(self.camera_trans),
                fx=camera_params[0],
                fy=camera_params[1],
                cx=camera_params[2],
                cy=camera_params[3],
            )
            self.images[img_ts] = image

        nav_timestamps = np.array([n[0] for n in nav_data])
        for sonar_ts, num_samples, slant_range, delay_range, freq, speed_of_sound in sonar_data:
            odo_idx = (np.abs(odo_timestamps - sonar_ts)).argmin()
            nav_idx = (np.abs(nav_timestamps - sonar_ts)).argmin()

            matched_nav = nav_data[nav_idx]
            matched_pose = odometry_data[odo_idx][1]
            matched_pose.timestamp = sonar_ts  # Update timestamp to match sonar

            sonar = SideScanSonar(
                num_samples=num_samples,
                slant_range=slant_range,
                delay_range=delay_range,
                frequency=int(freq),
                speed_of_sound=speed_of_sound,
                navigation=Navigation(
                    pose=matched_pose.translate(self.sonar_trans),
                    latitude=matched_nav[1],
                    longitude=matched_nav[2],
                    altitude=matched_nav[3],
                    roll=matched_nav[4],
                    pitch=matched_nav[5],
                    yaw=matched_nav[6],
                    speed=matched_nav[7]
                )
            )
            self.sonar[sonar_ts] = sonar

    def export_data(self):
        with open(self.cameras_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Image.headers)
            writer.writeheader()
            writer.writerows([image.to_dict() for image in self.images.values()])

        with open(self.sonar_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SideScanSonar.headers)
            writer.writeheader()
            writer.writerows([sonar.to_dict() for sonar in self.sonar.values()])

    def data_stats(self):
        min_x, min_y, min_z, min_alt = np.inf, np.inf, np.inf, np.inf
        max_x, max_y, max_z, max_alt = -np.inf, -np.inf, -np.inf, -np.inf

        for sonar in self.sonar.values():
            nav = sonar.navigation
            pose = sonar.navigation.pose

            if nav.altitude > max_alt: max_alt = nav.altitude
            elif nav.altitude < min_alt: min_alt = nav.altitude

            if pose.x > max_x: max_x = pose.x
            elif pose.x < min_x: min_x = pose.x

            if pose.y > max_y: max_y = pose.y
            elif pose.y < min_y: min_y = pose.y

            if pose.z > max_z: max_z = pose.z
            elif pose.z < min_z: min_z = pose.z

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
