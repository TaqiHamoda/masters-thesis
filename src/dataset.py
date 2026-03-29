import cv2, numpy as np

import csv
from rosbags.highlevel import AnyReader
from tabulate import tabulate
from pathlib import Path
from typing import Self, List, Tuple, Dict, Any


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
    headers = ("image_name", "timestamp", "x", "y", "z", "qw", "qx", "qy", "qz", "fx", "fy", "cx", "cy")

    def __init__(self,
        filename: str,
        pose: Pose,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ):
        self.filename = filename

        self.pose = pose

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        return Image(
            filename=data["image_name"],
            pose=Pose(
                data["timestamp"],
                data["x"],
                data["y"],
                data["z"],
                data["qw"],
                data["qx"],
                data["qy"],
                data["qz"]
            ),
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.filename,
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


class Dataset:
    def __init__(self,
        data_path: str,
        output_path: str,
        img_topic: str,
        odo_topic: str,
        info_topic: str,
    ):
        self.bag_paths = [bag for bag in Path(data_path).glob("*.bag")]

        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.csv_file = self.output_path / "camera_poses.csv"

        self.image_dir = self.output_path / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.img_topic = img_topic
        self.odo_topic = odo_topic
        self.info_topic = info_topic

        self.images: List[Image] = []

    def export_data(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=Image.headers)
            writer.writeheader()
            writer.writerows([image.to_dict() for image in self.images])

    def load_data_from_csv(self):
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.images.append(Image.from_dict(row))

    def load_data_from_bags(self):
        camera_params = None
        odometry_data: List[Tuple[int, Pose]] = []
        images_metadata: List[Tuple[int, str]] = []

        with AnyReader(self.bag_paths) as reader:
            # Filter connections once for performance
            connections = [c for c in reader.connections if c.topic in (self.img_topic, self.odo_topic, self.info_topic)]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # The reader now has its own deserialize method
                msg = reader.deserialize(rawdata, connection.msgtype)

                if connection.topic == self.img_topic:
                    filename = f"frame_{timestamp}.jpg"
                    images_metadata.append((timestamp, filename))

                    filepath = self.image_dir / filename
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

        if not odometry_data or not images_metadata:
            raise ValueError("Error: Missing data in topics. Check your bag topic names.")

        odo_timestamps = np.array([o[0] for o in odometry_data])
        for img_ts, img_name in images_metadata:
            idx = (np.abs(odo_timestamps - img_ts)).argmin()
            matched_pose = odometry_data[idx][1]

            image = Image(
                filename=img_name,
                pose=matched_pose,
                fx=camera_params[0],
                fy=camera_params[4],
                cx=camera_params[2],
                cy=camera_params[5]
            )
            self.images.append(image)

    def data_stats(self):
        min_x, min_y, min_z = 0, 0, 0
        max_x, max_y, max_z = 0, 0, 0

        for image in self.images:
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
