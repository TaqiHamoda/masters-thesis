import csv
from pathlib import Path
from typing import Self, List, Dict, Callable, Any


class Datatype:
    headers = []

    def __init__(self):
        pass

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass

    @staticmethod
    def _from_csv(csv_file: str | Path, parser: Callable[[Dict[str, Any]], Self]) -> List[Self]:
        data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(parser(row))

        return data

    @staticmethod
    def to_csv(file_path: Path, data: List[Self]) -> None:
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].headers)
            writer.writeheader()
            writer.writerows([d.to_dict() for d in data])