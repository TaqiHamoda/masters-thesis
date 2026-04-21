from typing import Self, Dict, Any


class Datatype:
    headers = []

    def __init__(self):
        pass

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Self:
        pass

    def to_dict(self) -> Dict[str, Any]:
        pass