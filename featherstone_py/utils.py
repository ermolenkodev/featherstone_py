from typing import Dict, Any


def take_last(data: Dict[Any, Any]) -> Any:
    return data[next(reversed(data.keys()))]
