from typing import Dict, Any


def take_last(data: Dict[Any, Any]) -> Any:
    """
    Returns the last inserted element in the dictionary.
    :param data: Dictionary to take the last element from.
    :return: last element in the dictionary.-
    """
    return data[next(reversed(data.keys()))]
