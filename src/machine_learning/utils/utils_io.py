import json
from typing import Dict, List


def read_json(filepath: str) -> Dict:
    """
    reads json dictionary from filepath
    :param filepath: where json dictionary is
    :return: json dictionary
    """
    with open(filepath) as f:
        return json.load(f)


def write_json(data: Dict, filepath: str):
    """
    writes json dictionary to filepath
    :param data: json dictionary
    :param filepath: where to write to
    :return: none
    """
    with open(filepath, "w") as f:
        json.dump(data, f)


def read_lines_from_text_file(filepath: str) -> List[str]:
    """
    reads lines from text file
    :param filepath: where text file is
    :return: lines
    """
    with open(filepath, "r") as f:
        lines = [location.rstrip() for location in f.readlines()]
    return lines


def write_lines_to_text_file(lines, filepath):
    """
    writes lines to text file
    :param lines: lines
    :param filepath: where text file will be
    :return: none
    """
    with open(filepath, "w") as f:
        for x in lines:
            f.writelines(f"{x}\n")
