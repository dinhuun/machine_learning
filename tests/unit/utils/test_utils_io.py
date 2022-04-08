from machine_learning.utils.utils_io import (
    read_json,
    read_lines_from_text_file,
    write_json,
    write_lines_to_text_file,
)

d = {"a": 0, "b": 1}
lines = ["line 0", "line 1"]


def test_read_write_json(tmp_path):
    """
    tests read_json(), write_json()
    :param tmp_path: pytest tmp_path fixture
    """
    filepath = str(tmp_path / "d.json")
    write_json(d, filepath)
    d_read = read_json(filepath)
    assert d_read == d


def test_read_write_lines(tmp_path):
    """
    tests read_lines_from_text_file(), write_lines_to_text_file()
    :param tmp_path: pytest tmp_path fixture
    """
    filepath = str(tmp_path / "lines.txt")
    write_lines_to_text_file(lines, filepath)
    lines_read = read_lines_from_text_file(filepath)
    assert lines_read == lines
